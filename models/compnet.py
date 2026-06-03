"""
CompNet model integration for drugrec_benchmark.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
	adj = np.asarray(adj, dtype=np.float32)
	if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
		raise ValueError("adjacency matrix must be square.")
	row_sum = adj.sum(axis=1, keepdims=True)
	row_sum[row_sum == 0] = 1.0
	return adj / row_sum


class CompNetCNN(nn.Module):
	"""CNN encoder used by the original CompNet for diagnosis/procedure codes."""

	def __init__(
		self,
		vocab_size: int,
		emb_size: int,
		num_channels: int,
		hidden_dim: int,
		dropout: float,
	) -> None:
		super().__init__()
		self.embedding = nn.Embedding(vocab_size + 1, emb_size, padding_idx=0)
		self.conv = nn.Sequential(
			nn.Conv1d(emb_size, num_channels, kernel_size=3, stride=2),
			nn.Tanh(),
			nn.Conv1d(num_channels, num_channels, kernel_size=3, stride=2),
			nn.Tanh(),
			nn.Conv1d(num_channels, num_channels, kernel_size=3, stride=2),
		)
		self.dropout = dropout
		self.out = nn.Linear(num_channels, hidden_dim)
		nn.init.kaiming_normal_(self.out.weight)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.dim() == 1:
			x = x.unsqueeze(0)
		x_emb = self.embedding(x).permute(0, 2, 1)
		x_conv = self.conv(x_emb)
		features = F.max_pool1d(x_conv, x_conv.size(2)).squeeze(2)
		features = F.dropout(features, p=self.dropout, training=self.training)
		return self.out(features)


class CompNetGCNLayer(nn.Module):
	def __init__(
		self,
		in_size: int,
		out_size: int,
		total_rel: int = 2,
		n_basis: int = 2,
	) -> None:
		super().__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.total_rel = total_rel
		self.n_basis = n_basis
		self.basis_weights = nn.Parameter(torch.empty(n_basis, in_size, out_size))
		self.basis_coeff = nn.Parameter(torch.empty(total_rel, n_basis))
		self.reset_parameters()

	def reset_parameters(self) -> None:
		nn.init.xavier_uniform_(self.basis_weights)
		nn.init.xavier_uniform_(self.basis_coeff)

	def forward(self, inp: Optional[torch.Tensor], adj_mat_list: torch.Tensor) -> torch.Tensor:
		rel_weights = torch.einsum("ij,jmn->imn", self.basis_coeff, self.basis_weights)
		weights = rel_weights.view(rel_weights.shape[0] * rel_weights.shape[1], rel_weights.shape[2])
		if inp is None:
			tmp = torch.cat([adj for adj in adj_mat_list], dim=1)
		else:
			tmp = torch.cat([torch.matmul(adj, inp) for adj in adj_mat_list], dim=1)
		return torch.matmul(tmp, weights)


class CompNetRGCN(nn.Module):
	def __init__(self, layer_sizes: Sequence[int], total_ent: int) -> None:
		super().__init__()
		self.layer_sizes = list(layer_sizes)
		self.layers = nn.ModuleList()
		in_size = total_ent
		for out_size in self.layer_sizes:
			self.layers.append(CompNetGCNLayer(in_size, out_size))
			in_size = out_size

	def forward(self, adj_mat_list: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		out: Optional[torch.Tensor] = None
		final_rep = []
		for idx, layer in enumerate(self.layers):
			if idx != 0 and out is not None:
				out = F.relu(out)
			out = layer(out, adj_mat_list)
			final_rep.append(out)
		if out is None:
			raise RuntimeError("CompNetRGCN requires at least one layer.")
		return out, torch.cat(final_rep, dim=1)


class CompNetCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		gcn_layers: Sequence[int] = (64,),
		num_channels: int = 128,
		dropout: float = 0.5,
		min_seq_len: int = 15,
	) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.num_drugs = vocab_size[2]
		self.emb_dim = emb_dim
		self.min_seq_len = min_seq_len
		self.diag_cnn = CompNetCNN(vocab_size[0], emb_dim, num_channels, emb_dim, dropout)
		self.proc_cnn = CompNetCNN(vocab_size[1], emb_dim, num_channels, emb_dim, dropout)
		self.rgcn = CompNetRGCN(gcn_layers, self.num_drugs)
		graph_dim = int(sum(gcn_layers))
		self.patient_proj = nn.Sequential(
			nn.Linear(emb_dim * 2, emb_dim),
			nn.Tanh(),
		)
		self.graph_proj = nn.Linear(graph_dim, emb_dim)
		self.state_proj = nn.Sequential(
			nn.Linear(emb_dim * 2, emb_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
		)
		self.output = nn.Linear(emb_dim, self.num_drugs)

		ehr_norm = _normalize_adj(ehr_adj)
		ddi_norm = _normalize_adj(ddi_adj)
		np.fill_diagonal(ehr_norm, 1.0)
		np.fill_diagonal(ddi_norm, 0.0)
		self.register_buffer("adj_mat_list", torch.as_tensor(np.stack([ehr_norm, ddi_norm]), dtype=torch.float32))

	def _pad_codes(self, codes: Sequence[int], vocab_dim: int, device: torch.device) -> torch.Tensor:
		valid = [int(code) + 1 for code in codes if 0 <= int(code) < vocab_dim]
		if len(valid) < self.min_seq_len:
			valid = valid + [0] * (self.min_seq_len - len(valid))
		return torch.as_tensor(valid, dtype=torch.long, device=device)

	def forward(self, visits: Sequence[Sequence[Sequence[int]]], device: torch.device) -> torch.Tensor:
		if not visits:
			return torch.empty((0, self.num_drugs), device=device)
		last_visit = visits[-1]
		diag_codes = last_visit[0] if len(last_visit) > 0 else []
		proc_codes = last_visit[1] if len(last_visit) > 1 else []
		diag_repr = self.diag_cnn(self._pad_codes(diag_codes, self.vocab_size[0], device))
		proc_repr = self.proc_cnn(self._pad_codes(proc_codes, self.vocab_size[1], device))
		patient_repr = self.patient_proj(torch.cat([diag_repr, proc_repr], dim=-1))

		_, drug_graph = self.rgcn(self.adj_mat_list.to(device))
		drug_repr = self.graph_proj(drug_graph)
		att = F.softmax(torch.matmul(drug_repr, patient_repr.squeeze(0)), dim=0)
		graph_context = torch.matmul(att.unsqueeze(0), drug_repr)
		state = self.state_proj(torch.cat([patient_repr, graph_context], dim=-1))
		return self.output(state)


class CompNet(BaseDrugRecommendationModel):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		gcn_layers: Sequence[int] = (64,),
		num_channels: int = 128,
		dropout: float = 0.5,
		threshold: float = 0.5,
		ddi_weight: float = 0.0,
		ddi_scale: float = 1e-5,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.threshold = threshold
		self.ddi_weight = ddi_weight
		self.ddi_scale = ddi_scale
		self.ddi_adj = np.asarray(ddi_adj, dtype=np.float32)
		self.register_buffer("tensor_ddi_adj", torch.as_tensor(self.ddi_adj, dtype=torch.float32, device=self.device))
		self.core = CompNetCore(
			vocab_size=vocab_size,
			ehr_adj=ehr_adj,
			ddi_adj=ddi_adj,
			emb_dim=emb_dim,
			gcn_layers=gcn_layers,
			num_channels=num_channels,
			dropout=dropout,
		)

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		if not patient:
			return {"logits": torch.empty((0, self.vocab_size[2]), device=self.device)}
		return {"logits": self.core(patient, self.device)}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)
		targets = self.build_target(batch)
		loss = F.binary_cross_entropy_with_logits(logits, targets)
		if self.ddi_weight > 0:
			probs = torch.sigmoid(logits)
			pair_prob = torch.matmul(probs.t(), probs)
			loss = loss + self.ddi_weight * self.ddi_scale * pair_prob.mul(self.tensor_ddi_adj).sum()
		return loss

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		return (torch.sigmoid(logits) >= self.threshold).float()

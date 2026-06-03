"""
GAMENet model implementation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel

class GraphConvolution(nn.Module):
	"""
	Simple GCN layer.
	"""

	def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter("bias", None)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		stdv = 1.0 / np.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, inputs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		support = torch.mm(inputs, self.weight)
		output = torch.mm(adj, support)
		if self.bias is not None:
			return output + self.bias
		return output


class GCN(nn.Module):
	def __init__(self, vocab_size: int, emb_dim: int, adj: np.ndarray, device: torch.device) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.emb_dim = emb_dim
		self.device = device

		adj = self._normalize(adj + np.eye(adj.shape[0]))
		self.adj = torch.FloatTensor(adj).to(device)
		self.x = torch.eye(vocab_size).to(device)

		self.gcn1 = GraphConvolution(vocab_size, emb_dim)
		self.dropout = nn.Dropout(p=0.3)
		self.gcn2 = GraphConvolution(emb_dim, emb_dim)

	def _normalize(self, mx: np.ndarray) -> np.ndarray:
		rowsum = np.array(mx.sum(1))
		r_inv = np.power(rowsum, -1).flatten()
		r_inv[np.isinf(r_inv)] = 0.0
		r_mat_inv = np.diagflat(r_inv)
		return r_mat_inv.dot(mx)

	def forward(self) -> torch.Tensor:
		node_embedding = self.gcn1(self.x, self.adj)
		node_embedding = F.relu(node_embedding)
		node_embedding = self.dropout(node_embedding)
		node_embedding = self.gcn2(node_embedding, self.adj)
		return node_embedding


class GAMENetCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		ddi_in_memory: bool = True,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.device = device
		self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
		self.ddi_in_memory = ddi_in_memory

		self.embeddings = nn.ModuleList(
			[nn.Embedding(vocab_size[i], emb_dim) for i in range(2)]
		)
		self.dropout = nn.Dropout(p=0.5)
		self.encoders = nn.ModuleList(
			[nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(2)]
		)

		self.query = nn.Sequential(
			nn.ReLU(),
			nn.Linear(emb_dim * 4, emb_dim),
		)

		self.ehr_gcn = GCN(vocab_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
		self.ddi_gcn = GCN(vocab_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
		self.inter = nn.Parameter(torch.FloatTensor(1))

		self.output = nn.Sequential(
			nn.ReLU(),
			nn.Linear(emb_dim * 3, emb_dim * 2),
			nn.ReLU(),
			nn.Linear(emb_dim * 2, vocab_size[2]),
		)

		self._init_weights()

	def _init_weights(self) -> None:
		initrange = 0.1
		for item in self.embeddings:
			item.weight.data.uniform_(-initrange, initrange)
		self.inter.data.uniform_(-initrange, initrange)

	def forward(self, visits: Sequence[Sequence[Sequence[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
		i1_seq: List[torch.Tensor] = []
		i2_seq: List[torch.Tensor] = []

		def mean_embedding(embedding: torch.Tensor) -> torch.Tensor:
			if embedding.size(1) == 0:
				return torch.zeros((1, 1, embedding.size(2)), device=embedding.device)
			return embedding.mean(dim=1, keepdim=True)

		for adm in visits:
			i1 = mean_embedding(
				self.dropout(
					self.embeddings[0](
						torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
					)
				)
			)
			i2 = mean_embedding(
				self.dropout(
					self.embeddings[1](
						torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
					)
				)
			)
			i1_seq.append(i1)
			i2_seq.append(i2)

		i1_seq = torch.cat(i1_seq, dim=1)
		i2_seq = torch.cat(i2_seq, dim=1)

		o1, _ = self.encoders[0](i1_seq)
		o2, _ = self.encoders[1](i2_seq)
		patient_repr = torch.cat([o1, o2], dim=-1).squeeze(dim=0)
		queries = self.query(patient_repr)

		query = queries[-1:]
		if self.ddi_in_memory:
			drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter
		else:
			drug_memory = self.ehr_gcn()

		if len(visits) > 1:
			history_keys = queries[: (queries.size(0) - 1)]
			history_values = np.zeros((len(visits) - 1, self.vocab_size[2]))
			for idx, adm in enumerate(visits):
				if idx == len(visits) - 1:
					break
				history_values[idx, adm[2]] = 1
			history_values = torch.FloatTensor(history_values).to(self.device)

		key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)
		fact1 = torch.mm(key_weights1, drug_memory)

		if len(visits) > 1:
			visit_weight = F.softmax(torch.mm(query, history_keys.t()), dim=-1)
			weighted_values = visit_weight.mm(history_values)
			fact2 = torch.mm(weighted_values, drug_memory)
		else:
			fact2 = fact1

		output = self.output(torch.cat([query, fact1, fact2], dim=-1))
		neg_pred_prob = torch.sigmoid(output)
		neg_pred_prob = neg_pred_prob.t() * neg_pred_prob
		batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

		return output, batch_neg


class GAMENet(BaseDrugRecommendationModel):
	"""
	GAMENet wrapper integrated with the benchmark interfaces.

	Expected batch format:
	{
		"visits": List[List[Tuple[List[int], List[int], List[int]]]],
		"labels": Tensor
	}
	"""

	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		ddi_in_memory: bool = True,
		target_ddi: float = 0.06,
		temperature: float = 2.0,
		decay_weight: float = 0.85,
		threshold: float = 0.5,
		use_ddi: bool = True,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.target_ddi = target_ddi
		self.temperature = temperature
		self.decay_weight = decay_weight
		self.threshold = threshold
		self.use_ddi = use_ddi
		self.ddi_adj = ddi_adj

		self.core = GAMENetCore(
			vocab_size=vocab_size,
			ehr_adj=ehr_adj,
			ddi_adj=ddi_adj,
			emb_dim=emb_dim,
			ddi_in_memory=ddi_in_memory,
			device=self.device,
		)

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		if not patient:
			return {
				"logits": torch.empty((0, self.vocab_size[2]), device=self.device),
				"ddi_loss": torch.empty((0,), device=self.device),
			}
		logits, ddi_penalty = self.core(patient)
		return {"logits": logits, "ddi_loss": ddi_penalty.unsqueeze(0)}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		patient = self.get_patient(batch)
		logits = outputs["logits"]
		ddi_penalty = outputs["ddi_loss"]
		targets = self.build_target(batch)
		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)

		target_multi = self.build_multilabel_target(targets)

		loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
		loss_multi = F.multilabel_margin_loss(torch.sigmoid(logits), target_multi)
		if not self.use_ddi:
			return 0.9 * loss_bce + 0.1 * loss_multi

		pred_labels = self._predict_labels(logits)
		current_ddi_rate = self._ddi_rate_from_labels(pred_labels)
		if current_ddi_rate <= self.target_ddi:
			return 0.9 * loss_bce + 0.1 * loss_multi

		rnd = np.exp((self.target_ddi - current_ddi_rate) / self.temperature)
		if np.random.rand(1) < rnd:
			return ddi_penalty[0]
		return 0.9 * loss_bce + 0.1 * loss_multi

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		probs = torch.sigmoid(logits)
		return (probs >= self.threshold).float()

	def on_epoch_end(self) -> None:
		self.temperature *= self.decay_weight

	def _build_target(self, patient: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		return self.build_target({"visit": patient})

	def _build_multilabel_target(self, target_bce: torch.Tensor) -> torch.Tensor:
		return self.build_multilabel_target(target_bce)

	def _predict_labels(self, logits: torch.Tensor) -> List[int]:
		probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
		preds = (probs >= self.threshold).astype(np.int32)
		return np.where(preds == 1)[0].tolist()

	def _ddi_rate_from_labels(self, labels: List[int]) -> float:
		if len(labels) < 2:
			return 0.0
		ddi_count = 0
		total_count = 0
		for i, med_i in enumerate(labels):
			for j in range(i + 1, len(labels)):
				med_j = labels[j]
				total_count += 1
				if self.ddi_adj[med_i, med_j] == 1 or self.ddi_adj[med_j, med_i] == 1:
					ddi_count += 1
		if total_count == 0:
			return 0.0
		return ddi_count / total_count

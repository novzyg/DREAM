"""
SafeDrug model implementation integrated with the benchmark base model.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel

class MaskLinear(nn.Module):
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

	def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		weight = torch.mul(self.weight, mask)
		output = torch.mm(inputs, weight)
		if self.bias is not None:
			return output + self.bias
		return output


class MolecularGraphNeuralNetwork(nn.Module):
	def __init__(self, n_fingerprint: int, dim: int, layer_hidden: int, device: torch.device) -> None:
		super().__init__()
		self.device = device
		self.embed_fingerprint = nn.Embedding(n_fingerprint, dim).to(self.device)
		self.W_fingerprint = nn.ModuleList(
			[nn.Linear(dim, dim).to(self.device) for _ in range(layer_hidden)]
		)
		self.layer_hidden = layer_hidden

	def pad(self, matrices: Sequence[torch.Tensor], pad_value: float) -> torch.Tensor:
		shapes = [m.shape for m in matrices]
		rows, cols = sum(s[0] for s in shapes), sum(s[1] for s in shapes)
		zeros = torch.FloatTensor(np.zeros((rows, cols))).to(self.device)
		pad_matrices = pad_value + zeros
		i, j = 0, 0
		for k, matrix in enumerate(matrices):
			m, n = shapes[k]
			pad_matrices[i : i + m, j : j + n] = matrix
			i += m
			j += n
		return pad_matrices

	def update(self, matrix: torch.Tensor, vectors: torch.Tensor, layer: int) -> torch.Tensor:
		hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
		return hidden_vectors + torch.mm(matrix, hidden_vectors)

	def sum(self, vectors: torch.Tensor, axis: Iterable[int]) -> torch.Tensor:
		sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
		return torch.stack(sum_vectors)

	def forward(
		self,
		inputs: Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Sequence[int]],
	) -> torch.Tensor:
		fingerprints, adjacencies, molecular_sizes = inputs
		fingerprints = torch.cat(list(fingerprints))
		adjacencies = self.pad(list(adjacencies), 0)

		fingerprint_vectors = self.embed_fingerprint(fingerprints)
		for layer in range(self.layer_hidden):
			fingerprint_vectors = self.update(adjacencies, fingerprint_vectors, layer)

		molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
		return molecular_vectors


class SafeDrugCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ddi_adj: np.ndarray,
		ddi_mask_h: np.ndarray,
		mpnn_set: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
		n_fingerprints: int,
		average_projection: torch.Tensor,
		emb_dim: int = 256,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.device = device

		self.embeddings = nn.ModuleList(
			[nn.Embedding(vocab_size[i], emb_dim) for i in range(2)]
		)
		self.dropout = nn.Dropout(p=0.5)
		self.encoders = nn.ModuleList(
			[nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)]
		)
		self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))

		self.bipartite_transform = nn.Sequential(nn.Linear(emb_dim, ddi_mask_h.shape[1]))
		self.bipartite_output = MaskLinear(ddi_mask_h.shape[1], vocab_size[2], False)

		self.mpnn_molecule_set = list(zip(*mpnn_set))
		with torch.no_grad():
			mpnn_emb = MolecularGraphNeuralNetwork(
				n_fingerprints, emb_dim, layer_hidden=2, device=device
			).forward(self.mpnn_molecule_set)
			mpnn_emb = torch.mm(
				average_projection.to(device=self.device),
				mpnn_emb.to(device=self.device),
			)
		self.register_buffer("mpnn_emb", mpnn_emb)
		self.mpnn_output = nn.Linear(vocab_size[2], vocab_size[2])
		self.mpnn_layernorm = nn.LayerNorm(vocab_size[2])

		self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
		self.tensor_ddi_mask_h = torch.FloatTensor(ddi_mask_h).to(device)
		self.init_weights()

	def forward(self, visits: Sequence[Sequence[Sequence[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
		i1_seq: List[torch.Tensor] = []
		i2_seq: List[torch.Tensor] = []

		def sum_embedding(embedding: torch.Tensor) -> torch.Tensor:
			return embedding.sum(dim=1).unsqueeze(dim=0)

		for adm in visits:
			i1 = sum_embedding(
				self.dropout(
					self.embeddings[0](
						torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
					)
				)
			)
			i2 = sum_embedding(
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
		query = self.query(patient_repr)[-1:, :]

		mpnn_match = torch.sigmoid(torch.mm(query, self.mpnn_emb.t()))
		mpnn_att = self.mpnn_layernorm(mpnn_match + self.mpnn_output(mpnn_match))

		bipartite_emb = self.bipartite_output(
			torch.sigmoid(self.bipartite_transform(query)),
			self.tensor_ddi_mask_h.t(),
		)

		logits = torch.mul(bipartite_emb, mpnn_att)

		neg_pred_prob = torch.sigmoid(logits)
		neg_pred_prob = neg_pred_prob.t() * neg_pred_prob
		batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

		return logits, batch_neg

	def init_weights(self) -> None:
		initrange = 0.1
		for item in self.embeddings:
			item.weight.data.uniform_(-initrange, initrange)


class SafeDrug(BaseDrugRecommendationModel):
	"""
	SafeDrug wrapper integrated with the benchmark interfaces.

	Expected batch format:
	{
		"visits": List[List[Tuple[List[int], List[int], List[int]]]]
	}

	Training/validation targets use the last admission per patient.
	"""

	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ddi_adj: np.ndarray,
		ddi_mask_h: np.ndarray,
		mpnn_set: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
		n_fingerprints: int,
		average_projection: torch.Tensor,
		emb_dim: int = 256,
		target_ddi: float = 0.06,
		kp: float = 0.05,
		threshold: float = 0.5,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.target_ddi = target_ddi
		self.kp = kp
		self.threshold = threshold
		self.ddi_adj = ddi_adj

		self.core = SafeDrugCore(
			vocab_size=vocab_size,
			ddi_adj=ddi_adj,
			ddi_mask_h=ddi_mask_h,
			mpnn_set=mpnn_set,
			n_fingerprints=n_fingerprints,
			average_projection=average_projection,
			emb_dim=emb_dim,
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
		return {
			"logits": logits,
			"ddi_loss": ddi_penalty.unsqueeze(0),
		}

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
		pred_labels = self._predict_labels(logits)
		current_ddi_rate = self._ddi_rate_from_labels(pred_labels)

		if current_ddi_rate <= self.target_ddi:
			return 0.95 * loss_bce + 0.05 * loss_multi

		beta = min(0.0, 1.0 + (self.target_ddi - current_ddi_rate) / self.kp)
		return beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * ddi_penalty[0]

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		probs = torch.sigmoid(logits)
		return (probs >= self.threshold).float()

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

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


def build_projection_and_smiles(
	molecule: Dict[str, Sequence[str]],
	med_voc: Dict[int, str],
) -> Tuple[torch.Tensor, List[str]]:
	try:
		from rdkit import Chem
	except ImportError as exc:
		raise ImportError("RDKit is required for MoleRec data preprocessing.") from exc

	average_index: List[int] = []
	smiles_all: List[str] = []

	for _, ndc in med_voc.items():
		smiles_list = list(molecule[ndc])
		counter = 0
		for smiles in smiles_list:
			mol = Chem.MolFromSmiles(smiles)
			if mol is not None:
				smiles_all.append(smiles)
				counter += 1
		average_index.append(counter)

	n_col = sum(average_index)
	n_row = len(average_index)
	average_projection = np.zeros((n_row, n_col), dtype=np.float32)

	col_counter = 0
	for i, item in enumerate(average_index):
		if item > 0:
			average_projection[i, col_counter : col_counter + item] = 1.0 / float(item)
		col_counter += item

	return torch.FloatTensor(average_projection), smiles_all


def graph_batch_from_smile(smiles_list: Sequence[str]) -> Any:
	try:
		from ogb.utils import smiles2graph
		from torch_geometric.data import Data
	except ImportError as exc:
		raise ImportError(
			"MoleRec graph encoding requires ogb and torch-geometric. "
			"Install packages or set model.use_embedding=true and provide precomputed graph features."
		) from exc

	edge_idxes: List[np.ndarray] = []
	edge_feats: List[np.ndarray] = []
	node_feats: List[np.ndarray] = []
	batch: List[np.ndarray] = []
	last_node = 0

	graphs = [smiles2graph(x) for x in smiles_list]
	for idx, graph in enumerate(graphs):
		edge_idxes.append(graph["edge_index"] + last_node)
		edge_feats.append(graph["edge_feat"])
		node_feats.append(graph["node_feat"])
		last_node += int(graph["num_nodes"])
		batch.append(np.ones(int(graph["num_nodes"]), dtype=np.int64) * idx)

	result = {
		"edge_index": np.concatenate(edge_idxes, axis=-1),
		"edge_attr": np.concatenate(edge_feats, axis=0),
		"batch": np.concatenate(batch, axis=0),
		"x": np.concatenate(node_feats, axis=0),
	}
	result_tensor = {k: torch.from_numpy(v) for k, v in result.items()}
	result_tensor["num_nodes"] = last_node
	return Data(**result_tensor)


class MAB(nn.Module):
	def __init__(
		self,
		qdim: int,
		kdim: int,
		vdim: int,
		number_heads: int,
		use_ln: bool = False,
	) -> None:
		super().__init__()
		self.vdim = vdim
		self.number_heads = number_heads
		if self.vdim % self.number_heads != 0:
			raise ValueError("Feature dim should be divisible by number_heads.")

		self.q_dense = nn.Linear(qdim, self.vdim)
		self.k_dense = nn.Linear(kdim, self.vdim)
		self.v_dense = nn.Linear(kdim, self.vdim)
		self.o_dense = nn.Linear(self.vdim, self.vdim)

		self.use_ln = use_ln
		if self.use_ln:
			self.ln1 = nn.LayerNorm(self.vdim)
			self.ln2 = nn.LayerNorm(self.vdim)

	def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		q, k, v = self.q_dense(x), self.k_dense(y), self.v_dense(y)
		batch_size = q.shape[0]
		dim_split = self.vdim // self.number_heads

		q_split = torch.cat(q.split(dim_split, dim=2), dim=0)
		k_split = torch.cat(k.split(dim_split, dim=2), dim=0)
		v_split = torch.cat(v.split(dim_split, dim=2), dim=0)

		attn = torch.matmul(q_split, k_split.transpose(1, 2))
		attn = torch.softmax(attn / math.sqrt(dim_split), dim=-1)
		o = q_split + torch.matmul(attn, v_split)
		o = torch.cat(o.split(batch_size, dim=0), dim=2)

		o = o if not self.use_ln else self.ln1(o)
		o = self.o_dense(o)
		o = o if not self.use_ln else self.ln2(o)
		return o


class SAB(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, number_heads: int, use_ln: bool = False) -> None:
		super().__init__()
		self.net = MAB(in_dim, in_dim, out_dim, number_heads, use_ln)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x, x)


class AdjAttenAgger(nn.Module):
	def __init__(self, qdim: int, kdim: int, mid_dim: int) -> None:
		super().__init__()
		self.model_dim = mid_dim
		self.q_dense = nn.Linear(qdim, mid_dim)
		self.k_dense = nn.Linear(kdim, mid_dim)

	def forward(
		self,
		main_feat: torch.Tensor,
		other_feat: torch.Tensor,
		fix_feat: torch.Tensor,
		mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		q = self.q_dense(main_feat)
		k = self.k_dense(other_feat)
		attn = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(self.model_dim)
		if mask is not None:
			attn = torch.masked_fill(attn, mask, -(1 << 32))
		attn = torch.softmax(attn, dim=-1)
		fix_feat = torch.diag(fix_feat)
		other_feat = torch.matmul(fix_feat, other_feat)
		return torch.matmul(attn, other_feat)


class MoleRecCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		emb_dim: int,
		substruct_num: int,
		ddi_adj: np.ndarray,
		ddi_mask_h: np.ndarray,
		mol_graphs: Any,
		average_projection: torch.Tensor,
		substruct_graphs: Optional[Any],
		use_embedding: bool,
		dropout: float,
		device: torch.device,
		gnn_num_layer: int,
		gnn_type: str,
		graph_pooling: str,
		virtual_node: bool,
	) -> None:
		super().__init__()
		self.device = device
		self.use_embedding = use_embedding

		try:
			from torch_geometric.nn import (
				GlobalAttention,
				Set2Set,
				global_add_pool,
				global_max_pool,
				global_mean_pool,
			)
		except ImportError as exc:
			raise ImportError(
				"MoleRec requires torch-geometric to build graph encoders."
			) from exc

		try:
			from drugrec_benchmark.utils.modules.GNNConv import GNN_node, GNN_node_Virtualnode
		except ImportError as exc:
			raise ImportError(
				"Cannot import MoleRec GNN layers from models/MoleRec/src. "
				"Please ensure the original MoleRec source tree exists in the workspace."
			) from exc

		class GNNGraph(nn.Module):
			def __init__(self) -> None:
				super().__init__()
				if virtual_node:
					self.gnn_node = GNN_node_Virtualnode(
						gnn_num_layer,
						emb_dim,
						JK="last",
						drop_ratio=dropout,
						residual=False,
						gnn_type=gnn_type,
					)
				else:
					self.gnn_node = GNN_node(
						gnn_num_layer,
						emb_dim,
						JK="last",
						drop_ratio=dropout,
						residual=False,
						gnn_type=gnn_type,
					)

				if graph_pooling == "sum":
					self.pool = global_add_pool
				elif graph_pooling == "mean":
					self.pool = global_mean_pool
				elif graph_pooling == "max":
					self.pool = global_max_pool
				elif graph_pooling == "attention":
					self.pool = GlobalAttention(
						gate_nn=nn.Sequential(
							nn.Linear(emb_dim, 2 * emb_dim),
							nn.BatchNorm1d(2 * emb_dim),
							nn.ReLU(),
							nn.Linear(2 * emb_dim, 1),
						)
					)
				elif graph_pooling == "set2set":
					self.pool = Set2Set(emb_dim, processing_steps=2)
				else:
					raise ValueError("Invalid graph pooling type for MoleRec.")

			def forward(self, batched_data: Any) -> torch.Tensor:
				h_node = self.gnn_node(batched_data)
				return self.pool(h_node, batched_data.batch)

		self.global_encoder = GNNGraph()
		if use_embedding:
			self.substruct_emb = nn.Parameter(torch.zeros(substruct_num, emb_dim))
			nn.init.xavier_uniform_(self.substruct_emb)
			self.substruct_encoder = None
		else:
			self.substruct_encoder = GNNGraph()
			self.substruct_emb = None

		self.embeddings = nn.ModuleList(
			[nn.Embedding(vocab_size[0], emb_dim), nn.Embedding(vocab_size[1], emb_dim)]
		)
		self.seq_encoders = nn.ModuleList(
			[nn.GRU(emb_dim, emb_dim, batch_first=True), nn.GRU(emb_dim, emb_dim, batch_first=True)]
		)
		self.rnn_dropout = nn.Dropout(p=dropout) if 0 < dropout < 1 else nn.Sequential()
		self.sab = SAB(emb_dim, emb_dim, 2, use_ln=True)
		self.query = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 4, emb_dim))
		self.substruct_rela = nn.Linear(emb_dim, substruct_num)
		self.aggregator = AdjAttenAgger(emb_dim, emb_dim, emb_dim)
		self.score_extractor = nn.Sequential(
			nn.Linear(emb_dim, emb_dim // 2),
			nn.ReLU(),
			nn.Linear(emb_dim // 2, 1),
		)

		self.register_buffer("tensor_ddi_adj", torch.FloatTensor(ddi_adj))
		self.register_buffer("tensor_ddi_mask_h", torch.FloatTensor(ddi_mask_h))
		self.register_buffer("average_projection", average_projection.float())

		self.mol_graphs = mol_graphs
		self.substruct_graphs = substruct_graphs
		self._init_weights()

	def _init_weights(self) -> None:
		initrange = 0.1
		for item in self.embeddings:
			item.weight.data.uniform_(-initrange, initrange)

	def _encode_drug_space(self) -> Tuple[torch.Tensor, torch.Tensor]:
		global_embeddings = self.global_encoder(self.mol_graphs)
		global_embeddings = torch.mm(self.average_projection, global_embeddings)

		if self.use_embedding:
			substruct_embeddings = self.sab(self.substruct_emb.unsqueeze(0)).squeeze(0)
		else:
			if self.substruct_encoder is None or self.substruct_graphs is None:
				raise RuntimeError("Substructure encoder or graphs are missing for MoleRec.")
			substruct_embeddings = self.sab(
				self.substruct_encoder(self.substruct_graphs).unsqueeze(0)
			).squeeze(0)

		return global_embeddings, substruct_embeddings

	def _encode_patient_query(self, patient_data: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		seq1: List[torch.Tensor] = []
		seq2: List[torch.Tensor] = []
		for adm in patient_data:
			idx1 = torch.LongTensor([adm[0]]).to(self.device)
			idx2 = torch.LongTensor([adm[1]]).to(self.device)
			repr1 = self.rnn_dropout(self.embeddings[0](idx1))
			repr2 = self.rnn_dropout(self.embeddings[1](idx2))
			seq1.append(torch.sum(repr1, keepdim=True, dim=1))
			seq2.append(torch.sum(repr2, keepdim=True, dim=1))

		if not seq1:
			return torch.zeros(self.query[1].out_features, device=self.device)

		seq1_tensor = torch.cat(seq1, dim=1)
		seq2_tensor = torch.cat(seq2, dim=1)
		output1, hidden1 = self.seq_encoders[0](seq1_tensor)
		output2, hidden2 = self.seq_encoders[1](seq2_tensor)

		seq_repr = torch.cat([hidden1, hidden2], dim=-1)
		last_repr = torch.cat([output1[:, -1], output2[:, -1]], dim=-1)
		patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
		return self.query(patient_repr)

	def _score_queries(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if queries.numel() == 0:
			return (
				torch.empty((0, self.tensor_ddi_adj.shape[0]), device=self.device),
				torch.empty((0,), device=self.device),
			)

		global_embeddings, substruct_embeddings = self._encode_drug_space()
		scores: List[torch.Tensor] = []
		ddi_penalties: List[torch.Tensor] = []
		for query in queries:
			substruct_weight = torch.sigmoid(self.substruct_rela(query))
			molecule_embeddings = self.aggregator(
				global_embeddings,
				substruct_embeddings,
				substruct_weight,
				mask=torch.logical_not(self.tensor_ddi_mask_h > 0),
			)
			score = self.score_extractor(molecule_embeddings).t()
			neg_pred_prob = torch.sigmoid(score)
			neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
			ddi_penalties.append(0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum())
			scores.append(score)
		return torch.cat(scores, dim=0), torch.stack(ddi_penalties)

	def forward_many(self, patients: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
		queries = torch.stack([self._encode_patient_query(patient) for patient in patients], dim=0)
		return self._score_queries(queries)

	def forward(self, patient_data: Sequence[Sequence[Sequence[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
		scores, ddi_penalties = self.forward_many([patient_data])
		return scores, ddi_penalties[0]



class MoleRec(BaseDrugRecommendationModel):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ddi_adj: np.ndarray,
		ddi_mask_h: np.ndarray,
		mol_graphs: Any,
		average_projection: torch.Tensor,
		substruct_graphs: Optional[Any],
		emb_dim: int = 64,
		target_ddi: float = 0.06,
		coef: float = 2.5,
		threshold: float = 0.5,
		use_embedding: bool = False,
		dropout: float = 0.7,
		gnn_num_layer: int = 4,
		gnn_type: str = "gin",
		graph_pooling: str = "mean",
		virtual_node: bool = False,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.supports_batched_training = True
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.ddi_adj = ddi_adj
		self.target_ddi = float(target_ddi)
		self.coef = float(coef)
		self.threshold = float(threshold)

		self.core = MoleRecCore(
			vocab_size=vocab_size,
			emb_dim=emb_dim,
			substruct_num=ddi_mask_h.shape[1],
			ddi_adj=ddi_adj,
			ddi_mask_h=ddi_mask_h,
			mol_graphs=mol_graphs,
			average_projection=average_projection,
			substruct_graphs=substruct_graphs,
			use_embedding=use_embedding,
			dropout=dropout,
			device=self.device,
			gnn_num_layer=gnn_num_layer,
			gnn_type=gnn_type,
			graph_pooling=graph_pooling,
			virtual_node=virtual_node,
		)

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patients = self.get_patients(batch)
		if not patients:
			return {
				"logits": torch.empty((0, self.vocab_size[2]), device=self.device),
				"ddi_loss": torch.empty((0,), device=self.device),
			}
		logits, ddi_penalty = self.core.forward_many(patients)
		return {
			"logits": logits,
			"ddi_loss": ddi_penalty,
		}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		ddi_penalty = outputs["ddi_loss"].reshape(-1)
		targets = self.build_target(batch)
		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)
		if targets.shape[0] != logits.shape[0]:
			raise ValueError(
				f"MoleRec target batch size {targets.shape[0]} does not match logits batch size {logits.shape[0]}."
			)

		target_multi = self.build_multilabel_target(targets)
		loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
		loss_multi = F.multilabel_margin_loss(torch.sigmoid(logits), target_multi)

		pred_label_rows = self._predict_label_rows(logits)
		ddi_rates = [self._ddi_rate_from_labels(row) for row in pred_label_rows]
		current_ddi_rate = float(np.mean(ddi_rates)) if ddi_rates else 0.0

		if current_ddi_rate <= self.target_ddi:
			return 0.95 * loss_bce + 0.05 * loss_multi

		beta = self.coef * (1.0 - (current_ddi_rate / self.target_ddi))
		beta = min(math.exp(beta), 1.0)
		return beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1.0 - beta) * ddi_penalty.mean()

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

	def _predict_label_rows(self, logits: torch.Tensor) -> List[List[int]]:
		probs = torch.sigmoid(logits).detach().cpu().numpy()
		preds = (probs >= self.threshold).astype(np.int32)
		return [np.where(row == 1)[0].tolist() for row in preds]

	def _predict_labels(self, logits: torch.Tensor) -> List[int]:
		rows = self._predict_label_rows(logits)
		return rows[0] if rows else []

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
		return 0.0 if total_count == 0 else float(ddi_count) / float(total_count)

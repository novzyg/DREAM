"""
DRecHGR model implementation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel

try:
	from dgl.nn.pytorch import GATConv
	import dgl
	_DGL_AVAILABLE = True
except ImportError:
	_DGL_AVAILABLE = False
	GATConv = None
	dgl = None


def _normalize_adj(mat: sp.spmatrix) -> sp.coo_matrix:
	"""Symmetric normalization D^{-1/2} A D^{-1/2}."""
	degree = np.array(mat.sum(axis=-1)).flatten()
	d_inv_sqrt = np.power(degree, -0.5)
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
	d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
	return (mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat)).tocoo()


def _sparse_to_torch(mat: sp.spmatrix, device: torch.device) -> torch.Tensor:
	"""Convert scipy sparse matrix to torch sparse tensor."""
	mat = mat.tocoo()
	idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
	vals = torch.from_numpy(mat.data.astype(np.float32))
	shape = torch.Size(mat.shape)
	return torch.sparse_coo_tensor(idxs, vals, shape, dtype=torch.float32, device=device)


def build_hetero_adj(
	records: Sequence[Sequence[Sequence[int]]],
	n_patients: int,
	n_items: int,
	item_idx: int,
	device: torch.device,
) -> torch.Tensor:
	"""
	Build patient-item heterogeneous adjacency matrix.
	item_idx: 0 for diagnosis, 2 for medication (in the [diag, proc, med] tuple).
	"""
	rows: List[int] = []
	cols: List[int] = []
	for p_idx, patient in enumerate(records):
		for visit in patient:
			if len(visit) > item_idx:
				for item in visit[item_idx]:
					if 0 <= item < n_items:
						rows.append(p_idx)
						cols.append(item)
	data = np.ones(len(rows), dtype=np.float32)
	hetero_mat = sp.csr_matrix(
		(data, (rows, cols)), shape=(n_patients, n_items), dtype=np.float32
	)
	top_left = sp.csr_matrix((n_patients, n_patients), dtype=np.float32)
	top_right = hetero_mat
	bottom_left = hetero_mat.transpose().tocsr()
	bottom_right = sp.csr_matrix((n_items, n_items), dtype=np.float32)
	full_adj = sp.vstack(
		[sp.hstack([top_left, top_right]), sp.hstack([bottom_left, bottom_right])]
	)
	full_adj = (full_adj != 0).astype(np.float32)
	full_adj = full_adj + sp.eye(full_adj.shape[0], dtype=np.float32)
	full_adj = _normalize_adj(full_adj)
	return _sparse_to_torch(full_adj, device)


def build_meta_path_edges(
	hetero_mat: sp.csr_matrix,
	threshold: int = 5,
) -> np.ndarray:
	"""
	Build meta-path adjacency edges from heterogeneous matrix.
	Computes H @ H.T to get patient-patient co-occurrence counts,
	then thresholds to get edges for DGL graph construction.
	"""
	meta = hetero_mat @ hetero_mat.T
	meta.data[meta.data < threshold] = 0
	meta.eliminate_zeros()
	meta = meta + meta.T
	meta.data = np.ones_like(meta.data)
	meta = meta.tocoo()
	edges = np.vstack([meta.row, meta.col]).T.astype(np.int64)
	return edges


def _row_normalize(mat: sp.csr_matrix) -> sp.csr_matrix:
	"""Row-normalize sparse matrix."""
	rowsum = np.array(mat.sum(1)).flatten()
	r_inv = np.power(rowsum, -1.0)
	r_inv[np.isinf(r_inv)] = 0.0
	r_mat_inv = sp.diags(r_inv)
	return r_mat_inv.dot(mat)


def build_meta_path_graph(
	edges: np.ndarray,
	n_patients: int,
	device: torch.device,
) -> "dgl.DGLGraph":
	"""
	Build a DGL graph from meta-path edges.
	Applies symmetrization, normalization, and self-loops.
	"""
	if not _DGL_AVAILABLE:
		raise ImportError("DGL is required. Install with: pip install dgl")
	adj = sp.csr_matrix(
		(np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])),
		shape=(n_patients, n_patients),
		dtype=np.float32,
	)
	adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
	adj = _row_normalize(adj + sp.eye(adj.shape[0], dtype=np.float32))
	adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
	choice = np.where(adj.toarray() != 0)
	edge_arr = np.array(list(map(list, zip(*choice))))
	graph = dgl.graph((edge_arr[:, 0], edge_arr[:, 1]), num_nodes=n_patients)
	graph = dgl.add_self_loop(graph)
	return graph.to(device)


def build_dgl_graph_from_adj(
	adj: np.ndarray,
	device: torch.device,
) -> "dgl.DGLGraph":
	"""Build DGL graph from adjacency matrix (for ddi/ehr)."""
	if not _DGL_AVAILABLE:
		raise ImportError("DGL is required. Install with: pip install dgl")
	choice = np.where(adj != 0)
	edges = np.array(list(map(list, zip(*choice))))
	num_nodes = adj.shape[0]
	graph = dgl.graph(
		(edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)),
		num_nodes=num_nodes,
	)
	graph = dgl.add_self_loop(graph)
	return graph.to(device)


class GCNLayer(nn.Module):
	"""Simple GCN layer with sparse adjacency matrix support."""

	def __init__(self) -> None:
		super().__init__()
		self.act = nn.LeakyReLU(negative_slope=0.5)

	def forward(self, adj: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
		return self.act(torch.spmm(adj, embeds))


class SemanticAttention(nn.Module):
	"""Semantic attention for aggregating multiple meta-path embeddings."""

	def __init__(self, in_size: int, hidden_size: int = 128) -> None:
		super().__init__()
		self.project = nn.Sequential(
			nn.Linear(in_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, 1, bias=False),
		)

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		w = self.project(z).mean(0)
		beta = torch.softmax(w, dim=0)
		beta = beta.expand((z.shape[0],) + beta.shape)
		return (beta * z).sum(1)


class HANLayer(nn.Module):
	"""
	Heterogeneous Attention Network layer.
	One GAT per meta-path, followed by semantic attention aggregation.
	"""

	def __init__(
		self,
		num_meta_paths: int,
		featuredim: int,
		nhid: int,
		layer_num_heads: int,
		dropout: float,
	) -> None:
		super().__init__()
		self.gat_layers = nn.ModuleList()
		for _ in range(num_meta_paths):
			self.gat_layers.append(
				GATConv(
					featuredim,
					nhid,
					layer_num_heads,
					dropout,
					dropout,
					activation=F.elu,
					allow_zero_in_degree=True,
				)
			)
		self.semantic_attention = SemanticAttention(
			in_size=nhid * layer_num_heads
		)
		self.num_meta_paths = num_meta_paths

	def forward(
		self, gs: List["dgl.DGLGraph"], h: List[torch.Tensor]
	) -> torch.Tensor:
		embeddings = []
		for i, g in enumerate(gs):
			x = self.gat_layers[i](g, h[i])
			x = x.flatten(1)
			embeddings.append(x)

		patient_semantic = torch.stack([embeddings[0], embeddings[1]], dim=1)
		med_semantic = torch.stack([embeddings[2], embeddings[3]], dim=1)

		patient = self.semantic_attention(patient_semantic)
		med = self.semantic_attention(med_semantic)
		return patient, med


class DRecHGRCore(nn.Module):
	"""
	Core computation module for DRecHGR.
	Performs GCN on patient-medication and patient-diagnosis heterogeneous graphs,
	then applies HAN on meta-path graphs to produce patient and medication embeddings.
	"""

	def __init__(
		self,
		num_meta_paths: int,
		n_patients: int,
		n_meds: int,
		n_diags: int,
		featuredim: int = 64,
		nhid: int = 8,
		num_heads: List[int] = None,
		dropout: float = 0.6,
		gnn_layer: int = 2,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		if num_heads is None:
			num_heads = [8]
		self.featdim = featuredim
		self.n_meds = n_meds
		self.n_patients = n_patients
		self.gnn_layer = gnn_layer
		self.device = device

		init = nn.init.xavier_uniform_

		self.pEmbed = nn.Parameter(init(torch.empty(n_patients, featuredim)))
		self.mEmbed = nn.Parameter(init(torch.empty(n_meds, featuredim)))
		self.dEmbed = nn.Parameter(init(torch.empty(n_diags, featuredim)))

		self.gcnLayer1 = GCNLayer()
		self.gcnLayer2 = GCNLayer()

		self.HANlayers = HANLayer(
			num_meta_paths, featuredim, nhid, num_heads[0], dropout
		)

		self.output1 = nn.Sequential(
			nn.ReLU(),
			nn.Linear(featuredim * 2, n_meds),
		)

	def forward(
		self,
		adj1: torch.Tensor,
		adj2: torch.Tensor,
		gs: List[Any],
		keepRate: float = 0.5,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		embeds1 = torch.concat([self.pEmbed, self.mEmbed], dim=0)
		embeds2 = torch.concat([self.pEmbed, self.dEmbed], dim=0)

		gnnLats1: List[torch.Tensor] = []
		gnnLats2: List[torch.Tensor] = []
		lats1 = [embeds1]
		lats2 = [embeds2]

		for _ in range(self.gnn_layer):
			tem1 = self.gcnLayer1(adj1, lats1[-1])
			tem2 = self.gcnLayer2(adj2, lats2[-1])
			gnnLats1.append(tem1)
			gnnLats2.append(tem2)

		gnnEmbeds1 = sum(gnnLats1)
		gnnEmbeds2 = sum(gnnLats2)

		pEmbed_gcn1 = gnnEmbeds1[: self.n_patients]
		pEmbed_gcn2 = gnnEmbeds2[: self.n_patients]
		mEmbed_gcn = gnnEmbeds1[self.n_patients :]
		dEmbed_gcn = gnnEmbeds2[self.n_patients :]

		# Free cached memory before heavy GAT computation
		if self.device.type == "cuda":
			torch.cuda.empty_cache()

		# HAN with 4 meta-paths: [pmp, pdp, ddi, ehr]
		h = [pEmbed_gcn1, pEmbed_gcn2, mEmbed_gcn, mEmbed_gcn]
		patient, med = self.HANlayers(gs, h)

		# Free cached memory after GAT computation
		if self.device.type == "cuda":
			torch.cuda.empty_cache()

		# Compute patient-medication similarity scores using chunked computation
		# to avoid allocating excessive GPU memory.
		# patient: (n_patients, featdim),  med: (n_meds, featdim)
		# output1 projects concat(patient[p], med[m]) from 2*featdim to n_meds.
		n_p = patient.shape[0]
		chunk_size = max(1, min(256, n_p))
		simi_pm_chunks: List[torch.Tensor] = []
		for start in range(0, n_p, chunk_size):
			end = min(start + chunk_size, n_p)
			p_chunk = patient[start:end]  # (C, D)
			# Expand patient chunk to (C, M, D)
			p_exp = p_chunk.unsqueeze(1).expand(-1, self.n_meds, -1).reshape(-1, self.featdim)
			# Expand med to (C, M, D)
			m_exp = med.unsqueeze(0).expand(end - start, -1, -1).reshape(-1, self.featdim)
			# Combine and predict
			combined = torch.cat((p_exp, m_exp), dim=1)  # (C*M, 2D)
			chunk_scores = self.output1(combined).reshape(-1, self.n_meds, self.n_meds)
			simi_pm_chunks.append(chunk_scores.sum(axis=1))
		simi_pm = torch.cat(simi_pm_chunks, dim=0)  # (n_patients, n_meds)

		return simi_pm, dEmbed_gcn, med, mEmbed_gcn, patient


class DRecHGR(BaseDrugRecommendationModel):
	"""
	DRecHGR: Drug Recommendation via Heterogeneous Graph Representation.
	A transductive model using GCN + HAN with meta-paths for drug recommendation.

	Expected batch format:
	{
		"visit": List[List[List[int]]],
		"patient_graph_idx": int,
		...
	}
	"""

	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		n_patients: int,
		adj1: torch.Tensor,
		adj2: torch.Tensor,
		meta_graphs: List[Any],
		patient_id_to_idx: Dict[int, int],
		featuredim: int = 64,
		nhid: int = 8,
		num_heads: List[int] = None,
		dropout: float = 0.6,
		gnn_layer: int = 2,
		keep_rate: float = 0.5,
		threshold: float = 0.5,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.threshold = threshold

		self._adj1 = adj1.to(self.device)
		self._adj2 = adj2.to(self.device)
		self._meta_graphs = [g.to(self.device) for g in meta_graphs]
		self._keep_rate = keep_rate
		self._patient_id_to_idx = patient_id_to_idx

		if num_heads is None:
			num_heads = [8]

		self.core = DRecHGRCore(
			num_meta_paths=len(meta_graphs),
			n_patients=n_patients,
			n_meds=vocab_size[2],
			n_diags=vocab_size[0],
			featuredim=featuredim,
			nhid=nhid,
			num_heads=num_heads,
			dropout=dropout,
			gnn_layer=gnn_layer,
			device=self.device,
		)

	def _get_patient_graph_idx(self, batch: Dict[str, Any]) -> int:
		"""Get the patient's graph node index from the batch."""
		idx = batch.get("patient_graph_idx")
		if idx is not None:
			return int(idx)
		visit_obj = batch.get("visit")
		if visit_obj is not None:
			obj_id = id(visit_obj)
			return self._patient_id_to_idx.get(obj_id, 0)
		return 0

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		if not patient:
			return {
				"logits": torch.empty((0, self.vocab_size[2]), device=self.device),
			}

		patient_idx = self._get_patient_graph_idx(batch)

		all_logits, d_emb, med_emb, m_emb_gcn, pat_emb = self.core(
			self._adj1, self._adj2, self._meta_graphs, self._keep_rate
		)

		logits = all_logits[patient_idx : patient_idx + 1]

		return {
			"logits": logits,
			"d_emb": d_emb,
			"med_emb": med_emb,
			"m_emb_gcn": m_emb_gcn,
			"pat_emb": pat_emb,
		}

	def compute_loss(
		self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
	) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)
		targets = self.build_target(batch)
		return F.binary_cross_entropy_with_logits(logits, targets)

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		probs = torch.sigmoid(logits)
		return (probs >= self.threshold).float()

	def on_epoch_end(self) -> None:
		pass

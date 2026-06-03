"""
ARMR model integration for drugrec_benchmark.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


@dataclass
class MambaModelArgs:
	d_model: int
	d_state: int = 16
	expand: int = 2
	dt_rank: Union[int, str] = "auto"
	d_conv: int = 4
	conv_bias: bool = True
	bias: bool = False

	def __post_init__(self) -> None:
		self.d_inner = int(self.expand * self.d_model)
		if self.dt_rank == "auto":
			self.dt_rank = math.ceil(self.d_model / 16)


class MambaBlock(nn.Module):
	def __init__(self, args: MambaModelArgs) -> None:
		super().__init__()
		self.args = args

		self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
		self.conv1d = nn.Conv1d(
			in_channels=args.d_inner,
			out_channels=args.d_inner,
			bias=args.conv_bias,
			kernel_size=args.d_conv,
			groups=args.d_inner,
			padding=args.d_conv - 1,
		)
		self.x_proj = nn.Linear(args.d_inner, int(args.dt_rank) + args.d_state * 2, bias=False)
		self.dt_proj = nn.Linear(int(args.dt_rank), args.d_inner, bias=True)

		base = torch.arange(1, args.d_state + 1, dtype=torch.float32)
		A = base.unsqueeze(0).repeat(args.d_inner, 1)
		self.A_log = nn.Parameter(torch.log(A))
		self.D = nn.Parameter(torch.ones(args.d_inner))
		self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch, length, _ = x.shape
		x_and_res = self.in_proj(x)
		x, res = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

		x = x.transpose(1, 2)
		x = self.conv1d(x)[:, :, :length]
		x = x.transpose(1, 2)
		x = F.silu(x)

		y = self.ssm(x)
		y = y * F.silu(res)
		return self.out_proj(y)

	def ssm(self, x: torch.Tensor) -> torch.Tensor:
		d_in, n = self.A_log.shape
		A = -torch.exp(self.A_log.float())
		D = self.D.float()
		x_dbl = self.x_proj(x)
		delta, B, C = x_dbl.split(split_size=[int(self.args.dt_rank), n, n], dim=-1)
		delta = F.softplus(self.dt_proj(delta))
		return self.selective_scan(x, delta, A, B, C, D)

	def selective_scan(
		self,
		u: torch.Tensor,
		delta: torch.Tensor,
		A: torch.Tensor,
		B: torch.Tensor,
		C: torch.Tensor,
		D: torch.Tensor,
	) -> torch.Tensor:
		batch, length, d_in = u.shape
		n = A.shape[1]

		deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
		deltaB_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

		x = torch.zeros((batch, d_in, n), device=u.device, dtype=u.dtype)
		ys: List[torch.Tensor] = []
		for idx in range(length):
			x = deltaA[:, idx] * x + deltaB_u[:, idx]
			y = torch.einsum("bdn,bn->bd", x, C[:, idx, :])
			ys.append(y)
		y = torch.stack(ys, dim=1)
		return y + u * D


class GraphConvolution(nn.Module):
	def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter("bias", None)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		stdv = 1.0 / math.sqrt(self.weight.size(1))
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
	def __init__(self, voc_size: int, emb_dim: int, ehr_adj: torch.Tensor, ddi_adj: torch.Tensor) -> None:
		super().__init__()
		device = ehr_adj.device
		self.ehr_adj = self._normalize(ehr_adj + torch.eye(ehr_adj.shape[0], device=device))
		self.ddi_adj = self._normalize(ddi_adj + torch.eye(ddi_adj.shape[0], device=device))
		self.x = torch.eye(voc_size, device=device)

		self.gcn1 = GraphConvolution(voc_size, emb_dim)
		self.dropout = nn.Dropout(p=0.3)
		self.gcn2 = GraphConvolution(emb_dim, emb_dim)
		self.gcn3 = GraphConvolution(emb_dim, emb_dim)

	def _normalize(self, mx: torch.Tensor) -> torch.Tensor:
		if mx.is_sparse:
			mx = mx.to_dense()
		rowsum = mx.sum(1)
		r_inv = rowsum.pow(-1).flatten()
		r_inv[torch.isinf(r_inv)] = 0.0
		r_mat_inv = torch.diag(r_inv)
		return torch.mm(r_mat_inv, mx)

	def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
		ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
		ehr_node_embedding = F.relu(ehr_node_embedding)
		ehr_node_embedding = self.dropout(ehr_node_embedding)
		ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)

		ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)
		ddi_node_embedding = F.relu(ddi_node_embedding)
		ddi_node_embedding = self.dropout(ddi_node_embedding)
		ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
		return ehr_node_embedding, ddi_node_embedding


class PiecewiseTSL(nn.Module):
	def __init__(self, emb_dim: int, k: int) -> None:
		super().__init__()
		self.k = k
		self.mamba = MambaBlock(MambaModelArgs(d_model=emb_dim))
		self.lin = nn.Linear(k * emb_dim, k * emb_dim)
		self.norm = nn.LayerNorm(k * emb_dim)

	def forward(self, seq: torch.Tensor) -> torch.Tensor:
		batch, seq_len, emb_dim = seq.shape
		if seq_len < self.k:
			pad_len = self.k - seq_len
			seq = torch.cat((seq, torch.zeros(batch, pad_len, emb_dim, device=seq.device)), dim=1)
			seq_len = self.k

		near_seq = seq[:, : self.k, :]
		near_h = self.lin(self.norm(near_seq.reshape(batch, -1))).reshape(batch, self.k, emb_dim) + near_seq
		far_h = torch.zeros(batch, self.k, emb_dim, device=seq.device)

		if seq_len > self.k:
			far_seq = self.mamba(torch.flip(seq[:, self.k :, :], [1]))
			scores = torch.bmm(near_h, far_seq.transpose(-2, -1))
			scores = F.softmax(scores, dim=-1)
			far_h = torch.bmm(scores, far_seq)

		return torch.cat((near_h, far_h), dim=1)


class PatientRepLearn(nn.Module):
	def __init__(self, emb_dim: int, k: int, d_voc_size: int, p_voc_size: int, m_voc_size: int) -> None:
		super().__init__()
		self.tsl_d = PiecewiseTSL(emb_dim, k)
		self.tsl_p = PiecewiseTSL(emb_dim, k)
		self.d_lin = nn.Linear(d_voc_size, emb_dim)
		self.p_lin = nn.Linear(p_voc_size, emb_dim)
		self.m_lin = nn.Linear(m_voc_size, emb_dim)

	def forward(self, diags: torch.Tensor, procs: torch.Tensor, meds: torch.Tensor) -> torch.Tensor:
		e_d = self.d_lin(diags)
		e_p = self.p_lin(procs)
		e_m = self.m_lin(meds)
		e_h = e_d + e_p + e_m
		h_d = self.tsl_d(e_d)
		h_p = self.tsl_p(e_p)

		if diags.size(1) < h_d.size(1):
			pad_len = h_d.size(1) - diags.size(1)
			e_h = torch.cat(
				(e_h, torch.zeros(e_h.shape[0], pad_len, e_h.shape[2], device=e_h.device)),
				dim=1,
			)
		else:
			e_h = e_h[:, : h_d.size(1), :]
		return torch.cat((e_h, h_d + h_p), dim=1)


class MedRepLearn(nn.Module):
	def __init__(self, emb_dim: int, k: int, m_voc_size: int) -> None:
		super().__init__()
		self.emb_dim = emb_dim
		self.m_voc_size = m_voc_size
		self.tsl_old = PiecewiseTSL(emb_dim, k)
		self.tsl_new = PiecewiseTSL(emb_dim, k)
		self.m_embs = nn.Embedding(m_voc_size, emb_dim)
		self.lin_expand = nn.Linear(emb_dim, 2 * k * emb_dim)

	def forward(self, meds: torch.Tensor) -> torch.Tensor:
		batch = meds.shape[0]
		history = torch.cat(
			(
				torch.zeros(batch, 1, self.m_voc_size, device=meds.device),
				(torch.cumsum(meds, dim=1) > 0).float()[:, :-1, :],
			),
			dim=1,
		)
		old = meds * history
		new = meds - old

		e_old = torch.matmul(old, self.m_embs.weight)
		e_new = torch.matmul(new, self.m_embs.weight)

		masks = (meds.sum(dim=1) > 0).float().unsqueeze(2).repeat(1, 1, self.emb_dim)
		old_m_embs = self.m_embs.weight.unsqueeze(0).repeat(batch, 1, 1) * masks
		new_m_embs = self.m_embs.weight.unsqueeze(0).repeat(batch, 1, 1) * (1 - masks)

		h_old = self.tsl_old(e_old).reshape(batch, -1)
		h_new = self.tsl_new(e_new).reshape(batch, -1)
		scores_old = torch.cosine_similarity(h_old.unsqueeze(1), self.lin_expand(old_m_embs), dim=2)
		scores_new = torch.cosine_similarity(h_new.unsqueeze(1), self.lin_expand(new_m_embs), dim=2)
		total_scores = scores_old + scores_new
		return self.m_embs.weight.unsqueeze(0).repeat(batch, 1, 1) * total_scores.unsqueeze(2)


class ARMRCore(nn.Module):
	def __init__(
		self,
		emb_dim: int,
		vocab_size: Tuple[int, int, int],
		history_k: int,
		ehr_adj: torch.Tensor,
		di_adj: torch.Tensor,
		blend_weight: float = 0.7,
	) -> None:
		super().__init__()
		self.d_voc_size = vocab_size[0]
		self.p_voc_size = vocab_size[1]
		self.m_voc_size = vocab_size[2]
		self.k = history_k
		self.blend_weight = blend_weight

		self.medrep = MedRepLearn(emb_dim, history_k, self.m_voc_size)
		self.patrep = PatientRepLearn(emb_dim, history_k, self.d_voc_size, self.p_voc_size, self.m_voc_size)
		self.lin1 = nn.Linear(2 * history_k * emb_dim, self.m_voc_size)
		self.lin2 = nn.Linear(2 * history_k * emb_dim, 2 * history_k * emb_dim)
		self.norm = nn.LayerNorm(2 * history_k * emb_dim)
		self.gcn = GCN(self.m_voc_size, emb_dim, ehr_adj, di_adj)
		self.lin_med_expand = nn.Linear(emb_dim, 2 * history_k * emb_dim)

	def forward(self, diags: torch.Tensor, procs: torch.Tensor, meds: torch.Tensor) -> torch.Tensor:
		procs = torch.zeros_like(procs)
		batch = diags.shape[0]
		pat = self.patrep(diags, procs, meds)
		h_patient = pat[:, 2 * self.k :, :].reshape(batch, -1)
		query = h_patient + self.lin2(self.norm(h_patient))

		ehr_meds, ddi_meds = self.gcn()
		h_meds = self.medrep(meds)
		h_meds = h_meds + ehr_meds.unsqueeze(0).repeat(batch, 1, 1)
		h_meds = h_meds + ddi_meds.unsqueeze(0).repeat(batch, 1, 1) * 0.5
		h_meds = self.lin_med_expand(h_meds)

		o_1 = self.lin1(pat[:, : 2 * self.k, :].reshape(batch, -1))
		o_2 = torch.cosine_similarity(query.unsqueeze(1).repeat(1, self.m_voc_size, 1), h_meds, dim=2)
		return o_1 * self.blend_weight + o_2 * (1 - self.blend_weight)


class ARMR(BaseDrugRecommendationModel):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 256,
		history_k: int = 3,
		max_visits: int = -1,
		threshold: float = 0.5,
		blend_weight: float = 0.7,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.history_k = history_k
		self.max_visits = max_visits
		self.threshold = threshold
		self.ddi_adj = ddi_adj

		ehr_adj_tensor = torch.as_tensor(ehr_adj, dtype=torch.float32, device=self.device)
		ddi_adj_tensor = torch.as_tensor(ddi_adj, dtype=torch.float32, device=self.device)

		self.core = ARMRCore(
			emb_dim=emb_dim,
			vocab_size=vocab_size,
			history_k=history_k,
			ehr_adj=ehr_adj_tensor,
			di_adj=ddi_adj_tensor,
			blend_weight=blend_weight,
		)

	def _to_multihot(self, codes: Sequence[int], vocab_size: int) -> torch.Tensor:
		vec = torch.zeros((vocab_size,), dtype=torch.float32, device=self.device)
		if codes:
			indices = torch.tensor(list(codes), dtype=torch.long, device=self.device)
			vec[indices] = 1.0
		return vec

	def _build_inputs(
		self,
		patient: Sequence[Sequence[Sequence[int]]],
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		if not patient:
			empty_seq = torch.empty((0, 0, 0), device=self.device)
			empty_target = torch.empty((0, self.vocab_size[2]), device=self.device)
			return empty_seq, empty_seq, empty_seq, empty_target

		if self.max_visits > 0:
			selected = patient[-self.max_visits :]
		else:
			selected = patient
		ordered = list(reversed(selected))

		diags: List[torch.Tensor] = []
		procs: List[torch.Tensor] = []
		meds: List[torch.Tensor] = []

		for idx, adm in enumerate(ordered):
			d_codes = adm[0] if len(adm) > 0 else []
			p_codes = adm[1] if len(adm) > 1 else []
			m_codes = adm[2] if len(adm) > 2 else []

			diags.append(self._to_multihot(d_codes, self.vocab_size[0]))
			procs.append(self._to_multihot(p_codes, self.vocab_size[1]))
			if idx == 0:
				meds.append(torch.zeros((self.vocab_size[2],), device=self.device))
			else:
				meds.append(self._to_multihot(m_codes, self.vocab_size[2]))

		diag_seq = torch.stack(diags, dim=0).unsqueeze(0)
		proc_seq = torch.stack(procs, dim=0).unsqueeze(0)
		med_seq = torch.stack(meds, dim=0).unsqueeze(0)

		target = torch.zeros((1, self.vocab_size[2]), dtype=torch.float32, device=self.device)
		last_adm = patient[-1]
		if len(last_adm) > 2 and last_adm[2]:
			target[0, last_adm[2]] = 1.0
		return diag_seq, proc_seq, med_seq, target

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		if not patient:
			return {
				"logits": torch.empty((0, self.vocab_size[2]), device=self.device),
				"targets": torch.empty((0, self.vocab_size[2]), device=self.device),
			}

		diag_seq, proc_seq, med_seq, target = self._build_inputs(patient)
		if patient:
			target = self.build_target(batch)
		logits = self.core(diag_seq, proc_seq, med_seq)
		return {
			"logits": logits,
			"targets": target,
		}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		targets = outputs["targets"]
		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)
		return F.binary_cross_entropy_with_logits(logits, targets)

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		probs = torch.sigmoid(logits)
		return (probs >= self.threshold).float()

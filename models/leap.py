"""
LEAP model implementation integrated with the benchmark base model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.core.io import Prediction
from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel
from drugrec_benchmark.utils.metrics import sequence_output_process

class LeapCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		emb_dim: int = 64,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.device = device
		self.sos_token = vocab_size[2]
		self.end_token = vocab_size[2] + 1

		self.enc_embedding = nn.Sequential(
			nn.Embedding(vocab_size[0], emb_dim),
			nn.Dropout(0.3),
		)
		self.dec_embedding = nn.Sequential(
			nn.Embedding(vocab_size[2] + 2, emb_dim),
			nn.Dropout(0.3),
		)

		self.dec_gru = nn.GRU(emb_dim * 2, emb_dim, batch_first=True)
		self.attn = nn.Linear(emb_dim * 2, 1)
		self.output = nn.Linear(emb_dim, vocab_size[2] + 2)

	def forward(self, adm: Sequence[Sequence[int]], max_len: int = 20) -> torch.Tensor:
		device = self.device
		input_tensor = torch.LongTensor(adm[0]).to(device)
		input_embedding = self.enc_embedding(input_tensor.unsqueeze(dim=0)).squeeze(
			dim=0
		)

		output_logits: List[torch.Tensor] = []
		hidden_state: Optional[torch.Tensor] = None
		if self.training:
			for med_code in [self.sos_token] + list(adm[2]):
				dec_input = torch.LongTensor([med_code]).unsqueeze(dim=0).to(device)
				dec_input = self.dec_embedding(dec_input).squeeze(dim=0)

				if hidden_state is None:
					hidden_state = dec_input
				hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1)

				combined_input = torch.cat(
					[hidden_state_repeat, input_embedding], dim=-1
				)
				attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1)
				input_embedding = attn_weight.mm(input_embedding)

				_, hidden_state = self.dec_gru(
					torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
					hidden_state.unsqueeze(dim=0),
				)
				hidden_state = hidden_state.squeeze(dim=0)

				output_logits.append(self.output(F.relu(hidden_state)))
			return torch.cat(output_logits, dim=0)

		for step in range(max_len):
			if step == 0:
				dec_input = torch.LongTensor([[self.sos_token]]).to(device)
			dec_input = self.dec_embedding(dec_input).squeeze(dim=0)
			if hidden_state is None:
				hidden_state = dec_input
			hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1)

			combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1)
			attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1)
			input_embedding = attn_weight.mm(input_embedding)

			_, hidden_state = self.dec_gru(
				torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
				hidden_state.unsqueeze(dim=0),
			)
			hidden_state = hidden_state.squeeze(dim=0)

			output = self.output(F.relu(hidden_state))
			output_logits.append(F.softmax(output, dim=-1))
			_, topi = output.data.topk(1)
			dec_input = topi.detach()

		return torch.cat(output_logits, dim=0)


class Leap(BaseDrugRecommendationModel):
	"""
	LEAP wrapper integrated with the benchmark interfaces.

	Expected batch format:
	{
		"visits": List[List[Tuple[List[int], List[int], List[int]]]]
	}
	"""

	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		emb_dim: int = 64,
		max_len: int = 20,
		threshold: float = 0.5,
		ddi_adj: Optional[torch.Tensor] = None,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "sequence"
		self.vocab_size = vocab_size
		self.max_len = max_len
		self.threshold = threshold
		self.sos_token = vocab_size[2]
		self.end_token = vocab_size[2] + 1
		self.ddi_adj = ddi_adj

		self.leap = LeapCore(
			vocab_size=vocab_size,
			emb_dim=emb_dim,
			device=self.device,
		)

	def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		patient = self.get_patient(batch)
		if not patient:
			return {"seq_logits": [], "logits": np.zeros((0, self.vocab_size[2] + 2), dtype=np.float32)}

		adm = patient[-1]
		seq_logits = self.leap(adm, max_len=self.max_len)
		return {
			"seq_logits": [seq_logits],
			"logits": seq_logits.detach().cpu().numpy(),
		}

	def decode(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Prediction:
		logits = outputs.get("logits")
		logits_array = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
		med_size = self.vocab_size[2]
		target = self.get_target_indices(batch)
		if logits_array.size == 0:
			return Prediction(
				med_indices=[],
				med_scores=np.zeros((med_size,), dtype=float),
				target=target,
				task="sequence",
			)
		out_list, ranked = sequence_output_process(logits_array, [self.sos_token, self.end_token])
		scores = np.mean(logits_array[:, :med_size], axis=0)
		return Prediction(
			med_indices=list(out_list),
			med_scores=np.asarray(scores, dtype=float),
			target=target,
			task="sequence",
			ranked_med_indices=list(ranked),
		)

	def compute_loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
		patient = self.get_patient(batch)
		seq_logits_list: List[torch.Tensor] = outputs.get("seq_logits", [])
		if not patient or not seq_logits_list:
			return torch.tensor(0.0, device=self.device)
		seq_logits = seq_logits_list[0]
		target = self.get_target_indices(batch) + [self.end_token]
		target_tensor = torch.LongTensor(target).to(self.device)
		return F.cross_entropy(seq_logits, target_tensor)


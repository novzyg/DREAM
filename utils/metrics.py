"""
Multi-label classification metrics.
"""
from __future__ import annotations

from typing import Optional

import warnings

import numpy as np
import torch

from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
)

import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import warnings
import dill
from collections import Counter
from rdkit import Chem
from collections import defaultdict


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
	if isinstance(tensor, torch.Tensor):
		return tensor.detach().cpu().numpy()
	return np.asarray(tensor)


def multilabel_jaccard(preds: torch.Tensor, targets: torch.Tensor) -> float:
	p = _to_numpy(preds)
	t = _to_numpy(targets)
	if p.size == 0:
		return 0.0
	values = []
	for i in range(p.shape[0]):
		pred_set = set(np.where(p[i] == 1)[0].tolist())
		tgt_set = set(np.where(t[i] == 1)[0].tolist())
		union = pred_set | tgt_set
		if not union:
			values.append(0.0)
		else:
			values.append(len(pred_set & tgt_set) / len(union))
	return float(np.mean(values))


def multilabel_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
	p = _to_numpy(preds)
	t = _to_numpy(targets)
	if p.size == 0:
		return 0.0
	values = []
	for i in range(p.shape[0]):
		tp = np.sum((p[i] == 1) & (t[i] == 1))
		fp = np.sum((p[i] == 1) & (t[i] == 0))
		fn = np.sum((p[i] == 0) & (t[i] == 1))
		denom = 2 * tp + fp + fn
		values.append(0.0 if denom == 0 else (2 * tp) / denom)
	return float(np.mean(values))


def multilabel_prauc(preds: torch.Tensor, targets: torch.Tensor) -> float:
	try:
		from sklearn.metrics import average_precision_score
	except ImportError:
		return 0.0
	warnings.filterwarnings(
		"ignore",
		message="No positive class found in y_true",
		category=UserWarning,
	)
	p = _to_numpy(preds)
	t = _to_numpy(targets)
	if p.size == 0:
		return 0.0
	try:
		return float(average_precision_score(t, p, average="macro"))
	except Exception:
		return 0.0


def multilabel_summary(preds: torch.Tensor, targets: torch.Tensor, probs: Optional[torch.Tensor] = None) -> dict:
	p = _to_numpy(preds)
	t = _to_numpy(targets)
	prob = _to_numpy(probs) if probs is not None else None
	if p.size == 0:
		return {
			"jaccard": 0.0,
			"prauc": 0.0,
			"avg_prc": 0.0,
			"avg_recall": 0.0,
			"avg_f1": 0.0,
			"avg_med": 0.0,
		}

	jac = multilabel_jaccard(preds, targets)
	if prob is None:
		prauc = multilabel_prauc(preds, targets)
	else:
		prauc = multilabel_prauc(torch.as_tensor(prob), targets)

	avg_prc_list = []
	avg_recall_list = []
	avg_f1_list = []
	med_counts = []

	for i in range(p.shape[0]):
		pred_idx = set(np.where(p[i] == 1)[0].tolist())
		tgt_idx = set(np.where(t[i] == 1)[0].tolist())
		inter = pred_idx & tgt_idx
		precision = 0.0 if not pred_idx else len(inter) / len(pred_idx)
		recall = 0.0 if not tgt_idx else len(inter) / len(tgt_idx)
		if precision + recall == 0:
			f1 = 0.0
		else:
			f1 = 2 * precision * recall / (precision + recall)
		avg_prc_list.append(precision)
		avg_recall_list.append(recall)
		avg_f1_list.append(f1)
		med_counts.append(len(pred_idx))

	return {
		"jaccard": float(jac),
		"prauc": float(prauc),
		"avg_prc": float(np.mean(avg_prc_list)),
		"avg_recall": float(np.mean(avg_recall_list)),
		"avg_f1": float(np.mean(avg_f1_list)),
		"avg_med": float(np.mean(med_counts)),
	}


def multi_label_metric(y_gt: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> tuple:
	def jaccard(gt: np.ndarray, pred: np.ndarray) -> float:
		score = []
		for b in range(gt.shape[0]):
			target = np.where(gt[b] == 1)[0]
			out_list = np.where(pred[b] == 1)[0]
			inter = set(out_list) & set(target)
			union = set(out_list) | set(target)
			jaccard_score = 0.0 if len(union) == 0 else len(inter) / len(union)
			score.append(jaccard_score)
		return float(np.mean(score))

	def average_prc(gt: np.ndarray, pred: np.ndarray) -> list:
		score = []
		for b in range(gt.shape[0]):
			target = np.where(gt[b] == 1)[0]
			out_list = np.where(pred[b] == 1)[0]
			inter = set(out_list) & set(target)
			prc_score = 0.0 if len(out_list) == 0 else len(inter) / len(out_list)
			score.append(prc_score)
		return score

	def average_recall(gt: np.ndarray, pred: np.ndarray) -> list:
		score = []
		for b in range(gt.shape[0]):
			target = np.where(gt[b] == 1)[0]
			out_list = np.where(pred[b] == 1)[0]
			inter = set(out_list) & set(target)
			recall_score = 0.0 if len(target) == 0 else len(inter) / len(target)
			score.append(recall_score)
		return score

	def average_f1(prc: list, recall: list) -> list:
		score = []
		for idx in range(len(prc)):
			if prc[idx] + recall[idx] == 0:
				score.append(0.0)
			else:
				score.append(2 * prc[idx] * recall[idx] / (prc[idx] + recall[idx]))
		return score

	try:
		from sklearn.metrics import average_precision_score
	except ImportError:
		average_precision_score = None

	if average_precision_score is None:
		prauc = 0.0
	else:
		prauc_list = []
		for b in range(len(y_gt)):
			try:
				prauc_list.append(
					float(average_precision_score(y_gt[b], y_prob[b], average="macro"))
				)
			except Exception:
				prauc_list.append(0.0)
		prauc = float(np.mean(prauc_list))

	ja = jaccard(y_gt, y_pred)
	avg_prc = average_prc(y_gt, y_pred)
	avg_recall = average_recall(y_gt, y_pred)
	avg_f1 = average_f1(avg_prc, avg_recall)

	return ja, prauc, float(np.mean(avg_prc)), float(np.mean(avg_recall)), float(np.mean(avg_f1))

def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(
                    2
                    * average_prc[idx]
                    * average_recall[idx]
                    / (average_prc[idx] + average_recall[idx])
                )
        return score

    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average="macro"))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)

    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0
    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]

    out_list = []
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [
        x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)
    ]
    return out_list, sorted_predict
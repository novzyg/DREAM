"""
MR-DTR: Time-aware Medication Recommendation via Intervention of Dynamic Treatment Regimes.

Integrates the MR-DTR model into the drugrec_benchmark framework.
Original model: https://github.com/MLTS-thu/MR-DTR
"""
from __future__ import annotations

import math
import os
import pickle
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from drugrec_benchmark.core.io import Prediction, as_model_output
from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel
from drugrec_benchmark.utils.dataset_utils import load_pickle



# =============================================================================
# Original MR-DTR helper modules (from MR-DTR/model.py)
# =============================================================================

class PeriodicTimeEncoder(nn.Module):
    def __init__(self, embedding_dimension: int):
        super(PeriodicTimeEncoder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.scale_factor = (1 / (embedding_dimension // 2)) ** 0.5
        self.w = nn.Parameter(torch.randn(1, embedding_dimension // 2))
        self.b = nn.Parameter(torch.randn(1, embedding_dimension // 2))

    def forward(self, input_relative_time: torch.Tensor):
        cos_encoding = torch.cos(torch.matmul(input_relative_time, self.w) + self.b)
        sin_encoding = torch.sin(torch.matmul(input_relative_time, self.w) + self.b)
        time_encoding = self.scale_factor * torch.cat([cos_encoding, sin_encoding], dim=-1)
        return time_encoding


class weighted_graph_conv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(weighted_graph_conv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, graph, node_features, edge_weights):
        graph = graph.local_var()
        graph.ndata['n'] = node_features
        graph.edata['e'] = edge_weights.t().unsqueeze(dim=-1)
        graph.update_all(fn.u_mul_e('n', 'e', 'msg'), fn.sum('msg', 'h'))
        node_features = graph.ndata.pop('h')
        output = self.linear(node_features)
        return output


class weighted_GCN(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int):
        super(weighted_GCN, self).__init__()
        gcns, relus, bns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        input_size = in_features
        for hidden_size in hidden_sizes:
            gcns.append(weighted_graph_conv(input_size, hidden_size))
            relus.append(nn.ReLU())
            bns.append(nn.BatchNorm1d(hidden_size))
            input_size = hidden_size
        gcns.append(weighted_graph_conv(hidden_sizes[-1], out_features))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(out_features))
        self.gcns, self.relus, self.bns = gcns, relus, bns

    def forward(self, graph: dgl.DGLGraph, node_features: torch.Tensor, edges_weight: torch.Tensor):
        h = node_features
        for gcn, relu, bn in zip(self.gcns, self.relus, self.bns):
            h = gcn(graph, h, edges_weight)
            if h.shape[0] != 1:
                h = bn(h.transpose(1, -1)).transpose(1, -1)
            h = relu(h)
        return h


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, nodes_embedding_projection, patient_feature, drug_memory):
        stacked_inputs = torch.stack([nodes_embedding_projection, patient_feature, drug_memory], dim=1)
        queries = self.query(stacked_inputs)
        keys = self.key(stacked_inputs)
        values = self.value(stacked_inputs)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)
        weighted_values = torch.matmul(attention_weights, values)
        output = torch.sum(weighted_values, dim=1)
        return output


class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(torch.nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))
        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)
        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)
        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)
        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class TimeRec_GCN(nn.Module):
    """
    Original MR-DTR core model (TimeRec_GCN).
    """
    def __init__(self, voc_size: list, num_users: int, embedding_dimension: int,
                 embedding_dropout: float, temporal_attention_dropout: float,
                 temporal_information_importance: float,
                 ehr_adj, ddi_adj, device, hop_num: int = 3,
                 temporal_feature_dimension: int = 1):
        super(TimeRec_GCN, self).__init__()
        self.voc_size = voc_size
        self.num_users = num_users
        self.hop_num = hop_num
        self.embedding_dimension = embedding_dimension
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.gru_fcn = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dimension * 2, embedding_dimension),
        )
        self.med_gcn = GCN(voc_size=voc_size[2], emb_dim=embedding_dimension,
                           ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.diagnosis_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], embedding_dimension),
            nn.Dropout(embedding_dropout)
        )
        self.procedure_embedding = nn.Sequential(
            nn.Embedding(voc_size[1], embedding_dimension),
            nn.Dropout(embedding_dropout)
        )
        self.medication_embedding = nn.Sequential(
            nn.Embedding(voc_size[2], embedding_dimension),
            nn.Dropout(embedding_dropout)
        )
        self.users_embedding = nn.Embedding(num_users, embedding_dimension)
        self.leaky_relu_func = nn.LeakyReLU(negative_slope=0.2)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.temporal_attention_dropout = nn.Dropout(temporal_attention_dropout)
        self.nhead = 2
        self.diagnosis_transformer_encoder = nn.TransformerEncoderLayer(
            embedding_dimension, self.nhead, batch_first=True, dropout=0.2)
        self.procedure_transformer_encoder = nn.TransformerEncoderLayer(
            embedding_dimension, self.nhead, batch_first=True, dropout=0.2)
        self.periodic_time_encoder = PeriodicTimeEncoder(embedding_dimension=embedding_dimension)
        self.fc_projection = nn.Linear(hop_num * embedding_dimension, embedding_dimension)
        self.final_fcn = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dimension, 1)
        )
        self.device = device
        self.inter = torch.nn.Parameter(torch.FloatTensor(1))
        self.inter3 = torch.nn.Parameter(torch.FloatTensor(1))
        self.inter4 = torch.nn.Parameter(torch.FloatTensor(1))
        self.init_weights()

    def forward(self, hops_nodes_indices, hops_nodes_temporal_features,
                central_nodes_temporal_feature, diagnosis_list, procedure_list, time_list):
        query_embeddings = self.medication_embedding(
            torch.LongTensor([i for i in range(self.voc_size[2])]).to(self.device))
        nodes_hops_embedding = []
        diagnosis_feature = [torch.sum(
            self.diagnosis_embedding(torch.LongTensor(i).to(self.device)),
            keepdim=True, dim=0) for i in diagnosis_list]
        procedure_feature = [torch.sum(
            self.procedure_embedding(torch.LongTensor(i).to(self.device)),
            keepdim=True, dim=0) for i in procedure_list]
        diagnosis_feature = torch.cat(diagnosis_feature, dim=0).unsqueeze(0)
        procedure_feature = torch.cat(procedure_feature, dim=0).unsqueeze(0)

        d_mask_matrix = torch.triu(torch.ones(len(diagnosis_list), len(diagnosis_list)),
                                   diagonal=1).repeat(self.nhead, 1, 1).to(self.device)
        diagnosis_feature = self.diagnosis_transformer_encoder(diagnosis_feature, src_mask=d_mask_matrix)
        procedure_feature = self.procedure_transformer_encoder(procedure_feature, src_mask=d_mask_matrix)
        diagnosis_feature = diagnosis_feature[:, -1, :]
        procedure_feature = procedure_feature[:, -1, :]

        patient_feature = torch.cat([diagnosis_feature, procedure_feature], dim=-1)
        patient_feature = self.gru_fcn(patient_feature)
        patient_feature = patient_feature * query_embeddings

        ehr_embedding, ddi_embedding = self.med_gcn()
        drug_memory = ehr_embedding - ddi_embedding * self.inter

        central_nodes_time_embedding = self.periodic_time_encoder(
            torch.Tensor([[central_nodes_temporal_feature]]).to(self.device))

        for hop_index in range(len(hops_nodes_indices)):
            hop_nodes_indices = hops_nodes_indices[hop_index]
            hop_nodes_temporal_features = hops_nodes_temporal_features[hop_index]

            if hop_index % 2 == 0:
                if hop_index == 0:
                    continue
                else:
                    hop_nodes_embedding = self.users_embedding(
                        torch.LongTensor([hop_nodes_indices]).to(self.device))
            else:
                hop_diagnosis_nodes_embedding = self.diagnosis_embedding(
                    torch.LongTensor([hop_nodes_indices[0]]).to(self.device))
                hop_procedure_nodes_embedding = self.procedure_embedding(
                    torch.LongTensor([hop_nodes_indices[1]]).to(self.device))
                hop_medication_nodes_embedding = self.medication_embedding(
                    torch.LongTensor([hop_nodes_indices[2]]).to(self.device))
                hop_nodes_embedding = torch.cat(
                    [hop_diagnosis_nodes_embedding, hop_procedure_nodes_embedding,
                     hop_medication_nodes_embedding], dim=1)

            hop_nodes_embedding = self.embedding_dropout(hop_nodes_embedding)
            attention = torch.einsum('if,bnf->bin', query_embeddings, hop_nodes_embedding)

            hop_nodes_time_embedding = self.periodic_time_encoder(
                torch.Tensor([hop_nodes_temporal_features]).unsqueeze(dim=-1).to(self.device))

            temporal_attention = torch.einsum(
                'bif,bnf->bin',
                torch.stack([central_nodes_time_embedding for _ in range(self.voc_size[2])], dim=1),
                hop_nodes_time_embedding)
            temporal_attention = self.temporal_attention_dropout(temporal_attention)
            attention = self.leaky_relu_func(attention)
            attention_scores = F.softmax(attention, dim=-1)
            hop_embedding = torch.bmm(attention_scores, hop_nodes_embedding)
            nodes_hops_embedding.append(hop_embedding)

        nodes_hops_embedding = self.embedding_dropout(torch.stack(nodes_hops_embedding, dim=2))
        nodes_embedding_projection = self.fc_projection(nodes_hops_embedding.flatten(start_dim=2))
        nodes_embedding_projection = self.inter4 * nodes_embedding_projection.squeeze(0) \
            + patient_feature + self.inter3 * drug_memory
        set_prediction = self.final_fcn(nodes_embedding_projection).t()

        neg_pred_prob = torch.sigmoid(set_prediction)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return set_prediction, batch_neg

    def init_weights(self):
        initrange = 0.1
        self.diagnosis_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.procedure_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.medication_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.users_embedding.weight.data.uniform_(-initrange, initrange)


# =============================================================================
# DREAM-compatible wrapper
# =============================================================================

class MRDTRModel(BaseDrugRecommendationModel):
    """
    MR-DTR model wrapped for the drugrec_benchmark framework.
    Uses pre-processed graph hop data.
    """

    def __init__(
        self,
        voc_size: Tuple[int, int, int],
        num_users: int,
        emb_dim: int = 64,
        dropout: float = 0.2,
        temporal_information_importance: float = 0.5,
        ehr_adj: Optional[np.ndarray] = None,
        ddi_adj: Optional[np.ndarray] = None,
        target_ddi: float = 0.06,
        kp: float = 2.5,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cpu")
        super().__init__(device=device)

        self.voc_size = voc_size
        self.num_users = num_users
        self.target_ddi = target_ddi
        self.kp = kp
        self.threshold = threshold
        self.model_type = "multilabel"
        self.supports_batched_training = False

        self._ddi_adj = ddi_adj
        if ddi_adj is not None:
            self.register_buffer("_ddi_adj_tensor", torch.FloatTensor(ddi_adj))

        self.core = TimeRec_GCN(
            voc_size=list(voc_size),
            num_users=num_users,
            embedding_dimension=emb_dim,
            embedding_dropout=dropout,
            temporal_attention_dropout=dropout,
            temporal_information_importance=temporal_information_importance,
            ehr_adj=ehr_adj,
            ddi_adj=ddi_adj,
            device=device,
        )
        self.core.to(device)

    def get_ddi_adj(self):
        return self._ddi_adj

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        hops_nodes_indices = batch["hops_nodes_indices"]
        hops_nodes_temporal_features = batch["hops_nodes_temporal_features"]
        central_nodes_temporal_feature = batch["central_nodes_temporal_feature"]
        diagnosis_list = batch["diagnosis_list"]
        procedure_list = batch["procedure_list"]
        time_list = batch["time_list"]

        logits, loss_ddi = self.core(
            hops_nodes_indices,
            hops_nodes_temporal_features,
            central_nodes_temporal_feature,
            diagnosis_list,
            procedure_list,
            time_list,
        )

        return {
            "logits": logits,
            "loss_ddi": loss_ddi,
            "task": "multilabel",
        }

    def compute_loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
        logits = outputs["logits"]
        loss_ddi = outputs["loss_ddi"]
        target_indices = batch["target_indices"]
        voc_size_med = self.voc_size[2]

        loss_bce_target = torch.zeros((1, voc_size_med), device=self.device)
        loss_bce_target[:, target_indices] = 1

        loss_multi_target = -torch.ones((1, voc_size_med), dtype=torch.long, device=self.device)
        for idx, item in enumerate(target_indices):
            if item >= 0:
                loss_multi_target[0][idx] = item

        loss_bce = F.binary_cross_entropy_with_logits(logits, loss_bce_target)
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(logits), loss_multi_target)

        current_ddi = self._compute_current_ddi(logits)

        if current_ddi <= self.target_ddi:
            loss = 0.95 * loss_bce + 0.05 * loss_multi
        else:
            beta = self.kp * (1 - (current_ddi / self.target_ddi))
            beta = min(math.exp(beta), 1)
            loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

        return loss

    def _compute_current_ddi(self, logits: torch.Tensor) -> float:
        if self._ddi_adj is None:
            return 0.0
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
        preds = (probs >= self.threshold).astype(np.float32)
        y_label = np.where(preds == 1)[0].tolist()
        try:
            ddi_adj = self._ddi_adj
            if isinstance(ddi_adj, str):
                import dill
                ddi_adj = dill.load(open(ddi_adj, "rb"))
            all_cnt = 0
            dd_cnt = 0
            for i, med_i in enumerate(y_label):
                for j, med_j in enumerate(y_label):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
                        dd_cnt += 1
            return float(dd_cnt / all_cnt) if all_cnt > 0 else 0.0
        except Exception:
            return 0.0

    def predict(self, outputs: Dict[str, Any]) -> torch.Tensor:
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).float()
        return preds

    def decode(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Prediction:
        logits = outputs["logits"]
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
        preds = (probs >= self.threshold).astype(np.float32)
        med_indices = np.where(preds == 1)[0].tolist()
        ranked = np.argsort(probs)[::-1].tolist()
        target = batch.get("target_indices", [])
        return Prediction(
            med_indices=med_indices,
            med_scores=np.asarray(probs, dtype=float),
            target=target,
            task="multilabel",
            ranked_med_indices=ranked,
        )


# =============================================================================
# Data preprocessing utilities
# =============================================================================

def _build_graph(data_list):
    transformed_data = {
        'patient': {}, 'diagnosis': {}, 'procedure': {},
        'medication': {}, 'temporal_feature': {}, 'label': {}
    }
    for patient_id, visits in enumerate(data_list):
        transformed_data['patient'][patient_id] = {
            'diagnosis': {}, 'procedure': {}, 'medication': {}
        }
        for visit in visits[:-1]:
            diagnosis_list, procedure_list, medication_list, timestamp = visit
            for diagnosis_id in diagnosis_list:
                if diagnosis_id not in transformed_data['diagnosis']:
                    transformed_data['diagnosis'][diagnosis_id] = {}
                if patient_id not in transformed_data['diagnosis'][diagnosis_id]:
                    transformed_data['diagnosis'][diagnosis_id][patient_id] = []
                if diagnosis_id not in transformed_data['patient'][patient_id]['diagnosis']:
                    transformed_data['patient'][patient_id]['diagnosis'][diagnosis_id] = []
                transformed_data['diagnosis'][diagnosis_id][patient_id].append(timestamp)
                transformed_data['patient'][patient_id]['diagnosis'][diagnosis_id].append(timestamp)
            for procedure_id in procedure_list:
                if procedure_id not in transformed_data['procedure']:
                    transformed_data['procedure'][procedure_id] = {}
                if patient_id not in transformed_data['procedure'][procedure_id]:
                    transformed_data['procedure'][procedure_id][patient_id] = []
                if procedure_id not in transformed_data['patient'][patient_id]['procedure']:
                    transformed_data['patient'][patient_id]['procedure'][procedure_id] = []
                transformed_data['procedure'][procedure_id][patient_id].append(timestamp)
                transformed_data['patient'][patient_id]['procedure'][procedure_id].append(timestamp)
            for medication_id in medication_list:
                if medication_id not in transformed_data['medication']:
                    transformed_data['medication'][medication_id] = {}
                if patient_id not in transformed_data['medication'][medication_id]:
                    transformed_data['medication'][medication_id][patient_id] = []
                if medication_id not in transformed_data['patient'][patient_id]['medication']:
                    transformed_data['patient'][patient_id]['medication'][medication_id] = []
                transformed_data['medication'][medication_id][patient_id].append(timestamp)
                transformed_data['patient'][patient_id]['medication'][medication_id].append(timestamp)
        if len(visits[-1]) >= 4:
            transformed_data['temporal_feature'][patient_id] = visits[-1][3]
        else:
            transformed_data['temporal_feature'][patient_id] = 0
        transformed_data['label'][patient_id] = visits[-1][2]
    return transformed_data


def _generate_graph_samples(data_dict, left, right, origin_data):
    sample_neighbors_num = 1000
    all_samples = []
    for patient_id in range(left, right):
        hops_nodes_indices_list = []
        hops_nodes_temporal_features_list = []
        central_node_temporal_feature = data_dict['temporal_feature'][patient_id]
        central_node_label = data_dict['label'][patient_id]
        last_node_indices = []
        patient_set = set()

        for hop in range(4):
            if hop == 0:
                node_indices = [int(patient_id)]
                node_temporal_features = data_dict['temporal_feature'][patient_id]
                last_node_indices = [patient_id]
            else:
                node_indices = []
                node_temporal_features = []
                tmp_last_node_indices = []

                if hop % 2 == 0:
                    for type_key in ['diagnosis', 'procedure', 'medication']:
                        for item_idx in last_node_indices:
                            item_dict = data_dict[type_key]
                            if item_idx in item_dict:
                                select_user_idx = list(item_dict[item_idx].keys())
                                if 0 < sample_neighbors_num < len(select_user_idx):
                                    select_user_idx = random.sample(select_user_idx, sample_neighbors_num)
                                select_user_idx = [x for x in select_user_idx if x not in patient_set]
                                tmp_last_node_indices += select_user_idx
                                for user_idx in select_user_idx:
                                    temporal_features_list = item_dict[item_idx][user_idx]
                                    node_indices += [int(user_idx)] * len(temporal_features_list)
                                    node_temporal_features += temporal_features_list
                    patient_set.update(tmp_last_node_indices)
                    last_node_indices = list(set(tmp_last_node_indices))
                else:
                    for user_idx in last_node_indices:
                        for type_key in ['diagnosis', 'procedure', 'medication']:
                            user_dict = data_dict['patient']
                            if user_idx in user_dict and type_key in user_dict[user_idx]:
                                select_item_idx = list(user_dict[user_idx][type_key].keys())
                                if hop != 1 and 0 < sample_neighbors_num < len(select_item_idx):
                                    select_item_idx = random.sample(select_item_idx, sample_neighbors_num)
                                for item_idx in select_item_idx:
                                    temporal_features_list = user_dict[user_idx][type_key][item_idx]
                                    node_indices += [int(item_idx)] * len(temporal_features_list)
                                    node_temporal_features += temporal_features_list
                    last_node_indices = list(set(node_indices))

            hops_nodes_indices_list.append(node_indices)
            hops_nodes_temporal_features_list.append(node_temporal_features)

        diagnosis_list = [i[0] for i in origin_data[patient_id]]
        procedure_list = [i[1] for i in origin_data[patient_id]]
        time_list = [i[3] for i in origin_data[patient_id]]

        all_samples.append((
            hops_nodes_indices_list,
            hops_nodes_temporal_features_list,
            central_node_temporal_feature,
            central_node_label,
            diagnosis_list,
            procedure_list,
            time_list,
        ))
    return all_samples


def preprocess_mrdtr_data(records, data_dir: str, prefix: str = "mrdtr"):
    graph = _build_graph(records)
    n = len(records)
    split_point = int(n * 2 / 3)
    eval_len = int((n - split_point) / 2)

    splits = {
        'train': (0, split_point),
        'eval': (split_point, split_point + eval_len),
        'test': (split_point + eval_len, n),
    }

    for name, (left, right) in splits.items():
        samples = _generate_graph_samples(graph, left, right, records)
        path = os.path.join(data_dir, f"{prefix}_{name}_{right - left}.pkl")
        with open(path, "wb") as f:
            for sample in samples:
                pickle.dump(sample, f)
        print(f"  Saved {name} data ({len(samples)} samples) to {path}", flush=True)

    return splits


def load_mrdtr_data(data_dir: str, prefix: str, split_name: str, num_samples: int):
    path = os.path.join(data_dir, f"{prefix}_{split_name}_{num_samples}.pkl")
    samples = []
    with open(path, "rb") as f:
        for _ in range(num_samples):
            try:
                samples.append(pickle.load(f))
            except EOFError:
                break
    return samples


def mrdtr_collate(sample):
    """
    Convert a pre-processed MR-DTR sample tuple into a DREAM batch dict.
    """
    (hops_nodes_indices, hops_nodes_temporal_features,
     central_nodes_temporal_feature, central_node_label,
     diagnosis_list, procedure_list, time_list) = sample

    return {
        "hops_nodes_indices": hops_nodes_indices,
        "hops_nodes_temporal_features": hops_nodes_temporal_features,
        "central_nodes_temporal_feature": central_nodes_temporal_feature,
        "diagnosis_list": diagnosis_list,
        "procedure_list": procedure_list,
        "time_list": time_list,
        "target_indices": central_node_label,
        "sample": None,
        "visit": [diagnosis_list, procedure_list, central_node_label, time_list[-1] if time_list else 0],
    }


# =============================================================================
# Builder function (for registry)
# =============================================================================

def build_mrdtr_model(
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[MRDTRModel, Dict[str, Any]]:
    data_dir = config["dataset"]["data_dir"]
    ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
    ehr_adj_path = os.path.join(data_dir, config["dataset"].get("ehr_adj_file", "ehr_adj_final.pkl"))
    voc_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
    records_path = os.path.join(data_dir, config["dataset"]["records_file"])

    voc = load_pickle(voc_path)
    med_voc = voc["med_voc"]
    diag_voc = voc["diag_voc"]
    pro_voc = voc["pro_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    ddi_adj = load_pickle(ddi_adj_path)
    ehr_adj = load_pickle(ehr_adj_path) if os.path.exists(ehr_adj_path) else np.eye(voc_size[2])
    records = load_pickle(records_path)

    model = MRDTRModel(
        voc_size=voc_size,
        num_users=len(records),
        emb_dim=config["model"].get("emb_dim", 64),
        dropout=config["model"].get("dropout", 0.2),
        temporal_information_importance=config["model"].get("temporal_information_importance", 0.5),
        ehr_adj=ehr_adj,
        ddi_adj=ddi_adj,
        target_ddi=config["model"].get("target_ddi", 0.06),
        kp=config["model"].get("kp", 2.5),
        threshold=config["evaluation"].get("threshold", 0.5),
        device=device,
    )

    meta = {
        "vocab_size": voc_size,
        "voc": voc,
        "n_patients": len(records),
    }
    return model, meta


__all__ = [
    "MRDTRModel",
    "TimeRec_GCN",
    "build_mrdtr_model",
    "preprocess_mrdtr_data",
    "load_mrdtr_data",
    "mrdtr_collate",
]

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy
import dill
from typing import Any,Dict,List,Optional,Sequence,Tuple

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


class Embeddings(nn.Module):
    def __init__(self,hidden_size,vocab_size,max_position_size,max_segment,word_emb_padding_idx,dropout_rate):
        super(Embeddings,self).__init__()
        self.word_embeddings=nn.Embedding(vocab_size,hidden_size,padding_idx=word_emb_padding_idx)
        self.position_embeddings=nn.Embedding(max_position_size,hidden_size,padding_idx=max_position_size-1)
        self.segment_embeddings=nn.Embedding(max_segment,hidden_size,padding_idx=max_segment-1)

        self.LayerNorm=nn.LayerNorm(hidden_size)
        self.dropout=nn.Dropout(dropout_rate)
        self.max_seq_len=max_position_size
        self.padding_idx_list=[word_emb_padding_idx,max_position_size-1,max_segment-1]

    def padding(self,input_ids,padding_idx):
        if input_ids.shape[1]<self.max_seq_len:
            input_ids=F.pad(input_ids[0],(0,self.max_seq_len-input_ids.shape[1]),'constant',padding_idx).unsqueeze(0)
        padding_mask=(input_ids==padding_idx)
        return input_ids,padding_mask

    def forward(self,input_ids_list):
        input_ids_list=[idx[:,1:] if i>0 else idx for i,idx in enumerate(input_ids_list)]
        seq_length=[idx.shape[1] for idx in input_ids_list]
        position_ids=[torch.arange(seq_length[i],dtype=torch.long,device=input_ids_list[i].device) for i in
                      range(len(seq_length))]
        position_ids=torch.cat([position_ids[i].unsqueeze(0) for i in range(len(position_ids))],dim=-1).expand_as(
            torch.cat(input_ids_list,dim=-1))
        segment_ids=[torch.zeros(seq_length[i],dtype=torch.long,device=input_ids_list[i].device)+k for k,i in
                     enumerate(range(len(seq_length)))]
        segment_ids=torch.cat([segment_ids[i].unsqueeze(0) for i in range(len(segment_ids))],dim=-1).expand_as(
            torch.cat(input_ids_list,dim=-1))
        input_ids_list=torch.cat(input_ids_list,dim=-1)
        input_ids_list,padding_mask=self.padding(input_ids_list,self.padding_idx_list[0])
        position_ids,_=self.padding(position_ids,self.padding_idx_list[1])
        segment_ids,_=self.padding(segment_ids,self.padding_idx_list[2])

        words_embeddings=self.word_embeddings(input_ids_list)
        position_embeddings=self.position_embeddings(position_ids)
        segment_embeddings=self.segment_embeddings(segment_ids)

        embeddings=words_embeddings+position_embeddings+segment_embeddings
        embeddings=self.LayerNorm(embeddings)
        embeddings=self.dropout(embeddings)
        return embeddings,padding_mask


class Transformer_Encoder(nn.Module):
    def __init__(self,n_layer,hidden_size,num_attention_heads,vocab_size,max_position_size,max_segment,
                 word_emb_padding_idx,dropout):
        super(Transformer_Encoder,self).__init__()
        self.emb=Embeddings(hidden_size,vocab_size,max_position_size,max_segment,word_emb_padding_idx,dropout)
        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size,nhead=num_attention_heads,
                                                 dim_feedforward=hidden_size*4,dropout=dropout)
        encoder_norm=nn.LayerNorm(hidden_size)
        self.encoder=nn.TransformerEncoder(encoder_layer,n_layer,encoder_norm)

    def forward(self,input_ids_list):
        emb,padding_mask=self.emb(input_ids_list)
        emb=self.encoder(emb.transpose(0,1),src_key_padding_mask=padding_mask).transpose(0,1)
        return emb


def _get_clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Transformer_Encoder2(nn.Module):
    def __init__(self,n_layer1,n_layer2,hidden_size,num_attention_heads,vocab_size,max_position_size1,
                 max_position_size2,max_segment,word_emb_padding_idx,dropout):
        super(Transformer_Encoder2,self).__init__()
        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size,nhead=num_attention_heads,
                                                 dim_feedforward=hidden_size*4,dropout=dropout)
        encoder_norm=nn.LayerNorm(hidden_size)
        self.encoder1=Transformer_Encoder(n_layer1,hidden_size,num_attention_heads,vocab_size,max_position_size1,
                                          max_segment,word_emb_padding_idx,dropout)
        self.encoder2=nn.TransformerEncoder(encoder_layer,n_layer2,encoder_norm)
        self.max_position_size2=max_position_size2

    def forward(self,input_med_ids_list):
        emb_list=[]
        for input_ids_list in input_med_ids_list:
            emb=self.encoder1(input_ids_list)
            emb_list.append(emb[:,0])
        if len(emb_list)<self.max_position_size2:
            emb_list+=[torch.zeros_like(emb[:,0])]*(self.max_position_size2-len(emb_list))
        emb=torch.cat(emb_list,dim=0).unsqueeze(0)
        padding_mask=torch.BoolTensor(
            [0]*len(input_med_ids_list)+[1]*(self.max_position_size2-len(input_med_ids_list))).unsqueeze(0)
        emb=self.encoder2(emb.transpose(0,1),src_key_padding_mask=padding_mask).transpose(0,1)
        return emb


class Mul_Attention(nn.Module):
    def __init__(self,hidden_size,device):
        super(Mul_Attention,self).__init__()
        self.key=nn.Sequential(nn.ReLU(),nn.Linear(hidden_size,hidden_size))
        self.q=nn.Sequential(nn.ReLU(),nn.Linear(hidden_size,hidden_size))
        self.device=device

    def attention(self,query,key,value,mask=None,dropout=None):
        d_k=query.size(-1)
        scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            assert scores.shape==mask.shape
            scores=scores.masked_fill(mask==0,-1e9)
        p_attn=F.softmax(scores,dim=-1)
        if dropout is not None:
            p_attn=dropout(p_attn)
        return torch.matmul(p_attn,value),p_attn

    def forward(self,input_seq_rep,k_mul):
        shape=input_seq_rep.shape
        input_seq_key=self.key(input_seq_rep)
        input_seq_q=self.q(input_seq_rep)
        mask=torch.zeros((shape[0],shape[1],shape[1])).to(self.device)
        for i in range(shape[1]):
            for j in range(i-k_mul,i+1):
                if j>=0:
                    mask[0,i,j]=1

        out,attn=self.attention(input_seq_q,input_seq_key,input_seq_key,mask=mask)
        return out,attn


class DrugRec_nosymCore(nn.Module):
    def __init__(
            self,
            voc_size: Tuple[int,int,int],
            fix_smi_rep: bool,
            ddi_adj: np.ndarray,
            input_smiles_init_rep,
            emb_dim: int = 256,
            device: torch.device = torch.device('cpu:0')):
        super(DrugRec_nosymCore,self).__init__()

        self.device=device
        trans_dim=emb_dim
        self.num_diag=voc_size[0]
        self.num_proc=voc_size[1]
        self.num_med=voc_size[2]
        vocab_size_for_transformer=max(self.num_diag,self.num_proc)
        self.transformer_dp=Transformer_Encoder(
            n_layer=2,hidden_size=trans_dim,num_attention_heads=4,
            vocab_size=vocab_size_for_transformer,max_position_size=576,
            max_segment=40,word_emb_padding_idx=0,dropout=0.1)
        self.lin=nn.Sequential(nn.ReLU(),nn.Linear(trans_dim,emb_dim))
        self.fix_smi_rep=fix_smi_rep
        self.diag_seq_enc=nn.GRU(emb_dim,emb_dim,batch_first=True)
        self.pro_seq_enc=nn.GRU(emb_dim,emb_dim,batch_first=True)

        self.query=nn.Sequential(
            nn.ReLU(),
            nn.Linear(2*emb_dim,emb_dim)
        )

        self.mlp=nn.Sequential(
            nn.Linear(1024,emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim,emb_dim)
        )
        print(voc_size[0])
        print(voc_size[1])
        print(voc_size[2])
        if self.fix_smi_rep:
            init_rep=input_smiles_init_rep.to(device=self.device)
            if voc_size[2]<=init_rep.shape[0]:
                self.med_rep=init_rep[:voc_size[2]]
            else:
                existing_rep=init_rep
                extra_count=voc_size[2]-init_rep.shape[0]
                extra_rep=nn.Embedding(extra_count,init_rep.shape[1]).weight.to(device=self.device)
                self.med_rep=torch.cat([existing_rep,extra_rep],dim=0)
        else:
            init_rep=torch.tensor(input_smiles_init_rep,requires_grad=True).to(device=self.device)
            if voc_size[2]<=init_rep.shape[0]:
                self.med_rep=init_rep[:voc_size[2]]
            else:
                existing_rep=init_rep
                extra_count=voc_size[2]-init_rep.shape[0]
                extra_rep=nn.Parameter(torch.randn(extra_count,init_rep.shape[1],device=self.device)*0.01,
                                       requires_grad=True)
                self.med_rep=torch.cat([existing_rep,extra_rep],dim=0)

        actual_med_size=self.med_rep.shape[0]
        self.tensor_ddi_adj=torch.FloatTensor(ddi_adj[:actual_med_size,:actual_med_size]).to(self.device)

    def _to_tensor(self,data):
        if isinstance(data,torch.Tensor):
            t=data.to(self.device)
        elif isinstance(data,np.ndarray):
            t=torch.from_numpy(data).to(self.device)
        elif isinstance(data,(list,tuple)):
            if len(data)==0:
                return torch.zeros(1,1,dtype=torch.long,device=self.device)
            if isinstance(data[0],(list,tuple,np.ndarray)):
                max_len=max(len(d) for d in data)
                padded=[]
                for d in data:
                    padded.append(list(d)+[0]*(max_len-len(d)))
                t=torch.tensor(padded,dtype=torch.long,device=self.device)
            else:
                t=torch.tensor([data],dtype=torch.long,device=self.device)
        else:
            t=torch.tensor([data],dtype=torch.long,device=self.device)

        if t.dim()==1:
            t=t.unsqueeze(0)
        return t

    def forward(self,input,type):
        DIAG_SEQ,PRO_SEQ=[],[]

        for j,adm in enumerate(input):
            diags=adm[0] if len(adm)>0 else []
            procs=adm[1] if len(adm)>1 else []
            if diags and len(diags)>0:
                diag_ids_list=[self._to_tensor(d) for d in diags]
            else:
                diag_ids_list=[torch.zeros(1,1,dtype=torch.long,device=self.device)]

            if procs and len(procs)>0:
                pro_ids_list=[self._to_tensor(p) for p in procs]
            else:
                pro_ids_list=[torch.zeros(1,1,dtype=torch.long,device=self.device)]
            diag_representations=self.transformer_dp(diag_ids_list)
            pro_representations=self.transformer_dp(pro_ids_list)
            diag_rep=diag_representations.mean(dim=1).unsqueeze(dim=0)
            pro_rep=pro_representations.mean(dim=1).unsqueeze(dim=0)
            DIAG_SEQ.append(diag_rep)
            PRO_SEQ.append(pro_rep)

        diag_seq=self.lin(torch.cat(DIAG_SEQ,dim=1))
        pro_seq=self.lin(torch.cat(PRO_SEQ,dim=1))

        diag_out,_=self.diag_seq_enc(diag_seq)
        pro_out,_=self.pro_seq_enc(pro_seq)
        patient_representations=torch.cat([diag_out,pro_out],dim=-1)
        query=self.query(patient_representations[:,-1,:])

        med_rep=self.mlp(self.med_rep)
        result=torch.mm(query,med_rep.t())

        neg_pred_prob=F.sigmoid(result)
        neg_pred_prob=torch.einsum("nc,nk->nck",[neg_pred_prob,neg_pred_prob])

        batch_neg=1/self.tensor_ddi_adj.shape[0]*neg_pred_prob.mul(self.tensor_ddi_adj).sum(dim=[1,2]).mean()

        if type=='train':
            return result,neg_pred_prob,batch_neg
        else:
            return result,batch_neg


class DrugRec_nosym(BaseDrugRecommendationModel):
    def __init__(
            self,
            vocab_size:Tuple[int,int,int],
            ddi_adj:np.ndarray,
            emb_dim:int = 256,
            target_ddi:float = 0.05,
            kp: float = 0.05,
            w_ddi: float = 0.5,
            decay: float = 0.0,
            fix_smi_rep: bool = True,
            threshold: float = 0.5,
            device: torch.device = torch.device('cpu:0')) -> None:
        super().__init__(device=device)
        self.model_type="multilabel"
        self.vocab_size=vocab_size
        self.ddi_adj=ddi_adj
        self.emb_dim=emb_dim
        self.target_ddi=target_ddi
        self.kp=kp
        self.w_ddi=w_ddi
        self.threshold=threshold

        self.core=DrugRec_nosymCore(
            voc_size=vocab_size,
            fix_smi_rep=fix_smi_rep,
            ddi_adj=ddi_adj,
            input_smiles_init_rep=dill.load(open('drugrec_benchmark/data/input_smiles_init_rep_iii.pkl','rb')),
            emb_dim=emb_dim,
            device=self.device)

        self._build_pair_dict()
        pair_size=int(vocab_size[2]*(vocab_size[2]-1)/2)
        self.register_buffer(
            'trans_pair_indices',
            torch.stack([
                torch.arange(pair_size,dtype=torch.long),
                torch.arange(pair_size,dtype=torch.long)
            ]))
        trans_p1,trans_p2=torch.zeros(pair_size,dtype=torch.long),torch.zeros(pair_size,dtype=torch.long)
        k=0
        for i in range(vocab_size[2]):
            for j in range(vocab_size[2]):
                if j>i:
                    trans_p1[k]=i
                    trans_p2[k]=j
                    k+=1
        self.register_buffer('trans_p1',trans_p1)
        self.register_buffer('trans_p2',trans_p2)

    def _build_pair_dict(self):
        pair_dict={}
        k=0
        for i in range(self.vocab_size[2]):
            for j in range(self.vocab_size[2]):
                if j>i:
                    pair_dict[(i,j)]=k
                    k+=1
        self.pair_dict=pair_dict
        self.pair_size=int(self.vocab_size[2]*(self.vocab_size[2]-1)/2)

    def forward(self,batch: Dict[str,Any]) -> Dict[str,torch.Tensor]:
        patients=self.get_patients(batch)
        if not patients or not patients[0]:
            return {
                "logits":torch.empty((0,self.vocab_size[2]),device=self.device),
                "ddi_loss":torch.zeros((1,),device=self.device),
                "neg_pred_prob":torch.empty((0,self.vocab_size[2],self.vocab_size[2]),device=self.device),
            }
        patient=patients[0]
        logits,neg_pred_prob,ddi_penalty=self.core(patient,'train')
        return {
            "logits":logits,
            "ddi_loss":ddi_penalty.unsqueeze(0),
            "neg_pred_prob":neg_pred_prob,
        }

    def compute_loss(self,outputs: Dict[str,torch.Tensor],batch: Dict[str,Any]) -> torch.Tensor:
        logits=outputs["logits"]
        ddi_penalty=outputs["ddi_loss"]
        neg_pred_prob=outputs["neg_pred_prob"]

        if logits.shape[0]==0:
            return torch.tensor(0.0,device=self.device)

        targets=self.build_target(batch)
        target_multi=self.build_multilabel_target(targets)

        loss_bce=F.binary_cross_entropy_with_logits(logits,targets)
        loss_multi=F.multilabel_margin_loss(F.sigmoid(logits),target_multi)

        result_pair=neg_pred_prob[:,self.trans_p1,self.trans_p2]
        loss_bce_pair_target=self._build_pair_target(batch)
        loss_bce_pair=F.binary_cross_entropy(result_pair,loss_bce_pair_target)

        pred_labels=self._predict_labels(logits)
        current_ddi_rate=self._ddi_rate_from_labels(pred_labels)

        if current_ddi_rate<=self.target_ddi:
            loss=loss_bce+0.1*loss_multi+loss_bce_pair
        else:
            loss=loss_bce+0.1*loss_multi+self.w_ddi*ddi_penalty[0]+loss_bce_pair

        return loss

    def predict(self,outputs: Dict[str,torch.Tensor]) -> torch.Tensor:
        logits=outputs["logits"]
        if logits.shape[0]==0:
            return torch.empty_like(logits)
        probs=torch.sigmoid(logits)
        return (probs>=self.threshold).float()

    def _build_pair_target(self,batch: Dict[str,Any]) -> torch.Tensor:
        target_rows=self.get_target_indices_list(batch)
        batch_size=len(target_rows)
        loss_bce_pair_target=torch.zeros((batch_size,self.pair_size),device=self.device)
        for idx,med_set in enumerate(target_rows):
            sorted_meds=sorted(med_set)
            for i_pos in range(len(sorted_meds)):
                for j_pos in range(i_pos+1,len(sorted_meds)):
                    med_i,med_j=sorted_meds[i_pos],sorted_meds[j_pos]
                    if (med_i,med_j) in self.pair_dict:
                        loss_bce_pair_target[idx,self.pair_dict[(med_i,med_j)]]=1
        return loss_bce_pair_target

    def _predict_labels(self,logits: torch.Tensor) -> List[int]:
        probs=torch.sigmoid(logits).detach().cpu().numpy()[0]
        preds=(probs>=self.threshold).astype(np.int32)
        return np.where(preds==1)[0].tolist()

    def _ddi_rate_from_labels(self,labels: List[int]) -> float:
        if len(labels)<2:
            return 0.0
        ddi_count=0
        total_count=0
        for i,med_i in enumerate(labels):
            for j in range(i+1,len(labels)):
                med_j=labels[j]
                total_count+=1
                if self.ddi_adj[med_i,med_j]==1 or self.ddi_adj[med_j,med_i]==1:
                    ddi_count+=1
        if total_count==0:
            return 0.0
        return ddi_count/total_count

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging

import os
import copy
import json
import math
import random
import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
import numpy as np
import torch.nn.functional as F
from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel
from typing import Any,Dict,List,Optional,Sequence,Tuple


#bert_models
def gelu(x):
    return 0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x+0.044715*torch.pow(x,3))))


class LayerNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm,self).__init__()
        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.bias=nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon=eps

    def forward(self,x):
        u=x.mean(-1,keepdim=True)
        s=(x-u).pow(2).mean(-1,keepdim=True)
        x=(x-u)/torch.sqrt(s+self.variance_epsilon)
        return self.weight*x+self.bias


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self,query,key,value,mask=None,dropout=None):
        scores=torch.matmul(query,key.transpose(-2,-1))\
               /math.sqrt(query.size(-1))

        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)

        p_attn=F.softmax(scores,dim=-1)

        if dropout is not None:
            p_attn=dropout(p_attn)

        return torch.matmul(p_attn,value),p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self,hidden_size,num_attention_heads,attention_probs_dropout_prob):
        super().__init__()
        assert hidden_size%num_attention_heads==0

        # We assume d_v always equals d_k
        self.d_k=hidden_size//num_attention_heads
        self.h=num_attention_heads

        self.linear_layers=nn.ModuleList(
            [nn.Linear(hidden_size,hidden_size,bias=False) for _ in range(3)])
        self.output_linear=nn.Linear(hidden_size,hidden_size)
        self.attention=Attention()

        self.dropout=nn.Dropout(p=attention_probs_dropout_prob)

    def forward(self,query,key,value,mask=None):
        batch_size=query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query,key,value=[l(x).view(batch_size,-1,self.h,self.d_k).transpose(1,2)
                         for l,x in zip(self.linear_layers,(query,key,value))]

        # 2) Apply attention on all the projected vectors in batch.
        x,attn=self.attention(
            query,key,value,mask=mask,dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x=x.transpose(1,2).contiguous().view(
            batch_size,-1,self.h*self.d_k)

        return self.output_linear(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self,hidden_size,hidden_dropout_prob):
        super(SublayerConnection,self).__init__()
        self.norm=LayerNorm(hidden_size)
        self.dropout=nn.Dropout(hidden_dropout_prob)

    def forward(self,x,sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x+self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self,hidden_size,intermediate_size,hidden_dropout_prob):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1=nn.Linear(hidden_size,intermediate_size)
        self.w_2=nn.Linear(intermediate_size,hidden_size)
        self.dropout=nn.Dropout(hidden_dropout_prob)

    def forward(self,x):
        return self.w_2(self.dropout(gelu(self.w_1(x))))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self,hidden_dropout_prob,hidden_size,num_attention_heads,attention_probs_dropout_prob,
                 intermediate_size):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention=MultiHeadedAttention(hidden_size,num_attention_heads,attention_probs_dropout_prob)
        self.feed_forward=PositionwiseFeedForward(hidden_size,intermediate_size,hidden_dropout_prob)
        self.input_sublayer=SublayerConnection(hidden_size,hidden_dropout_prob)
        self.output_sublayer=SublayerConnection(hidden_size,hidden_dropout_prob)
        self.dropout=nn.Dropout(p=hidden_dropout_prob)

    def forward(self,x,mask):
        x=self.input_sublayer(
            x,lambda _x:self.attention.forward(_x,_x,_x,mask=mask))
        x=self.output_sublayer(x,self.feed_forward)
        return self.dropout(x)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, visit and token_type embeddings.
    """

    def __init__(self,vocab_size,hidden_size,hidden_dropout_prob):
        super(BertEmbeddings,self).__init__()
        self.word_embeddings=nn.Embedding(
            vocab_size,hidden_size)
        self.token_type_embeddings=nn.Embedding(2,hidden_size)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm=LayerNorm(hidden_size,eps=1e-12)
        self.dropout=nn.Dropout(hidden_dropout_prob)

    def forward(self,input_ids,token_type_ids=None):
        if token_type_ids is None:
            token_type_ids=torch.zeros_like(input_ids)

        words_embeddings=self.word_embeddings(input_ids)

        embeddings=words_embeddings+self.token_type_embeddings(token_type_ids)
        embeddings=self.LayerNorm(embeddings)
        embeddings=self.dropout(embeddings)
        return embeddings


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self,vocab_size,hidden_size,num_hidden_layers,
                 num_attention_heads,intermediate_size,hidden_act,
                 hidden_dropout_prob,attention_probs_dropout_prob,
                 max_position_embeddings,type_vocab_size,initializer_range,
                 graph,graph_hidden_size,graph_heads,*inputs,**kwargs,):
        super(PreTrainedBertModel,self).__init__()
        self.initializer_range=initializer_range
        self.vocab_size=vocab_size
        self.hidden_size=hidden_size
        self.hidden_dropout_prob=hidden_dropout_prob
        self.num_attention_heads=num_attention_heads
        self.attention_probs_dropout_prob=attention_probs_dropout_prob
        self.max_position_embeddings=max_position_embeddings
        self.type_vocab_size=type_vocab_size
        self.graph=graph
        self.graph_hidden_size=graph_hidden_size
        self.graph_heads=graph_heads
        self.intermediate_size=intermediate_size
        self.hidden_act=hidden_act
        self.num_hidden_layers=num_hidden_layers

    def init_bert_weights(self,module):
        """ Initialize the weights.
        """
        if isinstance(module,(nn.Linear,nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0,std=self.initializer_range)
        elif isinstance(module,LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module,nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BERT(PreTrainedBertModel):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self,vocab_size,hidden_size,num_hidden_layers,
                 num_attention_heads,intermediate_size,hidden_act,
                 hidden_dropout_prob,attention_probs_dropout_prob,
                 max_position_embeddings,type_vocab_size,initializer_range,
                 graph,graph_hidden_size,graph_heads,dx_voc=None,rx_voc=None):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__(vocab_size,hidden_size,num_hidden_layers,
                         num_attention_heads,intermediate_size,hidden_act,
                         hidden_dropout_prob,attention_probs_dropout_prob,
                         max_position_embeddings,type_vocab_size,initializer_range,
                         graph,graph_hidden_size,graph_heads)
        if graph:
            assert dx_voc is not None
            assert rx_voc is not None

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding=BertEmbeddings(vocab_size,hidden_size,hidden_dropout_prob)\
            if graph else BertEmbeddings(vocab_size,hidden_size,hidden_dropout_prob)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks=nn.ModuleList(
            [TransformerBlock(hidden_dropout_prob,hidden_size,num_attention_heads,attention_probs_dropout_prob,
                              intermediate_size) for _ in range(num_hidden_layers)])

        # pool first output
        # self.pooler = BertPooler(config)

        self.apply(self.init_bert_weights)

    def forward(self,x,token_type_ids=None,input_positions=None,input_sides=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # (2*admin, max_seq_len) --> (2*admin, 1, max_seq_len, max_seq_len)
        # x: (bs, 2, max_seq_len) --> (bs, 2, 1, max_seq_len, max_seq_len)
        #mask = (x > 1).unsqueeze(2).repeat(1, 1, x.size(2), 1).unsqueeze(2)
        mask=(x>1).unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x=self.embedding(x,token_type_ids)  # (bs, 2, max_seq_len, hidden_size)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x=transformer.forward(x,mask)

        return x,x[:,0]  # like rnn, only return the last hidden state


class BertPooler(nn.Module):
    def __init__(self,hidden_size,):
        super(BertPooler,self).__init__()
        self.dense=nn.Linear(hidden_size,hidden_size)
        self.activation=nn.Tanh()

    def forward(self,hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor=hidden_states[:,0]
        pooled_output=self.dense(first_token_tensor)
        pooled_output=self.activation(pooled_output)
        return pooled_output


# pretaining
class BertPredictionHeadTransform(nn.Module):
    def __init__(self,hidden_size):
        super(BertPredictionHeadTransform,self).__init__()

        self.dense=nn.Linear(hidden_size,hidden_size)
        self.transform_act_fn=gelu
        self.LayerNorm=LayerNorm(hidden_size,eps=1e-12)

    def forward(self,hidden_states):
        hidden_states=self.dense(hidden_states)
        hidden_states=self.transform_act_fn(hidden_states)
        hidden_states=self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self,hidden_size,voc_size=None):
        super(BertLMPredictionHead,self).__init__()
        self.transform=BertPredictionHeadTransform(hidden_size)

        self.decoder=nn.Linear(hidden_size,voc_size)

    def forward(self,hidden_states):
        hidden_states=self.transform(hidden_states)
        hidden_states=self.decoder(hidden_states)
        return hidden_states


class SelfSupervisedHead(nn.Module):
    def __init__(self,hidden_size,dx_voc_size,rx_voc_size):
        super(SelfSupervisedHead,self).__init__()
        self.multi_cls=nn.ModuleList([ClsHead(hidden_size,dx_voc_size),ClsHead(
            hidden_size,dx_voc_size),ClsHead(hidden_size,rx_voc_size),ClsHead(hidden_size,rx_voc_size)])

    def forward(self,dx_inputs,rx_inputs):
        # inputs (B, hidden)
        # output logits
        return self.multi_cls[0](dx_inputs),self.multi_cls[1](rx_inputs),self.multi_cls[2](dx_inputs),self.multi_cls[3](
            rx_inputs)


class MappingHead(nn.Module):
    def __init__(self,hidden_size):
        super(MappingHead,self).__init__()
        self.dense=nn.Sequential(nn.Linear(hidden_size,hidden_size),
                                 nn.ReLU())

    def forward(self,input):
        return self.dense(input)


class PromptBERT(PreTrainedBertModel):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self,vocab_size,hidden_size,num_hidden_layers,
                 num_attention_heads,intermediate_size,hidden_act,
                 hidden_dropout_prob,attention_probs_dropout_prob,
                 max_position_embeddings,type_vocab_size,initializer_range,
                 graph,graph_hidden_size,graph_heads,dx_voc=None,rx_voc=None):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__(vocab_size,hidden_size,num_hidden_layers,
                         num_attention_heads,intermediate_size,hidden_act,
                         hidden_dropout_prob,attention_probs_dropout_prob,
                         max_position_embeddings,type_vocab_size,initializer_range,
                         graph,graph_hidden_size,graph_heads)
        if graph:
            assert dx_voc is not None
            assert rx_voc is not None

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding=BertEmbeddings(vocab_size,hidden_size,hidden_dropout_prob) if graph else BertEmbeddings(
            vocab_size,hidden_size,hidden_dropout_prob)

        # embedding for prompt
        #self.prompt_embedding = nn.Embedding(2, config.hidden_size)
        self.prompt_embedding=PromptEmbeddings(hidden_size,hidden_dropout_prob)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks=nn.ModuleList(
            [TransformerBlock(hidden_dropout_prob,hidden_size,num_attention_heads,attention_probs_dropout_prob,
                              intermediate_size) for _ in range(num_hidden_layers)])

        # pool first output
        # self.pooler = BertPooler(config)

        self.apply(self.init_bert_weights)

    def forward(self,x,token_type_ids=None,input_positions=None,input_sides=None,input_prompt=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # (2*admin, max_seq_len) --> (2*admin, 1, max_seq_len, max_seq_len)
        # x: (bs, 2, max_seq_len) --> (bs, 2, 1, max_seq_len, max_seq_len)
        #mask = (x > 1).unsqueeze(2).repeat(1, 1, x.size(2), 1).unsqueeze(2)
        #mask = (x > 1).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask=(torch.cat([2*torch.ones((x.shape[0],1),device=x.device),x],dim=1)>1).unsqueeze(1).repeat(1,x.size(1)+1,
                                                                                                       1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x=self.embedding(x,token_type_ids)  # (bs, max_seq_len, hidden_size)

        # get prompt and concat it to original input
        prompt=self.prompt_embedding(input_prompt.long())  # (bs, 1, hidden_size)
        x=torch.cat([x[:,0,:].unsqueeze(1),prompt,x[:,1:,:]],dim=1)  # [CLS] should be put before the prompt
        #x = torch.cat([prompt, x], dim=1)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x=transformer.forward(x,mask)

        return x,x[:,0]  # like rnn, only return the last hidden state


class PromptEmbeddings(nn.Module):
    """Construct the embeddings from word, visit and token_type embeddings.
    """

    def __init__(self,hidden_size,hidden_dropout_prob):
        super(PromptEmbeddings,self).__init__()
        self.word_embeddings=nn.Embedding(2,hidden_size)

        self.LayerNorm=LayerNorm(hidden_size,eps=1e-12)
        self.dropout=nn.Dropout(hidden_dropout_prob)

    def forward(self,input_ids):
        embeddings=self.word_embeddings(input_ids)

        embeddings=self.LayerNorm(embeddings)
        embeddings=self.dropout(embeddings)
        return embeddings


#predictive_models
logger=logging.getLogger(__name__)


def freeze_afterwards(model):
    for p in model.parameters():
        p.requires_grad=False


class ClsHead(nn.Module):
    def __init__(self,hidden_size,voc_size):
        super(ClsHead,self).__init__()
        self.cls=nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.ReLU(
        ),nn.Linear(hidden_size,voc_size))

    def forward(self,input):
        return self.cls(input)


class ClsHeadHos(nn.Module):
    def __init__(self,hidden_size,voc_size):
        super(ClsHeadHos,self).__init__()
        self.cls=nn.Sequential(nn.Linear(hidden_size+20,hidden_size),nn.ReLU(
        ),nn.Linear(hidden_size,voc_size))

    def forward(self,input):
        return self.cls(input)


class SelfSupervisedHeadHos(nn.Module):
    def __init__(self,hidden_size,dx_voc_size,rx_voc_size,hospital_emb_dim=20):
        super(SelfSupervisedHeadHos,self).__init__()
        self.multi_cls=nn.ModuleList([
            ClsHeadHos(hidden_size,dx_voc_size),
            ClsHeadHos(hidden_size,dx_voc_size),
            ClsHeadHos(hidden_size,rx_voc_size),
            ClsHeadHos(hidden_size,rx_voc_size)])

    def forward(self,dx_inputs,rx_inputs):
        # inputs (B, hidden)
        # output logits
        return self.multi_cls[0](dx_inputs),self.multi_cls[1](rx_inputs),self.multi_cls[2](dx_inputs),self.multi_cls[3](
            rx_inputs)


#prompt_models
class PMRec_Prompt(PreTrainedBertModel):
    def __init__(self,vocab_size,hidden_size,num_hidden_layers,
                 num_attention_heads,intermediate_size,hidden_act,
                 hidden_dropout_prob,attention_probs_dropout_prob,
                 max_position_embeddings,type_vocab_size,initializer_range,
                 graph,graph_hidden_size,graph_heads,med_voc_size,prompt_num):
        super().__init__(vocab_size,hidden_size,num_hidden_layers,
                         num_attention_heads,intermediate_size,hidden_act,
                         hidden_dropout_prob,attention_probs_dropout_prob,
                         max_position_embeddings,type_vocab_size,initializer_range,
                         graph,graph_hidden_size,graph_heads)
        self.prompt_num=prompt_num
        self.bert=MultiPromptBERT(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            graph=graph,
            graph_hidden_size=graph_hidden_size,
            graph_heads=graph_heads,
            prompt_num=prompt_num)
        self.dense=nn.ModuleList([MappingHead(hidden_size),MappingHead(hidden_size)])
        self.cls=nn.Sequential(nn.Linear(2*hidden_size,2*hidden_size),
                               nn.ReLU(),nn.Linear(2*hidden_size,med_voc_size))

        self.apply(self.init_bert_weights)

    def forward(self,input_ids,dx_labels=None,rx_labels=None,epoch=None,prompt=None):
        """
        :param input_ids: [2, max_seq_len] (old: [2, adm_num, max_seq_len])
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :return:
        """
        token_types_ids=torch.cat([torch.zeros((input_ids.size(0),1,input_ids.size(2))),torch.ones(
            (input_ids.size(0),1,input_ids.size(2)))],dim=1).long().to(input_ids.device)
        token_types_ids=token_types_ids.repeat(1,
                                               1 if input_ids.size(1)//2==0 else input_ids.size(1)//2,
                                               1)  # (bs, 2, max_seq_len)
        _,dx_bert_pool=self.bert(input_ids[:,0,:],token_types_ids[:,0,:],
                                 input_prompt=prompt[:,0,:])  # (bs, hidden_size)
        _,rx_bert_pool=self.bert(input_ids[:,1,:],token_types_ids[:,1,:],
                                 input_prompt=prompt[:,1,:])  # (bs, hidden_size)
        loss=0

        concat_vector=torch.cat([dx_bert_pool,rx_bert_pool],dim=-1)
        rx_logits=self.cls(concat_vector)

        loss=F.binary_cross_entropy_with_logits(rx_logits,rx_labels)
        return loss,rx_logits


class MultiPromptBERT(PreTrainedBertModel):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self,vocab_size,hidden_size,num_hidden_layers,
                 num_attention_heads,intermediate_size,hidden_act,
                 hidden_dropout_prob,attention_probs_dropout_prob,
                 max_position_embeddings,type_vocab_size,initializer_range,
                 graph,graph_hidden_size,graph_heads,dx_voc=None,rx_voc=None,prompt_num=0):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__(vocab_size,hidden_size,num_hidden_layers,
                         num_attention_heads,intermediate_size,hidden_act,
                         hidden_dropout_prob,attention_probs_dropout_prob,
                         max_position_embeddings,type_vocab_size,initializer_range,
                         graph,graph_hidden_size,graph_heads)

        self.prompt_num=prompt_num
        if graph:
            assert dx_voc is not None
            assert rx_voc is not None

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding=BertEmbeddings(vocab_size,hidden_size,hidden_dropout_prob)

        # embedding for prompt
        self.prompt_embedding_list=nn.ModuleList(
            [PromptEmbeddings(hidden_size,hidden_dropout_prob) for _ in range(self.prompt_num)])

        # multi-layers transformer blocks, deep network
        self.transformer_blocks=nn.ModuleList(
            [TransformerBlock(hidden_dropout_prob,hidden_size,num_attention_heads,attention_probs_dropout_prob,
                              intermediate_size) for _ in range(num_hidden_layers)])


        self.apply(self.init_bert_weights)

    def forward(self,x,token_type_ids=None,input_positions=None,input_sides=None,input_prompt=None):
        original_seq_len=x.size(1)
        mask=(torch.cat(
            [x[:,0].unsqueeze(1),self.prompt_num*torch.ones((x.shape[0],self.prompt_num),device=x.device),x[:,1:]],
            dim=1)>1).unsqueeze(1).repeat(1,original_seq_len+self.prompt_num,1).unsqueeze(1)

        x=self.embedding(x,token_type_ids)

        prompt=[]
        for prompt_embedding in self.prompt_embedding_list:
            prompt.append(prompt_embedding(input_prompt.long()))
        prompt=torch.cat(prompt,dim=1)
        x=torch.cat([x[:,0,:].unsqueeze(1),prompt,x[:,1:,:]],dim=1)

        for transformer in self.transformer_blocks:
            x=transformer.forward(x,mask)

        return x,x[:,0]


class PMRec_Pretrain_Contrastive(PreTrainedBertModel):
    def __init__(self,vocab_size,hidden_size,num_hidden_layers,
                 num_attention_heads,intermediate_size,hidden_act,
                 hidden_dropout_prob,attention_probs_dropout_prob,
                 max_position_embeddings,type_vocab_size,initializer_range,diag_voc_size,pro_voc_size,
                 graph,graph_hidden_size,graph_heads,num_hospitals,hospital_emb_dim=20,tau=0.1,loss_weight=1.0):
        super(PMRec_Pretrain_Contrastive,self).__init__(vocab_size,hidden_size,num_hidden_layers,
                                                        num_attention_heads,intermediate_size,hidden_act,
                                                        hidden_dropout_prob,attention_probs_dropout_prob,
                                                        max_position_embeddings,type_vocab_size,initializer_range,
                                                        graph,graph_hidden_size,graph_heads)

        self.tau=tau
        self.loss_weight=loss_weight
        self.dx_voc_size=diag_voc_size
        self.rx_voc_size=pro_voc_size
        self.hospital_emb_dim=hospital_emb_dim
        self.hospital_emb=nn.Embedding(num_hospitals,hospital_emb_dim)
        self.bert=BERT(vocab_size,hidden_size,num_hidden_layers,
                       num_attention_heads,intermediate_size,hidden_act,
                       hidden_dropout_prob,attention_probs_dropout_prob,
                       max_position_embeddings,type_vocab_size,initializer_range,
                       graph,graph_hidden_size,graph_heads)
        self.cls=SelfSupervisedHeadHos(
            hidden_size,self.dx_voc_size,self.rx_voc_size,hospital_emb_dim)
        self.project_dx=nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size)
        )
        self.project_rx=nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size)
        )

        self.apply(self.init_bert_weights)

    def forward(self,inputs,hospital_ids,dx_labels=None,rx_labels=None,inputs_raw=None):
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        hospital_emb=self.hospital_emb(hospital_ids)
        _,dx_bert_pool=self.bert(inputs[:,0,:],torch.zeros(
            (inputs.size(0),inputs.size(2))).long().to(inputs.device))
        _,rx_bert_pool=self.bert(inputs[:,1,:],torch.zeros(
            (inputs.size(0),inputs.size(2))).long().to(inputs.device))
        _,dx_bert_pool_raw=self.bert(inputs_raw[:,0,:],torch.zeros(
            (inputs_raw.size(0),inputs_raw.size(2))).long().to(inputs_raw.device))
        _,rx_bert_pool_raw=self.bert(inputs_raw[:,1,:],torch.zeros(
            (inputs_raw.size(0),inputs_raw.size(2))).long().to(inputs_raw.device))

        dx_bert_pool_raw=self.project_dx(dx_bert_pool_raw)
        rx_bert_pool_raw=self.project_rx(rx_bert_pool_raw)

        contrastive_loss=Contrastive_Loss(dx_bert_pool_raw,rx_bert_pool_raw,tau=self.tau)+\
                         Contrastive_Loss(rx_bert_pool_raw,dx_bert_pool_raw,tau=self.tau)
        dx_with_hos=torch.cat([dx_bert_pool,hospital_emb],dim=-1)
        rx_with_hos=torch.cat([rx_bert_pool,hospital_emb],dim=-1)
        dx2dx,rx2dx,dx2rx,rx2rx=self.cls(dx_with_hos,rx_with_hos)
        # output logits
        if rx_labels is None or dx_labels is None:
            return F.sigmoid(dx2dx),F.sigmoid(rx2dx),F.sigmoid(dx2rx),F.sigmoid(rx2rx)
        else:
            loss=F.binary_cross_entropy_with_logits(dx2dx,dx_labels)+F.binary_cross_entropy_with_logits(rx2rx,
                                                                                                        rx_labels)+self.loss_weight*contrastive_loss
            # loss = contrastive_loss + self.loss_weight * contrastive_loss

            return loss,F.sigmoid(dx2dx),F.sigmoid(rx2dx),F.sigmoid(dx2rx),F.sigmoid(rx2rx)


def Contrastive_Loss(X,Y,tau):
    '''
    X: (bs, hidden_size), Y: (bs, hidden_size)
    tau: the temperature factor
    '''
    #sim_matrix = X.mm(Y.t())    # (bs, bs)
    sim_matrix=F.cosine_similarity(X.unsqueeze(1),Y.unsqueeze(0),dim=2)
    pos=torch.exp(torch.diag(sim_matrix)/tau).unsqueeze(0)  # (1, bs)
    neg=torch.sum(torch.exp(sim_matrix/tau),dim=0)-pos  # (1, bs)
    loss=- torch.log(pos/neg)
    loss=torch.mean(loss)

    return loss

class TEMPT(BaseDrugRecommendationModel):
    PAD_ID=0
    CLS_ID=1
    MASK_ID=2

    def __init__(
            self,
            vocab_size,
            hidden_size:int,
            num_hidden_layers:int,
            num_attention_heads:int,
            intermediate_size:int,
            hidden_act:str,
            hidden_dropout_prob:float,
            attention_probs_dropout_prob:float,
            max_position_embeddings:int,
            type_vocab_size:int,
            initializer_range:float,
            graph:bool,
            graph_hidden_size:int,
            graph_heads:int,
            threshold: float = 0.5,
            device: Optional[torch.device] = None,
            train_records=None,
            pretrain_epochs: int = 0,
            pretrain_lr: float = 5e-4,
            pretrain_batch_size: int = 64,
            pretrain_max_seq_len: int = 100,
            pretrain_tau: float = 0.1,
            pretrain_loss_weight: float = 1.0,
            prompt_num: int = 4,
            hospital_ids=None,
            unique_hospital_ids=None,
    ) -> None:
        super().__init__(device=device)

        self.vocab_size=vocab_size
        self.diag_voc_size=vocab_size[0]
        self.pro_voc_size=vocab_size[1]
        self.med_voc_size=vocab_size[2]
        self.hidden_size=hidden_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.intermediate_size=intermediate_size
        self.hidden_act=hidden_act
        self.hidden_dropout_prob=hidden_dropout_prob
        self.attention_probs_dropout_prob=attention_probs_dropout_prob
        self.max_position_embeddings=max_position_embeddings
        self.type_vocab_size=type_vocab_size
        self.initializer_range=initializer_range
        self.graph=graph
        self.graph_hidden_size=graph_hidden_size
        self.graph_heads=graph_heads
        self.threshold=threshold

        self.pretrain_epochs=pretrain_epochs
        self.pretrain_lr=pretrain_lr
        self.pretrain_batch_size=pretrain_batch_size
        self.pretrain_max_seq_len=pretrain_max_seq_len
        self.pretrain_tau=pretrain_tau
        self.pretrain_loss_weight=pretrain_loss_weight
        self.prompt_num=prompt_num

        if unique_hospital_ids is not None and len(unique_hospital_ids)>0:
            self.unique_hospital_ids=sorted(unique_hospital_ids)
        else:
            self.unique_hospital_ids=[1]
        self.num_hospitals=len(self.unique_hospital_ids)
        self.hospital_to_idx={h:i for i,h in enumerate(self.unique_hospital_ids)}

        if hospital_ids is not None and len(hospital_ids)>0:
            self.hospital_ids=[int(h) for h in hospital_ids]
        else:
            self.hospital_ids=[self.unique_hospital_ids[0]]*(len(train_records) if train_records else 0)

        self.dx_offset=3
        self.pro_offset=3+self.diag_voc_size
        self.unified_vocab_size=3+self.diag_voc_size+self.pro_voc_size+self.med_voc_size

        self._finetune_model=PMRec_Prompt(
            vocab_size=self.unified_vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act='relu',
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            graph=self.graph,
            graph_hidden_size=self.graph_hidden_size,
            graph_heads=self.graph_heads,
            med_voc_size=self.med_voc_size,
            prompt_num=self.prompt_num
        )
        self._finetune_model.to(self.device)

        if self.pretrain_epochs>0 and train_records is not None:
            self._run_pretrain(train_records)

    def _run_pretrain(self,train_records):

        pretrain_model=PMRec_Pretrain_Contrastive(
            vocab_size=self.unified_vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act='relu',
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            graph=self.graph,
            graph_hidden_size=self.graph_hidden_size,
            graph_heads=self.graph_heads,
            diag_voc_size=self.diag_voc_size,
            pro_voc_size=self.pro_voc_size,
            tau=self.pretrain_tau,
            loss_weight=self.pretrain_loss_weight,
            num_hospitals=self.num_hospitals
        )
        pretrain_model.to(self.device)

        rng=random.Random(42)
        max_seq_len=self.pretrain_max_seq_len
        batch_size=self.pretrain_batch_size

        batches=self._build_pretrain_batches(train_records,max_seq_len,batch_size,rng)
        if not batches:
            return

        optimizer=torch.optim.Adam(pretrain_model.parameters(),lr=self.pretrain_lr)

        for epoch in range(self.pretrain_epochs):
            pretrain_model.train()
            indices=list(range(len(batches)))
            random.shuffle(indices)
            total_loss=0.0
            for idx in indices:
                masked,orig,dx_labels,rx_labels,hos_ids=batches[idx]
                masked=masked.to(self.device)
                orig=orig.to(self.device)
                dx_labels=dx_labels.to(self.device)
                rx_labels=rx_labels.to(self.device)
                hos_ids=hos_ids.to(self.device)
                loss,_,_,_,_=pretrain_model(
                    masked,hospital_ids=hos_ids,dx_labels=dx_labels,rx_labels=rx_labels,inputs_raw=orig)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss+=float(loss.detach().cpu().item())
            avg=total_loss/max(len(batches),1)

        self._transfer_pretrained_weights(pretrain_model)
        del pretrain_model

    def _build_pretrain_batches(self,records,max_seq_len,batch_size,rng):
        batches=[]
        batch_masked,batch_orig,batch_dx,batch_rx,batch_hos=[],[],[],[],[]
        for patient_idx,patient in enumerate(records):
            last_visit=patient[-1]
            diagnoses=list(last_visit[0]) if len(last_visit)>0 else []
            procedures=list(last_visit[1]) if len(last_visit)>1 else []
            if not diagnoses and not procedures:
                continue
            dx_label=np.zeros(self.diag_voc_size,dtype=np.float32)
            rx_label=np.zeros(self.pro_voc_size,dtype=np.float32)
            for d in diagnoses:
                if 0<=d<self.diag_voc_size:
                    dx_label[d]=1.0
            for p in procedures:
                if 0<=p<self.pro_voc_size:
                    rx_label[p]=1.0

            masked_diag=self._visit_to_seq(diagnoses,max_seq_len,apply_mask=True,mask_rng=rng,offset=self.dx_offset)
            masked_proc=self._visit_to_seq(procedures,max_seq_len,apply_mask=True,mask_rng=rng,offset=self.pro_offset)
            orig_diag=self._visit_to_seq(diagnoses,max_seq_len,apply_mask=False,offset=self.dx_offset)
            orig_proc=self._visit_to_seq(procedures,max_seq_len,apply_mask=False,offset=self.pro_offset)

            masked=np.stack([masked_diag,masked_proc],axis=0)  # (2, max_seq_len)
            orig=np.stack([orig_diag,orig_proc],axis=0)  # (2, max_seq_len)

            if patient_idx<len(self.hospital_ids):
                raw_hospital_id=self.hospital_ids[patient_idx]
            else:
                raw_hospital_id=self.unique_hospital_ids[0]
            hospital_idx=self.hospital_to_idx.get(raw_hospital_id,0)

            batch_masked.append(masked)
            batch_orig.append(orig)
            batch_dx.append(dx_label)
            batch_rx.append(rx_label)
            batch_hos.append(hospital_idx)

            if len(batch_masked)==batch_size:
                batches.append((
                    torch.tensor(np.array(batch_masked),dtype=torch.long),
                    torch.tensor(np.array(batch_orig),dtype=torch.long),
                    torch.tensor(np.array(batch_dx),dtype=torch.float),
                    torch.tensor(np.array(batch_rx),dtype=torch.float),
                    torch.tensor(batch_hos,dtype=torch.long),
                ))
                batch_masked,batch_orig,batch_dx,batch_rx,batch_hos=[],[],[],[],[]

        if batch_masked:
            batches.append((
                torch.tensor(np.array(batch_masked),dtype=torch.long),
                torch.tensor(np.array(batch_orig),dtype=torch.long),
                torch.tensor(np.array(batch_dx),dtype=torch.float),
                torch.tensor(np.array(batch_rx),dtype=torch.float),
                torch.tensor(batch_hos,dtype=torch.long),
            ))
        return batches

    def _visit_to_seq(self,codes,max_seq_len,apply_mask=False,mask_rng=None,offset=0):
        token_ids=[self.CLS_ID]
        for c in codes[:max_seq_len-1]:
            token_ids.append(c+offset)

        if apply_mask and mask_rng is not None:
            for i in range(1,len(token_ids)):
                if mask_rng.random()<0.15:
                    prob=mask_rng.random()
                    if prob<0.8:
                        token_ids[i]=self.MASK_ID
                    elif prob<0.9:
                        token_ids[i]=mask_rng.randint(3,self.unified_vocab_size-1)

        while len(token_ids)<max_seq_len:
            token_ids.append(self.PAD_ID)

        return np.array(token_ids,dtype=np.int64)

    def _random_word(self,token_ids,rng):
        result=list(token_ids)
        for i in range(len(result)):
            if rng.random()<0.15:
                prob=rng.random()
                if prob<0.8:
                    result[i]=self.MASK_ID
                elif prob<0.9:
                    result[i]=rng.randint(3,self.unified_vocab_size)
        return result

    def _transfer_pretrained_weights(self,pretrain_model):
        pretrain_state=pretrain_model.state_dict()
        finetune_state=self._finetune_model.state_dict()
        transfer_keys=[k for k in pretrain_state if k.startswith("bert.")]
        transfer_count=0
        for key in transfer_keys:
            if key in finetune_state and pretrain_state[key].shape==finetune_state[key].shape:
                finetune_state[key]=pretrain_state[key]
                transfer_count+=1
        self._finetune_model.load_state_dict(finetune_state)

    def _build_finetune_inputs(self,visit,max_seq_len):
        diagnoses=list(visit[0]) if len(visit)>0 else []
        procedures=list(visit[1]) if len(visit)>1 else []
        diag_seq=self._visit_to_seq(diagnoses,max_seq_len,offset=self.dx_offset)
        proc_seq=self._visit_to_seq(procedures,max_seq_len,offset=self.pro_offset)
        input_ids=np.stack([diag_seq,proc_seq],axis=0)
        input_tensor=torch.tensor(input_ids,dtype=torch.long).unsqueeze(0)
        diag_prompt=torch.zeros(1,dtype=torch.long)
        proc_prompt=torch.ones(1,dtype=torch.long)
        prompt_tensor=torch.stack([diag_prompt,proc_prompt],dim=0).unsqueeze(0)

        return input_tensor,prompt_tensor

    def forward(self,batch: Dict[str,Any]) -> Dict[str,Any]:
        visit=batch.get("visit")
        if visit is None:
            visits=batch.get("visits")
            if isinstance(visits,list) and len(visits)>0:
                visit=visits[0]
        if visit is None:
            raise ValueError("TEMPT requires 'visit' in batch.")

        if isinstance(visit,(list,tuple)) and len(visit)>0:
            first_elem=visit[0]
            if isinstance(first_elem,(list,tuple)) and len(first_elem)>0:
                if isinstance(first_elem[0],(list,tuple)):
                    visit=visit[-1]

        max_seq_len=self.pretrain_max_seq_len
        input_tensor,prompt_tensor=self._build_finetune_inputs(visit,max_seq_len)
        input_tensor=input_tensor.to(self.device)
        prompt_tensor=prompt_tensor.to(self.device)
        medications=list(visit[2]) if len(visit)>2 else []
        med_label=np.zeros(self.med_voc_size,dtype=np.float32)
        for m in medications:
            if 0<=m<self.med_voc_size:
                med_label[m]=1.0
        rx_labels_tensor=torch.tensor(med_label,dtype=torch.float).unsqueeze(0).to(self.device)
        loss,rx_logits=self._finetune_model(input_ids=input_tensor,rx_labels=rx_labels_tensor,prompt=prompt_tensor)
        return {
            "loss":loss,
            "logits":rx_logits,
            "y_true":rx_labels_tensor
        }

    def compute_loss(self,outputs: Dict[str,Any],batch: Dict[str,Any]) -> torch.Tensor:
        logits=outputs["logits"]
        target=self.build_target(batch)
        if target.shape[-1]!=logits.shape[-1]:
            target=target[:,:logits.shape[-1]]
        return F.binary_cross_entropy_with_logits(logits,target)

    def predict(self,outputs: Dict[str,Any]) -> torch.Tensor:
        logits=outputs["logits"]
        return (torch.sigmoid(logits)>0.5).long()

    def predict_set(self,outputs: Dict[str,Any]) -> List[int]:
        logits=outputs["logits"]
        probs=torch.sigmoid(logits).detach().cpu().numpy()[0]
        return np.where(probs>0.5)[0].tolist()

    def parameters(self,recurse=True):
        return self._finetune_model.parameters(recurse=recurse)

    def train(self,mode=True):
        super().train(mode)
        if self._finetune_model is not None:
            self._finetune_model.train(mode)
        return self

    def eval(self):
        super().eval()
        if self._finetune_model is not None:
            self._finetune_model.eval()
        return self

    def to(self,*args,**kwargs):
        super().to(*args,**kwargs)
        if self._finetune_model is not None:
            self._finetune_model.to(*args,**kwargs)
        return self

from typing import Mapping, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from collie.models.base import CollieModelForCausalLM
from collie.models import LlamaForCausalLM
from collie.config import CollieConfig
import math
import random
from typing import Any
from collie.utils import env
from .utils import gradient_hook
from functools import partial
from .pruner import PerceiverPruner, PerceiverPrunerLayer

class AutoFuser:
    @staticmethod
    def from_pretrained(fuser_type, pretrained_model_name_or_path, config, perceiver_path):
        if fuser_type == 'sparse_fuser':
            pruner = SparseFuserPerceiver.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        else:
            raise NotImplementedError
        return pruner

class SparseFuserLayer(PerceiverPrunerLayer):
    def __init__(self, query_len, eval_query_len, d_query, d_model, d_head, chunk_size, temperature) -> None:
        super().__init__(query_len, eval_query_len, d_query, d_model, chunk_size)
        self.k_fuse = nn.Parameter(torch.eye(d_head))
        self.v_fuse = nn.Parameter(torch.eye(d_head))
        self.k_attn_layer_norm = nn.LayerNorm(d_head)
        self.v_attn_layer_norm = nn.LayerNorm(d_head)
        self.k_ffn_layer_norm = nn.LayerNorm(d_head)
        self.v_ffn_layer_norm = nn.LayerNorm(d_head)
        self.ffn = nn.Sequential(
            nn.Linear(d_head, d_query),
            nn.ReLU(),
            nn.Linear(d_query, d_head),
        )
        self.fuse_k = 4
        self.temperature=temperature

    def get_attention_scores(self, key, value):
        bsz, seq_len, num_heads, head_dim = key.shape
        query_len = self.query_len if self.training else self.eval_query_len
        query = self.query[:query_len].unsqueeze(dim=0).expand(bsz, query_len, -1)
        projected_query = self.Wq(query) # (bsz, query_len, d_query)
        projected_key = self.Wk(key.view(bsz, seq_len, -1)) # (bsz, seq_len, d_query)
        
        q_heads = projected_query.reshape(bsz, query_len, num_heads, -1)
        k_heads = projected_key.reshape(bsz, seq_len, num_heads, -1)
        
        logits = torch.einsum('bqhd,bkhd->bhqk', q_heads, k_heads)
        
        attention_scores = torch.softmax(logits, dim=-1)
        return attention_scores

    def fuse(self, anchor_indices, keys, values):
        # anchor_keys = torch.gather(keys, dim=1, index=anchor_indices) # (bsz, target_len, num_heads, head_dim)
        # anchor_values = torch.gather(values, dim=1, index=anchor_indices)

        # 找和锚点最近的k对kv，融合这k对kv
        bsz, seq_len, num_heads, head_dim = keys.shape
        num_anchor = anchor_indices.shape[1]
        fuse_indices = anchor_indices.view(bsz, -1, 1, num_heads, head_dim) + \
            torch.arange(self.fuse_k, device=anchor_indices.device).view(1,1,self.fuse_k,1,1) - (self.fuse_k - 1) # (bsz, target_len, fuse_k, num_heads, head_dim)
        fuse_indices = torch.clamp(fuse_indices, 0)
        # k_scores = torch.einsum('bahD,Dd,bshd->bash', anchor_keys, self.k_fuse, keys) # bilinear
        # fuse_scores, fuse_indices = torch.topk(k_scores, dim=2, k=self.fuse_k)
        # fuse_scores = torch.softmax(fuse_k_scores/self.temperature, dim=2)
        # fuse_indices = fuse_indices.view(bsz, -1, num_heads, 1).expand(bsz, num_anchor * self.fuse_k , num_heads, head_dim)

        fuse_indices = fuse_indices.reshape(bsz, -1, num_heads, head_dim)
        keys_to_fuse = torch.gather(keys, index=fuse_indices, dim=1).view(bsz, num_anchor, self.fuse_k, num_heads, head_dim) # bhafd
        values_to_fuse = torch.gather(values, index=fuse_indices, dim=1).view(bsz, num_anchor, self.fuse_k, num_heads, head_dim) # bhafd
        anchor_keys = keys_to_fuse[:, :, -1] # bsz, num_anchor, num_heads, head_dim
        fuse_scores = torch.einsum('bahD,Dd,bafhd->bafh', anchor_keys, self.k_fuse, keys_to_fuse) # bilinear
        fuse_scores = torch.softmax(fuse_scores/self.temperature, dim=2)

        fused_keys = torch.einsum('bafhd,bafh->bahd', keys_to_fuse, fuse_scores)
        fused_values = torch.einsum('bafhd,bafh->bahd', values_to_fuse, fuse_scores)
        return fused_keys, fused_values

    def forward(self, key, value, attention, target_len):
        # ignore the target len, the output len can only be query len anyway
        ################## estimate attention scores ###################
        # key: (bsz, seq, num_heads, head_dim)
        bsz, seq_len, num_heads, head_dim = key.shape
        attention_scores = self.get_attention_scores(key, value)

        ################## gather keys and values ###################
        topk_indices, topk_probs = self.get_indices_pos_routing(attention_scores, target_len)
        
        topk_indices = topk_indices.unsqueeze(dim=-1).expand(bsz, num_heads, target_len, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
        topk_indices = topk_indices.transpose(1, 2) # (bsz, num_heads, target_len, head_dim) -> (bsz, target_len, num_heads, head_dim)
        if topk_probs is not None:
            topk_probs = topk_probs.unsqueeze(dim=-1).expand(bsz, num_heads, target_len, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
            topk_probs = topk_probs.transpose(1, 2) # (bsz, num_heads, target_len, head_dim) -> (bsz, target_len, num_heads, head_dim)

        fused_keys, fused_values = self.fuse(topk_indices, key, value)
        fused_keys = self.k_attn_layer_norm(fused_keys)
        fused_values = self.v_attn_layer_norm(fused_values)
        
        fused_keys = self.k_ffn_layer_norm(self.ffn(fused_keys) + fused_keys) # add & norm
        fused_values = self.v_ffn_layer_norm(self.ffn(fused_values) + fused_values)
        
        fused_keys = fused_keys * (1 + topk_probs - topk_probs.detach())
        return fused_keys, fused_values * (1 + topk_probs - topk_probs.detach())

class SparseFuserPerceiver(PerceiverPruner):
    def __init__(self, config, chunk_size, query_len, eval_query_len, model=None, num_sink_tokens=0, 
                 num_layers=0, memory_type=None, d_query=0, d_model=0, num_heads=0, temperature=1.0, **kwargs):
        self.d_head = d_model // num_heads
        super().__init__(config, chunk_size, query_len, eval_query_len, model, num_sink_tokens, num_layers, memory_type, d_query, d_model, temperature, **kwargs)

    def build_perceiver_layer(self, query_len, eval_query_len, d_query, d_model, chunk_size, temperature):
        return SparseFuserLayer(query_len, eval_query_len, d_query, d_model, self.d_head, chunk_size, temperature)

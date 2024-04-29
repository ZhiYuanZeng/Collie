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
from .pruner import SparseParallelPerceiver, SparseParallelLayer

class AutoFuser:
    @staticmethod
    def from_pretrained(fuser_type, pretrained_model_name_or_path, config, perceiver_path):
        if fuser_type == 'sparse_fuser':
            pruner = SparseFuserPerceiver.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        else:
            raise NotImplementedError
        return pruner

class SparseFuserLayer(SparseParallelLayer):
    def __init__(self, query_len, eval_query_len, d_query, d_model, d_head) -> None:
        super().__init__(query_len, eval_query_len, d_query, d_model)
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

    def forward(self, key, value, target_len):
        # ignore the target len, the output len can only be query len anyway
        ################## estimate attention scores ###################
        # key: (bsz, seq, num_heads, head_dim)
        bsz, seq_len, num_heads, head_dim = key.shape
        attention_scores = self.get_attention_scores(key, value)

        ################## gather keys and values ###################
        target_len = self.query_len * self.fuse_k
        topk_indices, topk_probs = self.get_retrieve_indices(attention_scores, target_len)
        assert topk_indices.shape[-1] == target_len
        topk_indices = topk_indices.unsqueeze(dim=-1).expand(bsz, num_heads, target_len, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
        topk_indices = topk_indices.transpose(1, 2) # (bsz, num_heads, target_len, head_dim) -> (bsz, target_len, num_heads, head_dim)
        if topk_probs is not None:
            topk_probs = topk_probs.unsqueeze(dim=-1).expand(bsz, num_heads, target_len, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
            topk_probs = topk_probs.transpose(1, 2) # (bsz, num_heads, target_len, head_dim) -> (bsz, target_len, num_heads, head_dim)

        selected_keys = torch.gather(key, dim=1, index=topk_indices)
        selected_values = torch.gather(value, dim=1, index=topk_indices)

        selected_keys = selected_keys.view(bsz, self.query_len, self.fuse_k, num_heads, head_dim)
        selected_values = selected_values.view(bsz, self.query_len, self.fuse_k, num_heads, head_dim)
        topk_probs = topk_probs.reshape(bsz, self.query_len, self.fuse_k, num_heads, head_dim)
        fuse_keys = torch.sum(selected_keys * topk_probs, dim=2) / torch.sum(topk_probs, dim=2) # soft combine
        fuse_values = torch.sum(selected_values * topk_probs, dim=2) / torch.sum(topk_probs, dim=2).detach()
        
        fuse_keys = self.k_attn_layer_norm(fuse_keys)
        fuse_values = self.v_attn_layer_norm(fuse_values)
        
        fuse_keys = self.k_ffn_layer_norm(self.ffn(fuse_keys) + fuse_keys) # add & norm
        fuse_values = self.v_ffn_layer_norm(self.ffn(fuse_values) + fuse_values)
        
        return fuse_keys, fuse_values

class SparseFuserPerceiver(SparseParallelPerceiver):
    def __init__(self, config, chunk_size, query_len, eval_query_len, d_query, d_model, num_layers, num_sink_tokens, num_heads, model=None, **kwargs):
        self.d_head = d_model // num_heads
        super().__init__(config, chunk_size, query_len, eval_query_len, d_query, d_model, num_layers, num_sink_tokens, model, **kwargs)

    def build_perceiver_layer(self, query_len, eval_query_len, d_query, d_model):
        return SparseFuserLayer(query_len, eval_query_len, d_query, d_model, self.d_head)
    
import gc
import json
import math
import os
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from deepspeed.accelerator import get_accelerator
from deepspeed.pipe import LayerSpec, TiedLayerSpec
from einops import rearrange
from megatron.core import parallel_state, tensor_parallel
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import dtype_byte_size
from collie.config import CollieConfig
from collie.driver.io import IODriver
from collie.log.logger import logger
from collie.models.base import CollieModelForCausalLM
from collie.models.utils import (
    flash_attention, 
    kv_cache_to_inputs_for_layer, inputs_to_kv_cache_for_layer, 
    kv_cache_to_inputs_for_model, inputs_to_kv_cache_for_model, 
)
from collie.module import (
    ColumnParallelLinearWithoutBias,
    ColumnParallelLMHead,
    RowParallelLinearWithoutBias,
)
from collie.utils import concat_tensor, dict_as_params, env, progress

from collie.models.llama import LlamaForCausalLM, LlamaLayer
from typing import Any

# TODO:
# support recurrent compression
# support inference with long generation (compress on-the-fly)

class Pass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x

class PerceiverLayer(nn.Module):
    def __init__(self, d_query, d_model, num_heads, d_ffn) -> None:
        super().__init__()
        self.d_model = d_model
        # TODO: MultiheadAttention的output projection只有一个，但是输出的kv cache和perceiver的dimension不一样，kv cache: d_model, perceiver: d_query，也就是需要两个projection
        # 当前的实现是先做d_query x d_query，然后d_query x d_model
        self.k_cross_attention = nn.MultiheadAttention(d_query, num_heads, kdim=d_model, vdim=d_model, batch_first=True)
        self.v_cross_attention = nn.MultiheadAttention(d_query, num_heads, kdim=d_model, vdim=d_model, batch_first=True)
        # we share the ffn of keys and values, but use different attention and layer norm parameters for them
        self.k_attn_layer_norm = nn.LayerNorm(d_query)
        self.v_attn_layer_norm = nn.LayerNorm(d_query)
        self.k_ffn_layer_norm = nn.LayerNorm(d_query)
        self.v_ffn_layer_norm = nn.LayerNorm(d_query)
    
        self.ffn = nn.Sequential(
            nn.Linear(d_query, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_query),
        )

    def _forward(self, query, kv_cache, key_forward=True):
        if key_forward:
            cross_attention_func = self.k_cross_attention
            attn_layer_norm = self.k_attn_layer_norm
            ffn_layer_norm = self.k_ffn_layer_norm
        else:
            cross_attention_func = self.v_cross_attention
            attn_layer_norm = self.v_attn_layer_norm
            ffn_layer_norm = self.v_ffn_layer_norm

        res = query
        query = attn_layer_norm(query)
        # query: batch_size, seq_len, d_query  
        # kv_cache: batch_size, num_heads, seq_len, head_dim
        bsz, seq_len, num_heads, head_dim = kv_cache.shape
        assert num_heads * head_dim == self.d_model, f'{num_heads=}, {head_dim=}, {self.d_model=}'
        kv_cache = kv_cache.view(bsz, seq_len, -1)
        
        attention_outputs, _ = cross_attention_func(
            query=query,
            key=kv_cache,
            value=kv_cache,
        )
        query = attention_outputs + res
        
        res = query
        query = ffn_layer_norm(query)
        ffn_outputs = self.ffn(query)
        query = ffn_outputs + res
        return query

    def forward(self, compressed_k, compressed_v, keys, values):
        # compress keys to compressed_k, compress values to compressed_v
        compressed_k = self._forward(compressed_k, keys, self.k_cross_attention)
        compressed_v = self._forward(compressed_v, values, self.v_cross_attention)
        return compressed_k, compressed_v

class MemPerceiver(nn.Module):
    def __init__(self, model: torch.nn.Module, num_layers, query_len, d_query, d_model, d_ffn, num_heads, chunk_size):
        # cross-attention+ffn
        # soft query tokens at each layer
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.chunk_size = chunk_size
        self.query_len = query_len

        self.k_query_embed = nn.Parameter(torch.zeros(query_len, d_query))
        self.v_query_embed = nn.Parameter(torch.zeros(query_len, d_query))
        self.perceiver_layers = nn.ModuleList(
            [PerceiverLayer(d_query=d_query, d_model=d_model, d_ffn=d_ffn, num_heads=num_heads) for i in range(num_layers)])
        self.output_key_projections = nn.ModuleList([nn.Linear(d_query, d_model) for _ in range(num_layers)])
        self.output_value_projections = nn.ModuleList([nn.Linear(d_query, d_model) for _ in range(num_layers)])

    def compress(self, keys, values):
        # output compressed kv_cachee
        bsz, seq_len, num_heads, head_dim = keys[0].shape
        compressed_k = self.k_query_embed
        compressed_v = self.v_query_embed
        compressed_k = torch.expand_copy(compressed_k.unsqueeze(dim=0), [bsz, *compressed_k.shape]) # (seq_len, d) -> (bsz, seq_len, d)
        compressed_v = torch.expand_copy(compressed_v.unsqueeze(dim=0), [bsz, *compressed_v.shape]) # (seq_len, d) -> (bsz, seq_len, d)

        compressed_k_layers, compressed_v_layers = [], []
        for i, (layer, k, v) in enumerate(zip(self.perceiver_layers, keys, values)):
            assert isinstance(k, torch.Tensor)
            assert isinstance(v, torch.Tensor)
            # print(f'k shape: {k.shape}, v shape: {v.shape}', flush=True)
            compressed_k, compressed_v = layer(compressed_k, compressed_v, k, v)

            output_compressed_k = self.output_key_projections[i](compressed_k)
            output_compressed_v = self.output_key_projections[i](compressed_v)

            compressed_k_layers.append(output_compressed_k.reshape(bsz, self.query_len, num_heads, head_dim))
            compressed_v_layers.append(output_compressed_v.reshape(bsz, self.query_len, num_heads, head_dim))
            # print(f'after compression | k shape: {compressed_k_layers[-1].shape}, v shape: {compressed_k_layers[-1].shape}', flush=True)

        return compressed_k_layers, compressed_v_layers
    
    def forward(self, input_ids: Any | None = None, attention_mask: Any | None = None, past_key_values: Tuple | None = None, **kwargs):
        """
        split input_ids into chunks
        loop over chunks
            prepare inputs for llm forward
            if the first chunk
              mem is none
            else
              input the cached_mem to the forward of llm (like transformer-xl)
            call llm forward and get kv cache
            call compress to compress kv cache and append it to cached_mem

        if seq_len == 1: (incremental decoding)
            do not compress
        else
            do compress
        """
        chunked_input_ids = torch.split(input_ids, self.chunk_size, dim=1) # TODO: 支持长度无法被均分的情况
        num_chunks = len(chunked_input_ids)
        chunked_attention_mask = [attention_mask[i*self.chunk_size:(i+1)*self.chunk_size, i*self.chunk_size:(i+1)*self.chunk_size] 
                                  for i in range(num_chunks)]
        
        cached_llm_outpus = []
        seq_len = input_ids.shape[1]
        
        if seq_len > 1:
            # compress the kv cache of prompt
            assert past_key_values is None
            cached_compressed_kv = None
            for i in range(num_chunks):
                # is_grad_enabled = torch.is_grad_enabled()
                # if i == 0:
                #     torch.set_grad_enabled(False)
                model_outputs = self.model(chunked_input_ids[i], chunked_attention_mask[i], past_key_values=cached_compressed_kv)

                # torch.set_grad_enabled(is_grad_enabled)

                kv_cache = model_outputs.past_key_values # kv_cache is list of tuple: [(key of layer0, value of layer0), ...]
                keys, values = [kv[0] for kv in kv_cache], [kv[1] for kv in kv_cache]
                compressed_keys, compressed_values = self.compress(keys, values)
                cached_llm_outpus.append(model_outputs)
                # if i!=0:
                #     cached_llm_outpus.append(model_outputs) # drop the first rank, since the first chunk does not use compressed memory
                if cached_compressed_kv is None:
                    cached_compressed_kv = [torch.stack([ck,cv], dim=0) for ck, cv in zip(compressed_keys, compressed_values)] # ([2, 1, seq_len, num_heads, head_dim], [...], ...)
                else:
                    cached_compressed_kv = [
                        torch.cat([cached_kv, torch.stack([ck,cv], dim=0)], dim=2)
                        for cached_kv, ck, cv in zip(cached_compressed_kv, compressed_keys, compressed_values)]

            accum_logits = torch.cat([o.logits for o in cached_llm_outpus], dim=1) # batch_size, seq_len
            # print(f"hidden states shape: {cached_llm_outpus[0][0].shape}, {cached_llm_outpus[0][1].shape}, {cached_llm_outpus[0][2].shape}")
            # print(f"{accum_logits.shape=}")
            # accum_hidden_states = torch.cat([o.hidden_states for o in cached_llm_outpus], dim=1) # batch_size, seq_len, d_model
            lm_output = CausalLMOutputWithPast(
                loss=None,
                logits=accum_logits,
                past_key_values=cached_compressed_kv,
                hidden_states=None,
                attentions=None,
            )
            return lm_output
        
        else:
            # incremental decoding
            assert past_key_values is not None
            return self.model(chunked_input_ids, chunked_attention_mask, past_key_values=past_key_values)
        
        

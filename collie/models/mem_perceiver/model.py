from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from collie.models.base import CollieModelForCausalLM
from .utils import Scale, PerceiverAttention, gradient_hook

from typing import Any

# TODO:
# implement from_pretrained, load_state_dict, save_state_dict
# 支持不定长的inputs (inference)
# 实现kv cache压缩
# support recurrent compression
# support inference with long generation (compress on-the-fly)

class PerceiverLayer(nn.Module):
    def __init__(self, d_query, d_model, num_heads, d_ffn, collie_config=None) -> None:
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
        self.collie_config = collie_config

    def __forward(self, query, kv_cache, key_forward=True):
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

    def _forward(self, compressed_k, compressed_v, keys, values):
        # compress keys to compressed_k, compress values to compressed_v
        compressed_k = self.__forward(compressed_k, keys, key_forward=True)
        compressed_v = self.__forward(compressed_v, values, key_forward=False)
        return compressed_k, compressed_v

    def forward(self, compressed_k, compressed_v, keys, values):
        if self.collie_config is not None and self.collie_config.checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                compressed_k,
                compressed_v,
                keys,
                values
            )
        else:
            return self._forward(compressed_k, compressed_v, keys, values)

class MemPerceiver(CollieModelForCausalLM):
    def __init__(self, config, num_layers, query_len, d_query, d_model, d_ffn, num_heads, chunk_size, model=None):
        # cross-attention+ffn
        # soft query tokens at each layer
        super().__init__(config)
        self.model = model
        if self.model is not None:
            self.frozen_model()
        self.chunk_size = chunk_size
        self.query_len = query_len

        self.k_query_embed = nn.Parameter(torch.zeros(query_len, d_query))
        self.v_query_embed = nn.Parameter(torch.zeros(query_len, d_query))
        self.perceiver_layers = nn.ModuleList(
            [PerceiverLayer(d_query=d_query, d_model=d_model, d_ffn=d_ffn, num_heads=num_heads, collie_config=config) for _ in range(num_layers)])
        self.output_key_projections = nn.ModuleList([nn.Linear(d_query, d_model) for _ in range(num_layers)])
        self.output_value_projections = nn.ModuleList([nn.Linear(d_query, d_model) for _ in range(num_layers)])

    def frozen_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

    @classmethod
    def from_config(cls, config, model):
        kwargs = config.mem_perceiver_config
        mem_perceiver = super().from_config(config, **kwargs) # from config会对模型做随机初始化，所以初始化完后再传入pretrained model
        mem_perceiver.model = model
        mem_perceiver.frozen_model()
        return mem_perceiver

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
            if self.training:
                assert compressed_k.requires_grad and compressed_v.requires_grad
            # print(f'k shape: {k.shape}, v shape: {v.shape}', flush=True)

            compressed_k, compressed_v = layer(compressed_k, compressed_v, k, v)
            # assert compressed_k.requires_grad and compressed_v.requires_grad

            output_compressed_k = self.output_key_projections[i](compressed_k)
            output_compressed_v = self.output_value_projections[i](compressed_v)

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
        seq_len = input_ids.shape[1]
        
        if seq_len > self.chunk_size:
            chunked_input_ids = torch.split(input_ids, self.chunk_size, dim=1) # TODO: 支持长度无法被均分的情况
            chunked_attention_mask = torch.split(attention_mask, self.chunk_size, dim=1)
            num_chunks = len(chunked_input_ids)
            
            cached_llm_outpus = []

            # print("compress prompt...", flush=True)
            # compress the kv cache of prompt
            assert past_key_values is None
            cached_compressed_kv = None
            for i in range(num_chunks):
                # is_grad_enabled = torch.is_grad_enabled()
                # if i == 0:
                #     torch.set_grad_enabled(False)
                # print(f'input shape: {chunked_input_ids[i].shape}, {chunked_attention_mask[i].shape}, {attention_mask.shape}')
                model_outputs = self.model(chunked_input_ids[i], chunked_attention_mask[i], past_key_values=cached_compressed_kv)
 
                # torch.set_grad_enabled(is_grad_enabled)

                kv_cache = model_outputs.past_key_values # kv_cache is list of tuple: [(key of layer0, value of layer0), ...]
                keys, values = [kv[0].detach() for kv in kv_cache], [kv[1].detach() for kv in kv_cache]
                compressed_keys, compressed_values = self.compress(keys, values)
                
                # assert compressed_keys.requires_grad and compressed_values.requires_grad
                cached_llm_outpus.append(model_outputs)
                # if i!=0:
                #     cached_llm_outpus.append(model_outputs) # drop the first rank, since the first chunk does not use compressed memory
                new_compressed_kv = torch.stack([torch.stack([ck,cv], dim=0) for ck, cv in zip(compressed_keys, compressed_values)], dim=0) # [num_layers, 2, bsz, seq_len, num_heads, head_dim]
                if cached_compressed_kv is None:
                    cached_compressed_kv = new_compressed_kv
                    # cached_compressed_kv.register_hook(gradient_hook)
                else:
                    cached_compressed_kv = torch.cat([cached_compressed_kv, new_compressed_kv], dim=3)
            accum_logits = torch.cat([o.logits for o in cached_llm_outpus], dim=1) # batch_size, seq_len
            # TODO: accumulate hidden states
            # print(f"hidden states shape: {cached_llm_outpus[0][0].shape}, {cached_llm_outpus[0][1].shape}, {cached_llm_outpus[0][2].shape}")
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
            # print("incremental decoding...", flush=True)
            # incremental decoding
            assert past_key_values is not None
            # print(f'{input_ids.shape=}, {attention_mask.shape=}, {past_key_values[0].shape=}')
            # the compressed kv cache should all be attened, so the attention_mask should be None
            return self.model(input_ids, None, past_key_values=past_key_values)
    
    def set_cache(self, use_cache):
        return self.model.set_cache(use_cache)
    
    def clean_cache(self):
        return self.model.clean_cache()

# TODO: 去掉multihead attention中的value projection，控制temperature
class ParallelPerceiverLayer(nn.Module):
    # 每一层并行地压缩kv cache，不同层的压缩没有关联
    # key和value共享attention weights，所以直接把他们拼接在一起作为cross-attention的values
    def __init__(self, query_len, d_query, d_model, num_heads, temperature=1.0):
        super().__init__()
        self.query_embedding = nn.Parameter(torch.randn(query_len, d_query))
        # PerceiverAttention is different from multiheadattention that it does not need value projection and output projection
        self.cross_attention = PerceiverAttention(
            d_query, num_heads, kdim=d_model, vdim=2*d_model, batch_first=True)
        self.num_heads = num_heads
        self.key_scale = Scale(d_model) # bsz, seq, d_model
        self.value_scale = Scale(d_model)
        self.temperature = temperature
        self.d_model = d_model
        self.query_len = query_len
        assert self.temperature == 0.1
    
    def forward(self, keys, values):
        bsz, seq_len, num_heads, head_dim = keys.shape
        cross_attention_query = torch.expand_copy(self.query_embedding.unsqueeze(dim=0), [bsz, *self.query_embedding.shape]) / self.temperature
        
        cross_attention_key = keys.view(bsz, seq_len, self.d_model)
        # we want keys and values to share attention weights
        cross_attention_value = torch.cat(
            [keys, values], dim=-1) # (bsz seq_len, num_heads, 2*head_dim)
        cross_attention_value = cross_attention_value.view(bsz, seq_len, 2*self.d_model)
        attention_outputs = self.cross_attention(cross_attention_query, cross_attention_key, cross_attention_value)
        attention_outputs = attention_outputs[0].view(bsz, self.query_len, num_heads, 2, head_dim)
        compressed_keys, compresser_values = attention_outputs[:, :, :, 0, :], attention_outputs[:, :, :, 1, :]
        compressed_keys = compressed_keys.reshape(bsz, self.query_len, self.d_model)
        compresser_values = compresser_values.reshape(bsz, self.query_len, self.d_model)
        

        return self.key_scale(compressed_keys), self.value_scale(compresser_values)

class ParallelMemPerceiver(MemPerceiver):
    def __init__(self, config, num_layers, query_len, d_query, d_model, num_heads, chunk_size, temperature=1.0, model=None):
        # cross-attention+ffn
        # soft query tokens at each layer
        CollieModelForCausalLM.__init__(self, config)

        self.model = model
        self.chunk_size = chunk_size
        self.query_len = query_len

        self.perceiver_layers = nn.ModuleList(
            [ParallelPerceiverLayer(query_len, d_query, d_model, num_heads, temperature) for _ in range(num_layers)]
        )
    
    def compress(self, keys, values):
        # output compressed kv_cachee
        bsz, seq_len, num_heads, head_dim = keys[0].shape

        compressed_k_layers, compressed_v_layers = [], []
        for i, (layer, k, v) in enumerate(zip(self.perceiver_layers, keys, values)):
            assert isinstance(k, torch.Tensor)
            assert isinstance(v, torch.Tensor)
            # print(f'k shape: {k.shape}, v shape: {v.shape}', flush=True)

            compressed_k, compressed_v = layer(k, v)
            # assert compressed_k.requires_grad and compressed_v.requires_grad
            compressed_k_layers.append(compressed_k.reshape(bsz, self.query_len, num_heads, head_dim))
            compressed_v_layers.append(compressed_v.reshape(bsz, self.query_len, num_heads, head_dim))
            # print(f'after compression | k shape: {compressed_k_layers[-1].shape}, v shape: {compressed_k_layers[-1].shape}', flush=True)

        return compressed_k_layers, compressed_v_layers
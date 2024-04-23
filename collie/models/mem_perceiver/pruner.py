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

class H2oPruner(CollieModelForCausalLM):
    def __init__(self, config, chunk_size, query_len, model=None, num_sink_tokens=0, **kwargs):
        super().__init__(config)
        self.model = model
        self.chunk_size = chunk_size
        self.query_len = query_len
        if self.model is not None:
            self.frozen_model()
        self.num_sink_tokens = num_sink_tokens
        # TODO: accumulate important scores AND normalization scores
        # self.accumulate_scores = None
        # self.accumulate_norm = None
    
    def frozen_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

    @classmethod
    def from_config(cls, config, model):
        # assert not config.use_flash
        kwargs = config.mem_perceiver_config
        mem_perceiver = super().from_config(config, **kwargs) # from config会对模型做随机初始化，所以初始化完后再传入pretrained model
        mem_perceiver.model = model
        mem_perceiver.frozen_model()
        return mem_perceiver

    def get_indices(self, attention, seq_len,  target_len):
        # attention: [bsz, num_heads, key_len, key_len]
        # key: [bsz, seq_len, num_heads, head_dim]
        important_scores = attention.mean(dim=2) # [bsz, num_heads, seq_len, key_len] -> [bsz, num_heads, key_len]
        normalized_scores = important_scores / (important_scores.shape[-1] - torch.arange(important_scores.shape[-1]).type_as(important_scores))
        topk_indices = torch.topk(normalized_scores, k=target_len, largest=True, dim=-1).indices
        # topk_indices: (bsz, num_heads, k)
        return topk_indices

    def compress(self, keys, values, attentions, target_len):
        keeped_keys = []
        keeped_values = []
        bsz, seq_len, num_heads, head_dim = keys[0].shape
        for key, value, attention in zip(keys, values, attentions):
            if attention is None:
                attention_shape = (bsz, num_heads, seq_len, seq_len)
            topk_indices = self.get_indices(attention, attention_shape, target_len).to(key.device)
            topk_indices = topk_indices.unsqueeze(dim=-1).expand(bsz, num_heads, target_len, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
            topk_indices = topk_indices.transpose(1, 2) # (bsz, num_heads, target_len, head_dim) -> (bsz, target_len, num_heads, head_dim)

            selected_keys = torch.gather(key, index=topk_indices, dim=1)
            selected_values = torch.gather(value, index=topk_indices, dim=1)

            assert selected_keys.shape == (bsz, target_len, num_heads, head_dim)
            keeped_keys.append(selected_keys)
            keeped_values.append(selected_values)
        # print(f'{keys[0].shape=}, {keeped_keys[0].shape}', flush=True)
        return keeped_keys, keeped_values

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
                model_outputs = self.model(chunked_input_ids[i], None, past_key_values=cached_compressed_kv)
 
                kv_cache = model_outputs.past_key_values # kv_cache is list of tuple: [(key of layer0, value of layer0), ...]
                attention_scores = model_outputs.attentions

                keys, values = [kv[0].detach() for kv in kv_cache], [kv[1].detach() for kv in kv_cache]
                if self.query_len > 0:
                    compressed_keys, compressed_values = self.compress(keys, values, attention_scores, target_len=(i+1)*self.query_len)
                
                    # assert compressed_keys.requires_grad and compressed_values.requires_grad
                    cached_compressed_kv = torch.stack([torch.stack([ck,cv], dim=0) for ck, cv in zip(compressed_keys, compressed_values)], dim=0) # [num_layers, 2, bsz, seq_len, num_heads, head_dim]
                else: # local window
                    cached_compressed_kv = None
                # we need to detach output to free memory
                model_outputs =CausalLMOutputWithPast(
                    logits=model_outputs.logits,
                )

                cached_llm_outpus.append(model_outputs) # drop the first rank, since the first chunk does not use compressed memory
                
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

    @staticmethod
    def save_parallel_state_dict(state_dict: dict,
        path: str,
        config,
        process_exclusion: bool = False,
        protocol: str = "file",
    ):
        if env.rank == 0:
            config.save_pretrained(path, protocol=protocol)
            new_state_dict = {}
            for k,v in state_dict.items():
                if 'model.' not in k:
                    new_state_dict[k] = v
            torch.save(new_state_dict, path + '/pytorch_model.bin')

    @staticmethod
    def load_parallel_state_dict(
        path: str,
        **kwargs,
    ):
        state_dict = torch.load(path + '/pytorch_model.bin', map_location='cpu')
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict=False, assign=assign)   

    @classmethod
    def from_pretrained(cls, model_path_or_name: str, config: CollieConfig, perceiver_path=None, **kwargs):
        model = LlamaForCausalLM.from_pretrained(model_path_or_name, config=config)
        mem_perceiver = cls.from_config(config=config, model=model)
        if perceiver_path is not None:
            state_dict = cls.load_parallel_state_dict(perceiver_path)
            mem_perceiver.load_state_dict(state_dict)
        return mem_perceiver

class StreamingLMPruner(H2oPruner):
    def get_indices(self, attention, attention_shape, target_len):
        # indices shape: (bsz, num_heads, target_len)
        bsz, num_heads = attention_shape[0], attention_shape[1]
        indices = torch.arange(self.num_sink_tokens, device=attention.device)
        return indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, self.num_sink_tokens)
    
    def compress(self, keys, values, attentions, target_len):
        keeped_keys = []
        keeped_values = []
        bsz, seq_len, num_heads, head_dim = keys[0].shape
        for key, value, attention in zip(keys, values, attentions):
            attention_shape = (bsz, num_heads, seq_len, seq_len)
            topk_indices = self.get_indices(attention_shape, target_len).to(key.device)
            topk_indices = topk_indices.unsqueeze(dim=-1).expand(bsz, num_heads, self.num_sink_tokens, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
            topk_indices = topk_indices.transpose(1, 2) # (bsz, num_heads, target_len, head_dim) -> (bsz, target_len, num_heads, head_dim)

            selected_keys = torch.gather(key, index=topk_indices, dim=1)
            selected_values = torch.gather(value, index=topk_indices, dim=1)

            assert selected_keys.shape == (bsz, self.num_sink_tokens, num_heads, head_dim)
            keeped_keys.append(selected_keys)
            keeped_values.append(selected_values)
        return keeped_keys, keeped_values

class RandomPruner(H2oPruner):
    # sink tokens + random tokens
    def get_indices(self, attention, attention_shape, target_len, segment_size=64):
        bsz, num_heads, seq_len = attention_shape[0], attention_shape[1], attention_shape[-1]
        assert seq_len % segment_size == 0
        assert target_len % segment_size == 0
        target_segment_num = target_len // segment_size
        segment_num = seq_len // segment_size
        assert seq_len >= target_len
        segment_indices = random.sample(range(segment_num), target_segment_num)
        token_indices = torch.arange(seq_len).view(segment_num, segment_size)[segment_indices]
        token_indices = token_indices.view(-1)
        token_indices[:self.num_sink_tokens] = torch.arange(self.num_sink_tokens)
        print(token_indices, flush=True)
        # assert len(token_indices) == target_len
        return token_indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, target_len)

class ChunkPrefix(H2oPruner):
    # keep the first k tokens for each chunk
    def get_indices(self, attention, attention_shape, target_len):
        # indices shape: (bsz, num_heads, target_len)
        bsz, num_heads = attention_shape[0], attention_shape[1]
        indices = torch.arange(target_len)
        return indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, target_len)

class SparseParallelLayer(nn.Module):
    def __init__(self, query_len, eval_query_len, d_query, d_model, num_sink_tokens) -> None:
        super().__init__()
        self.query_len = query_len
        self.eval_query_len = eval_query_len
        assert self.query_len >= self.eval_query_len
        self.query = nn.Parameter(torch.randn(query_len, d_query))
        self.Wq = nn.Linear(d_query, d_query, bias=False)
        self.Wk = nn.Linear(d_model, d_query, bias=False)
        self.d_query = d_query
        self.num_sink_tokens = num_sink_tokens
    
    def forward(self, key, value, target_len):
        # key: (bsz, seq, num_heads, head_dim)
        bsz, seq_len, num_heads, head_dim = key.shape
        query_len = self.query_len if self.training else self.eval_query_len
        query = self.query[:query_len].unsqueeze(dim=0).expand(bsz, query_len, -1)
        projected_query = self.Wq(query) # (bsz, query_len, d_query)
        projected_key = self.Wk(key.view(bsz, seq_len, -1)) # (bsz, seq_len, d_query)
        
        q_heads = projected_query.reshape(bsz, query_len, num_heads, -1)
        k_heads = projected_key.reshape(bsz, seq_len, num_heads, -1)
        
        logits = torch.einsum('bqhd,bkhd->bhqk', q_heads, k_heads)
        
        attention_scores = torch.softmax(logits/math.sqrt(self.d_query), dim=-1)
        assert target_len % query_len == 0
        num_kv_per_query = target_len // query_len
        topk_values, topk_indices = torch.topk(attention_scores, dim=-1, k=num_kv_per_query) # (bsz, num_heads, query_len, seq_len) -> (bsz, num_heads, query_len, k)
        
        topk_indices, topk_values = topk_indices.view(bsz, num_heads, target_len), topk_values.view(bsz, num_heads, target_len)
        
        # find the kv with the lowest scores, and replace them with sink tokens
        if self.num_sink_tokens > 0:
            sink_token_indices = topk_values.topk(k=self.num_sink_tokens, dim=-1).indices # (bsz, num_heads, num_sink)
            topk_indices = torch.scatter(topk_indices, index=sink_token_indices, dim=-1, 
                                        src=torch.arange(self.num_sink_tokens).type_as(topk_indices).view(1,1,self.num_sink_tokens).expand_as(sink_token_indices))
            topk_values = torch.scatter(topk_values, index=sink_token_indices, dim=-1, 
                                        src=torch.ones([1,1,self.num_sink_tokens]).type_as(topk_values).expand_as(sink_token_indices))
        topk_indices, topk_values = topk_indices.unsqueeze(dim=-1), topk_values.unsqueeze(dim=-1)
        assert torch.all(topk_indices < seq_len)

        # if not self.training:
        #     print(topk_indices, flush=True)

        topk_indices = topk_indices.transpose(1, 2).expand(bsz, target_len, num_heads, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, query_len, num_heads, head_dim)
        topk_values = topk_values.transpose(1, 2).expand(bsz, target_len, num_heads, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, query_len, num_heads, head_dim)

        selected_keys = torch.gather(key, index=topk_indices, dim=1)
        selected_values = torch.gather(value, index=topk_indices, dim=1)

        assert selected_keys.shape == (bsz, target_len, num_heads, head_dim)
        assert selected_keys.shape == topk_values.shape
        selected_keys = selected_keys * (1 + topk_values - topk_values.detach()) # straight-through trick
        selected_values = selected_values * (1 + topk_values - topk_values.detach())
        return selected_keys, selected_values

class SparseParallelPerceiver(H2oPruner):
    def __init__(self, config, chunk_size, query_len, eval_query_len, d_query, d_model, num_layers, num_sink_tokens, model=None, **kwargs):
        super().__init__(config, chunk_size, query_len, model, **kwargs)
        self.perceiver_layers = nn.ModuleList([
            SparseParallelLayer(query_len, eval_query_len, d_query, d_model, num_sink_tokens)
            for i in range(num_layers)
        ])
        self.num_layers = num_layers

    def compress(self, keys, values, attentions, target_len):
        keeped_keys = []
        keeped_values = []
        assert len(keys) == self.num_layers
        for i, (key, value, attention) in enumerate(zip(keys, values, attentions)):
            selected_keys, selected_values = self.perceiver_layers[i](key[:, -self.chunk_size:], value[:, -self.chunk_size:], self.query_len)
            keeped_keys.append(torch.cat([key[:, :-self.chunk_size], selected_keys], dim=1))
            keeped_values.append(torch.cat([value[:, :-self.chunk_size], selected_values], dim=1))
        # print(f'key shape before compression: {keys[0].shape}, after compress: {keeped_keys[0].shape}')
        return keeped_keys, keeped_values
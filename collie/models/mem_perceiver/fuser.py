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
from .pruner import PerceiverPruner, PerceiverPrunerLayer, TovaPruner, build_llm_from_name_or_path, MemoryType
from peft import LoraConfig, TaskType
from peft import get_peft_model
from collie.utils.peft_utils import load_peft


class FuserType:
    PERCEIVER='perceiver'
    LLM='llm'

class AutoFuser:
    @staticmethod
    def from_pretrained(fuser_type, pretrained_model_name_or_path, config, perceiver_path):
        if fuser_type == 'perceiver':
            fuser = SparseFuserPerceiver.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif fuser_type == 'llm':
            fuser = LLMFuser.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        else:
            raise NotImplementedError
        return fuser

class SparseFuserLayer(PerceiverPrunerLayer):
    def __init__(self, query_len, compressed_chunk_size, d_query, d_model, d_head, chunk_size, temperature) -> None:
        super().__init__(query_len, compressed_chunk_size, d_query, d_model, chunk_size, temperature)
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
        query_len = self.query_len if self.training else self.compressed_chunk_size
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
        target_len = topk_indices.shape[-1]
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
    def __init__(self, config, chunk_size, query_len, compressed_chunk_size, model=None, num_sink_tokens=0, 
                 num_layers=0, memory_type=None, d_query=0, d_model=0, num_heads=0, temperature=1.0, **kwargs):
        self.d_head = d_model // num_heads
        super().__init__(config, chunk_size, query_len, compressed_chunk_size, model, num_sink_tokens, num_layers, memory_type, d_query, d_model, temperature, **kwargs)

    def build_perceiver_layer(self, query_len, compressed_chunk_size, d_query, d_model, chunk_size, temperature):
        return SparseFuserLayer(query_len, compressed_chunk_size, d_query, d_model, self.d_head, chunk_size, temperature)

class LLMFuser(TovaPruner):
    def __init__(self, config, chunk_size, compressed_chunk_size=0, model=None, num_sink_tokens=0, num_layers=0, memory_type=None, query_len=0, **kwargs):
        super().__init__(config, chunk_size, compressed_chunk_size, model, num_sink_tokens, num_layers, memory_type)
        self.query_len = query_len

    @classmethod
    def from_pretrained(cls, model_path_or_name: str, config: CollieConfig, perceiver_path=None, **kwargs):
        # TODO: get lora config from args
        model = build_llm_from_name_or_path(model_path_or_name, config)
        mem_perceiver = cls.from_config(config=config, model=model)
        peft_config = LoraConfig(
            base_model_name_or_path=model_path_or_name,
            r=128,
            lora_alpha=256,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
            ],
            lora_dropout=0.3,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["embed_tokens"]
        )
        mem_perceiver.agument_embedding()
        mem_perceiver = get_peft_model(mem_perceiver, peft_config)

        mem_perceiver.print_trainable_parameters()
        print('Trainable parameters:...............')
        if env.rank == 0:
            for n,p in mem_perceiver.named_parameters():
                if p.requires_grad:
                    print(n)
        if perceiver_path is not None:
            load_peft(mem_perceiver, config, perceiver_path)
        return mem_perceiver
            
    def frozen_model(self):
        # peft would frozen the model automatically
        pass

    def get_lm_head(self):
        # do not allow resize_token_embeddings to resize lm_head
        return None, None

    def agument_embedding(self):
        vocab_size = self.model.config.vocab_size
        self.memory_token_offset = vocab_size
        if env.rank == 0:
            print('before agumenting:')
            print(self.model)
            setattr(self.model, 'get_lm_head', self.get_lm_head)
        self.model.resize_token_embeddings(vocab_size + self.query_len)
        if env.rank == 0:
            print('after agumenting:')
            print(self.model)
        # self.memory_token_embed = nn.Embedding(self.chunk_size, embedding_dim=self.config.hidden_size)

    def read_and_write_memory(self, compress_func, **kwargs_of_compress_func):
        chunk_step = kwargs_of_compress_func.pop('chunk_step')
        # write short-term memory to long-term memory, read from memory
        keys = kwargs_of_compress_func['keys']
        values = kwargs_of_compress_func['values']
        sink_keys = [key[:, :self.num_sink_tokens] for key in keys]
        sink_values = [value[:, :self.num_sink_tokens] for value in values]
        keys = [key[:, self.num_sink_tokens:] for key in keys]
        values = [value[:, self.num_sink_tokens:] for value in values]
        kwargs_of_compress_func['keys'] = keys
        kwargs_of_compress_func['values'] = values
        
        if self.memory_type == MemoryType.CHUNK_STREAMING:
            # 类似rmt读写compressed kv cache,但是每次写入会覆盖之前保存的compressed kv cache
            kwargs_of_compress_func['target_len'] = self.compressed_chunk_size # the compressed memory size is constant
            compressed_key, compressed_value = compress_func(**kwargs_of_compress_func)
            return [(torch.cat([sink_keys[i], compressed_key[i]], dim=1), torch.cat([sink_values[i], compressed_value[i]], dim=1)) for i in range(self.num_layers)]
        
        elif self.memory_type == MemoryType.FIXED_INCREMENTAL:
            # 类似AutoCompresser, 压缩后的kv cache都缓存下来, 且都被读取
            incremental_keys = [key[:, :-self.chunk_size] for key in keys]
            incremental_values = [value[:, :-self.chunk_size] for value in values]
            kwargs_of_compress_func['target_len'] = self.compressed_chunk_size # the compressed key size is constant
            compressed_key, compressed_value = compress_func(**kwargs_of_compress_func)
            return [(torch.cat([sink_keys[i], incremental_keys[i], compressed_key[i]], dim=1), torch.cat([sink_values[i], incremental_values[i], compressed_value[i]], dim=1)) for i in range(self.num_layers)]
        
        elif self.memory_type == MemoryType.DYNAMIC_INCREMENTAL:
            # memory在随着chunk数量的增加而变长，但是每次增长会刷新整个memory，而incremental memory只会在之前的基础上拼接新的memory
            kwargs_of_compress_func['target_len'] = self.compressed_chunk_size * (chunk_step + 1) # incremental memory size
            compressed_key, compressed_value = compress_func(**kwargs_of_compress_func)
            return [(torch.cat([sink_keys[i], compressed_key[i]], dim=1), torch.cat([sink_values[i], compressed_value[i]], dim=1)) for i in range(self.num_layers)]
        elif self.memory_type == MemoryType.DYNAMIC_INCREMENTAL_DOUBLE_COMPRESS:
            keys = [key[:, -self.chunk_size:] for key in keys]
            values = [value[:, -self.chunk_size:] for value in values]

            cached_keys = self.cached_keys
            cached_values = self.cached_values

            if cached_keys[0] is not None:
                kwargs_of_compress_func['keys'] = [torch.cat([ck, k], dim=1) for ck,k in zip(cached_keys, keys)]
                kwargs_of_compress_func['values'] = [torch.cat([cv, v], dim=1) for cv,v in zip(cached_values, values)]

            kwargs_of_compress_func['target_len'] = self.compressed_chunk_size * (chunk_step + 1) # incremental memory size
            kwargs_of_compress_func['double_compress'] = True
            compressed_keys, compressed_values, double_compressed_keys, double_compressed_value = compress_func(**kwargs_of_compress_func)

            self.cached_keys = compressed_keys
            self.cached_values = compressed_values

            return [(torch.cat([sink_keys[i], double_compressed_keys[i]], dim=1), torch.cat([sink_values[i], double_compressed_value[i]], dim=1)) for i in range(self.num_layers)]
        else:
            raise NotImplementedError

    def construct_sequence(self, original_sequence, n):
        k = len(original_sequence)
        if k == 0:
            return []
        
        # Calculate the base repetition count and the remainder
        base_count = n // k
        remainder = n % k
        
        # Construct the new sequence
        new_sequence = []
        for i in range(k):
            repetitions = base_count + (1 if i < remainder else 0)
            new_sequence.extend([original_sequence[i]] * repetitions)
        
        return new_sequence

    def llm_forward(self, keys, values, target_len):
        bsz = keys[0].shape[0]
        if target_len > self.query_len:
            mem_token_ids = self.repeat_elements(range(self.query_len), target_len) # repeat mem token id to construct seq of length of target len
        else:
            mem_token_ids = range(target_len)
        # mem_token_ids = [0 for _ in range(target_len)]
        input_ids = torch.tensor(mem_token_ids, device=keys[0].device) + self.memory_token_offset
        input_ids = input_ids.view(1, -1).expand(bsz, target_len)
        past_key_values = torch.stack([torch.stack([ck,cv], dim=0) for ck, cv in zip(keys, values)], dim=0)
        llm_outputs = self.model(input_ids, None, past_key_values=past_key_values)
        new_keys = [kv[0] for kv in llm_outputs.past_key_values]
        new_values = [kv[1] for kv in llm_outputs.past_key_values]
        return new_keys, new_values

    def split_sequence(self, n, k):
        if k <= 0 or n <= 0:
            return []

        # 计算完整的长度为k的子序列的数量
        q = n // k
        # 计算剩余的长度
        r = n % k

        # 形成长度为k的子序列
        subsequences = [k] * q

        # 如果有剩余的长度，则加入长度为r的子序列
        if r > 0:
            subsequences.append(r)

        return subsequences

    def repeat_elements(self, original_list, n):
        k = len(original_list)
        if k == 0:
            return []

        # 构建新的长度为n的列表
        new_list = [original_list[i % k] for i in range(n)]
        return new_list

    def _compress(self, keys, values, target_len, double_compress=False, **kwargs):
        # if self.query_len !=0 and target_len > self.query_len:
        #     pass
            # sub_target_lens = self.split_sequence(target_len, self.query_len)
            # cached_keys, cached_values = [], []
            # for sub_len in sub_target_lens:
            #     # print(f'before compression: {keys[0].shape}')
            #     keys, values = self.llm_forward(keys, values, target_len=sub_len)
            #     # print(f'after compression: {keys[0].shape}')
            #     cached_keys.append(torch.stack([k[:, -sub_len:] for k in keys], dim=0))
            #     cached_values.append(torch.stack([v[:, -sub_len:] for v in values], dim=0))
            # cached_keys = torch.cat(cached_keys, dim=2)
            # cached_values = torch.cat(cached_values, dim=2)
            # assert cached_keys.shape[2] == target_len
            # assert cached_values.shape[2] == target_len
            # return cached_keys, cached_values
        # else:
        new_keys, new_values = self.llm_forward(keys, values, target_len)
        new_keys = [k[:, -target_len:] for k in new_keys]
        new_values = [v[:, -target_len:] for v in new_values]
        if double_compress:
            # FIXME: 实现真正的double compression
            double_compressed_keys = [k[:, -self.compressed_chunk_size:] for k in new_keys]
            double_compressed_values = [v[:, -self.compressed_chunk_size:] for v in new_values]
            return new_keys, new_values, double_compressed_keys, double_compressed_values
        else:
            return new_keys, new_values

    def compress(self, keys, values, attentions, chunk_step):
        new_key_values = self.read_and_write_memory(self._compress, keys=keys, values=values, attentions=attentions, chunk_step=chunk_step)
        new_keys, new_values = [kv[0] for kv in new_key_values], [kv[1].detach() for kv in new_key_values]
        return new_keys, new_values
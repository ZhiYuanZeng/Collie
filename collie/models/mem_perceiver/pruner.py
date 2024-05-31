from typing import Mapping, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from collie.models.base import CollieModelForCausalLM
from collie.models import LlamaForCausalLM, InternLM2ForCausalLM
from collie.config import CollieConfig
import math
import random
from typing import Any
from collie.utils import env
from .utils import gradient_hook
from functools import partial
from enum import Enum, unique
from collie.module import GPTLMLoss

class MemoryType:
    CHUNK_STREAMING="Chunk_Streaming"
    FIXED_INCREMENTAL="Incremental_Chunk_Streaming_Fixed_History"
    DYNAMIC_INCREMENTAL="Incremental_Chunk_Streaming_Dynamic_History"
    DYNAMIC_INCREMENTAL_DOUBLE_COMPRESS="dynamic_incremental_double_compress"
    RETRIEVE_INCREMENTAL="Incremental_Chunk_Streaming_Retrieved_History"
    RETRIEVE_ALL_KV="Cache_All_KV_Retrieve_History"

class PrunerType:
    H2O="h2o"
    STREAMING="streaming_llm"
    CHUNK_PREFIX="chunk_prefix"
    TOVA="tova"
    RANDOM="random"
    LOCAL_WINDOW="local_window"
    NO_COMPRESS="no_compress"
    PERCEIVER="perceiver"
    ROCO="roco"
    CONV="conv"

def build_llm_from_name_or_path(model_name_or_path, config):
    if 'llama' in model_name_or_path.lower():
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
    elif 'internlm' in model_name_or_path.lower() or 'moss' in model_name_or_path.lower():
        model = InternLM2ForCausalLM.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
    else:
        raise NotImplementedError
    return model

class AutoPruner:
    @staticmethod
    def from_pretrained(pruner_type, pretrained_model_name_or_path, config, perceiver_path=None):
        if pruner_type == PrunerType.STREAMING:
            pruner = StreamingLMPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.CHUNK_PREFIX:
            pruner = ChunkPrefix.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.TOVA:
            config.use_flash = False
            print('Warning: the h2o pruner requires attention scores, therefore the flash_attention is set to False!')
            pruner = TovaPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.RANDOM:
            pruner = RandomPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.LOCAL_WINDOW: # remove context
            config.mem_perceiver_config['compressed_chunk_size'] = 0
            pruner = TovaPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.NO_COMPRESS:
            pruner = build_llm_from_name_or_path(pretrained_model_name_or_path, config) # no compress
        elif pruner_type == PrunerType.ROCO:
            config.use_flash = False
            pruner = ROCOPruner.from_pretrained(pretrained_model_name_or_path, config)
        elif pruner_type == PrunerType.CONV:
            config.use_flash = False
            pruner = ConvPruner.from_pretrained(pretrained_model_name_or_path, config)
        # parameters required
        elif pruner_type == PrunerType.PERCEIVER:
            pruner = PerceiverPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        else:
            raise NotImplementedError
        if pruner_type in (PrunerType.H2O):
            assert config.mem_perceiver_config['memory_type'] in (MemoryType.CHUNK_STREAMING, MemoryType.DYNAMIC_INCREMENTAL)

        return pruner

class TovaPruner(CollieModelForCausalLM):
    def __init__(self, config, chunk_size, compressed_chunk_size=0, model=None, num_sink_tokens=0, num_layers=0, memory_type=None, separate_compress=False, memory_size_limit=None, **kwargs):
        super().__init__(config)
        self.model = model
        self.chunk_size = chunk_size
        self.compressed_chunk_size = compressed_chunk_size # the query len at inference may be different from that of training
        if self.model is not None:
            self.frozen_model()
        assert num_layers > 0
        self.num_sink_tokens = num_sink_tokens
        self.num_layers = num_layers
        self.memory_type = memory_type
        self.cached_keys = [None for _ in range(num_layers)]
        self.cached_values = [None for _ in range(num_layers)]
        self.cached_attentions = [None for _ in range(num_layers)]
        self.memory_size_limit = memory_size_limit

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


    def get_indices(self, attention, attention_shape, target_len):
        assert attention is not None
        chunk_size = min(self.chunk_size, attention.shape[-1])
        new_attention_scores = attention[:, :, -self.chunk_size:, :] # (bsz, num_heads, chunk_size, seq_len)
        accum_attention_scores = new_attention_scores.sum(dim = 2) # (bsz, num_heads, seq_len)

        normalization_scores = torch.ones(attention.shape[-1], device=attention.device) * chunk_size
        normalization_scores[-chunk_size:] = chunk_size - torch.arange(chunk_size, device=attention.device)
        # print(normalization_scores)
        important_scores = accum_attention_scores / normalization_scores.view(1, 1, -1).expand_as(accum_attention_scores)

        important_indices = torch.topk(important_scores, dim=-1, k=target_len).indices
        important_indices, _ = torch.sort(important_indices, dim=-1) # 方便位置编码
        # print(self.move_avg('memory_ratio',
        #     torch.sum(important_indices < self.compressed_chunk_size).item() / torch.numel(important_indices)
        # ))
        
        return important_indices

    def update_cached_indices(self, seq_len, topk_indices, chunk_step, layer_idx):
        bsz, num_heads, _ = topk_indices.shape
        if self.cached_indices[layer_idx] is not None:
            if self.memory_type == MemoryType.FIXED_INCREMENTAL:
                assert seq_len <= self.chunk_size
                new_chunk_size = seq_len
                global_indices = self.chunk_size * chunk_step + \
                    torch.arange(new_chunk_size, device=topk_indices.device).view(1,1,new_chunk_size).expand(bsz, num_heads, new_chunk_size)
            else:
                new_chunk_size = seq_len - self.cached_indices[layer_idx].shape[-1]
                new_chunk_indices = self.chunk_size * chunk_step + \
                    torch.arange(new_chunk_size, device=topk_indices.device).view(1,1,new_chunk_size).expand(bsz, num_heads, new_chunk_size)
                global_indices = torch.cat([self.cached_indices[layer_idx], new_chunk_indices], dim=-1)
                # if layer_idx == 0:
                #     print(self.cached_indices[layer_idx][0,0][:(seq_len-new_chunk_size)])

        else:
            new_chunk_indices = self.num_sink_tokens + torch.arange(seq_len, device=topk_indices.device).view(1,1,seq_len).expand(bsz, num_heads, seq_len)
            global_indices = new_chunk_indices
        assert global_indices.shape[-1] == seq_len

        if self.memory_type == MemoryType.FIXED_INCREMENTAL:
            if self.cached_indices[layer_idx] is None:
                self.cached_indices[layer_idx] = torch.gather(global_indices, dim=-1, index=topk_indices)
            else:
                self.cached_indices[layer_idx] = torch.cat(
                    [self.cached_indices[layer_idx], torch.gather(global_indices, dim=-1, index=topk_indices)], dim=-1
                )
        else:
            self.cached_indices[layer_idx] = torch.gather(global_indices, dim=-1, index=topk_indices)

    def update_cached_freqs(self, seq_len, topk_indices, chunk_step, layer_idx):
        if self.memory_type == MemoryType.FIXED_INCREMENTAL:
            return
        bsz, num_heads, target_len = topk_indices.shape
        if self.cached_frequences[layer_idx] is None:
            self.cached_frequences[layer_idx] = torch.ones(target_len, device=topk_indices.device).view(1,1,-1).expand(bsz, num_heads, target_len)
        else:
            self.cached_frequences[layer_idx] = self.cached_frequences[layer_idx] + 1
            new_len = seq_len-self.cached_frequences[layer_idx].shape[-1]
            new_frequences = torch.ones(new_len, device=topk_indices.device).view(1,1,-1).expand(bsz, num_heads, new_len)
            new_frequences = torch.cat([self.cached_frequences[layer_idx], new_frequences], dim=-1)
            self.cached_frequences[layer_idx] = torch.gather(new_frequences, index=topk_indices, dim=-1)
            # if layer_idx == 0:
            #     print(self.cached_frequences[layer_idx][0,0][:(seq_len-new_len)])

    def update_avg_memory_rate(self, k, v):
        if self.avg_memory_rate[k] is None:
            self.avg_memory_rate[k] = {'avg': 0, 'count': 0}

        # 更新计数和总和
        self.avg_memory_rate[k]['avg'] = (self.avg_memory_rate[k]['avg'] * self.avg_memory_rate[k]['count'] + v) / (self.avg_memory_rate[k]['count'] + 1)
        self.avg_memory_rate[k]['count'] += 1

    def update_keep_memory_rate(self, topk_indices, layer_idx):
        if self.memory_type == MemoryType.FIXED_INCREMENTAL:
            return
        if self.cached_indices[layer_idx] is None:
            return
        else:
            mem_size = self.cached_indices[layer_idx].shape[-1]
            self.keep_memory_rate[layer_idx] = torch.sum(topk_indices < mem_size) / torch.numel(topk_indices)
        self.update_avg_memory_rate(layer_idx, self.keep_memory_rate[layer_idx])

    def report_keep_memory_rate(self):
        if env.rank == 0:
            sum_rate = 0
            for i,r in enumerate(self.avg_memory_rate):
                print(f'keep memory rate of layer {i}: {r["avg"]}')
                sum_rate += r['avg']
            print(f'overall keep memory rate: {sum_rate / len(self.avg_memory_rate)}')

        bsz, seq_len, num_heads, head_dim = key.shape
        if attention is None:
            attention_shape = (bsz, num_heads, seq_len, seq_len)
        else:
            attention_shape = attention.shape
        if attention_shape[1] != num_heads:
            # support gqa
            assert attention_shape[1] % num_heads == 0
            num_groups = attention_shape[1] // num_heads
            attention = attention.view(bsz, num_groups, num_heads, attention.shape[-2], attention.shape[-1])
            attention = attention.sum(dim=1)
            assert attention.shape[1] == num_heads
        
        topk_indices = self.get_indices(attention, attention_shape, target_len)
        topk_indices = topk_indices.to(key.device)
        # self.record_indices(topk_indices.view(-1) + chunk_step * self.chunk_size)
        topk_indices = topk_indices.unsqueeze(dim=-1).expand(bsz, num_heads, target_len, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
        topk_indices = topk_indices.transpose(1, 2) # (bsz, num_heads, target_len, head_dim) -> (bsz, target_len, num_heads, head_dim)

        selected_keys = torch.gather(key, index=topk_indices, dim=1)
        selected_values = torch.gather(value, index=topk_indices, dim=1)
        assert selected_keys.shape == (bsz, target_len, num_heads, head_dim)

        return selected_keys, selected_values

    def compress(self, keys, values, attentions, chunk_step):
        keeped_keys = []
        keeped_values = []
        for i, (key, value, attention) in enumerate(zip(keys, values, attentions)):
            selected_keys, selected_values = self.read_and_write_memory(
                self.compress_layer,
                key=key, value=value, attention=attention, chunk_step=chunk_step)
            keeped_keys.append(selected_keys)
            keeped_values.append(selected_values)
        # print(f'before compression, key shape: {keys[0].shape}, value shape: {values[0].shape}, attention shape: {attentions[0].shape}')
        # print(f'after compression, key shape: {keeped_keys[0].shape}, value shape: {keeped_values[0].shape}')

        return keeped_keys, keeped_values
    
    def read_and_write_memory(self, compress_func, **kwargs_of_compress_func):
        chunk_step = kwargs_of_compress_func.pop('chunk_step')
        layer_idx = kwargs_of_compress_func['layer_idx']
        # write short-term memory to long-term memory, read from memory
        key = kwargs_of_compress_func['key']
        value = kwargs_of_compress_func['value']
        sink_key = key[:, :self.num_sink_tokens]
        sink_value = value[:, :self.num_sink_tokens]
        key = key[:, self.num_sink_tokens:]
        value = value[:, self.num_sink_tokens:]
        kwargs_of_compress_func['key'] = key
        kwargs_of_compress_func['value'] = value
        if kwargs_of_compress_func['attention'] is not None:
            kwargs_of_compress_func['attention'] = kwargs_of_compress_func['attention'][:,:,:,self.num_sink_tokens:]
        
        if self.memory_type == MemoryType.CHUNK_STREAMING:
            # 类似rmt读写compressed kv cache,但是每次写入会覆盖之前保存的compressed kv cache
            kwargs_of_compress_func['target_len'] = self.compressed_chunk_size # the compressed memory size is constant
            compressed_key, compressed_value = compress_func(**kwargs_of_compress_func)
            return torch.cat([sink_key, compressed_key], dim=1), torch.cat([sink_value, compressed_value], dim=1)
        
        elif self.memory_type == MemoryType.FIXED_INCREMENTAL:
            # 类似AutoCompresser, 压缩后的kv cache都缓存下来, 且都被读取
            incremental_key = key[:, :-self.chunk_size]
            incremental_value = value[:, :-self.chunk_size]
            if self.memory_size_limit is not None and incremental_key.shape[1] >= self.memory_size_limit:
                return torch.cat([sink_key, incremental_key]), torch.cat([sink_value, incremental_value])
            
            kwargs_of_compress_func['key'] = key[:, -self.chunk_size:]
            kwargs_of_compress_func['value'] = value[:, -self.chunk_size:]
            if kwargs_of_compress_func['attention'] is not None:
                kwargs_of_compress_func['attention'] = kwargs_of_compress_func['attention'][:, :, :, -self.chunk_size:]
            
            kwargs_of_compress_func['target_len'] = self.compressed_chunk_size # the compressed key size is constant
            compressed_key, compressed_value = compress_func(**kwargs_of_compress_func)
            return torch.cat([sink_key, incremental_key, compressed_key], dim=1), torch.cat([sink_value, incremental_value, compressed_value], dim=1)
        
        elif self.memory_type == MemoryType.DYNAMIC_INCREMENTAL:
            # memory在随着chunk数量的增加而变长，但是每次增长会刷新整个memory，而incremental memory只会在之前的基础上拼接新的memory
            if self.memory_size_limit is not None:
                kwargs_of_compress_func['target_len'] = min(self.compressed_chunk_size * (chunk_step + 1), self.memory_size_limit)
            else:
                kwargs_of_compress_func['target_len'] = self.compressed_chunk_size * (chunk_step + 1) # incremental memory size

            compressed_key, compressed_value = compress_func(**kwargs_of_compress_func)
                        
            return torch.cat([sink_key, compressed_key], dim=1), torch.cat([sink_value, compressed_value], dim=1)
        else:
            raise NotImplementedError

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
        if seq_len > 1:
            chunked_input_ids = torch.split(input_ids, self.chunk_size, dim=1) # TODO: 支持长度无法被均分的情况
            # chunked_attention_mask = torch.split(attention_mask, self.chunk_size, dim=1)
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
                if self.compressed_chunk_size > 0:
                    # print(chunked_input_ids[i].shape, flush=True)
                    if chunked_input_ids[i].shape[1] > self.compressed_chunk_size:
                        compressed_keys, compressed_values = self.compress(keys=keys, values=values, attentions=attention_scores, chunk_step=i)
                    else:
                        compressed_keys, compressed_values = keys, values # we do not need to compress it, since it is already very small
                    # assert compressed_keys.requires_grad and compressed_values.requires_grad
                    cached_compressed_kv = torch.stack([torch.stack([ck,cv], dim=0) for ck, cv in zip(compressed_keys, compressed_values)], dim=0) # [num_layers, 2, bsz, seq_len, num_heads, head_dim]
                else: # local windows
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
            self.clear_perceiver_cache()
            return lm_output
        
        else:
            # print("incremental decoding...", flush=True)
            # incremental decoding
            assert past_key_values is not None
            use_flash = self.collie_config.use_flash
            for layer in self.model.model.layers:
                layer.config.use_flash = True # force to use flash attention at decoding

            # print(f'{input_ids.shape=}, {attention_mask.shape=}, {past_key_values[0].shape=}')
            # the compressed kv cache should all be attened, so the attention_mask should be None
            model_outputs = self.model(input_ids, None, past_key_values=past_key_values)
            for layer in self.model.model.layers:
                layer.config.use_flash = use_flash # force to use flash attention at decoding
            return model_outputs
        
    def set_cache(self, use_cache):
        return self.model.set_cache(use_cache)
    
    def clean_cache(self):
        return self.model.clean_cache()

    def clear_perceiver_cache(self):
        self.cached_keys = [None for _ in range(self.num_layers)]
        self.cached_values = [None for _ in range(self.num_layers)]
        self.cached_attentions = [None for _ in range(self.num_layers)]

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
        model = build_llm_from_name_or_path(model_path_or_name, config)
        if perceiver_path is not None:
            try:
                load_config = CollieConfig.from_pretrained(perceiver_path, trust_remote_code=True)
                config.mem_perceiver_config['query_len'] = load_config.mem_perceiver_config['query_len']
            except Exception:
                pass
            mem_perceiver = cls.from_config(config=config, model=model)
            state_dict = cls.load_parallel_state_dict(perceiver_path)
            mem_perceiver.load_state_dict(state_dict)
        else:
            mem_perceiver = cls.from_config(config=config, model=model)
        return mem_perceiver

class StreamingLMPruner(TovaPruner):
    def get_indices(self, attention, attention_shape, target_len, *args, **kwargs):
        # indices shape: (bsz, num_heads, target_len)
        bsz, num_heads, seq_len = attention_shape[0], attention_shape[1], attention_shape[-1]
        indices = torch.arange(seq_len)[-target_len:]
        indices = indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, target_len)
        return indices

class RandomPruner(TovaPruner):
    # sink tokens + random tokens
    def get_indices(self, attention, attention_shape, target_len, *args, **kwargs):
        bsz, num_heads, seq_len = attention_shape[0], attention_shape[1], attention_shape[-1]
        assert seq_len >= target_len
        sample_indices = random.sample(range(seq_len), target_len)
        token_indices = torch.arange(seq_len)[sample_indices]
        token_indices = token_indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, target_len)
        return token_indices

class ChunkPrefix(TovaPruner):
    def get_indices(self, attention, attention_shape, target_len):
        bsz, num_heads = attention_shape[0], attention_shape[1]
        indices = torch.arange(target_len)
        return indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, target_len)

class ConvPruner(TovaPruner):
    def __init__(self, config, chunk_size, compressed_chunk_size=0, model=None, num_sink_tokens=0, num_layers=0, memory_type=None, kernel_size=5, **kwargs):
        super().__init__(config, chunk_size, compressed_chunk_size, model, num_sink_tokens, num_layers, memory_type, **kwargs)
        self.kernel_size = kernel_size
        assert kernel_size % 2 != 0

    def get_indices(self, attention, attention_shape, target_len):
        assert attention is not None
        chunk_size = min(self.chunk_size, attention.shape[-1])
        new_attention_scores = attention[:, :, -self.chunk_size:, :] # (bsz, num_heads, chunk_size, seq_len)
        accum_attention_scores = new_attention_scores.sum(dim = 2) # (bsz, num_heads, seq_len)
        
        accum_attention_scores = nn.functional.avg_pool1d(accum_attention_scores, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        assert accum_attention_scores.shape[-1] == attention_shape[-1], f'{accum_attention_scores.shape=}, {attention_shape=}'
        
        normalization_scores = torch.ones(attention.shape[-1], device=attention.device) * chunk_size
        normalization_scores[-chunk_size:] = chunk_size - torch.arange(chunk_size, device=attention.device)
        important_scores = accum_attention_scores / normalization_scores.view(1, 1, -1).expand_as(accum_attention_scores)

        important_indices = torch.topk(important_scores, dim=-1, k=target_len).indices
        important_indices, _ = torch.sort(important_indices, dim=-1) # 方便位置编码

        return important_indices

class ROCOPruner(TovaPruner):
    def get_indices(self, attention, attention_shape, target_len):
        assert attention is not None
        chunk_size = min(self.chunk_size, attention.shape[-1])
        new_attention_scores = attention[:, :, -self.chunk_size:, :] # (bsz, num_heads, chunk_size, seq_len)
        accum_attention_scores = new_attention_scores.sum(dim = 2) # (bsz, num_heads, seq_len)
        assert accum_attention_scores.shape[-1] == attention_shape[-1]
        
        normalization_scores = torch.ones(attention.shape[-1], device=attention.device) * chunk_size
        normalization_scores[-chunk_size:] = chunk_size - torch.arange(chunk_size, device=attention.device)
        # print(normalization_scores)
        important_scores = accum_attention_scores / normalization_scores.view(1, 1, -1).expand_as(accum_attention_scores)
        # v(x)=e(x^2)-e(x)^2
        std_scores = torch.sqrt(torch.sum(new_attention_scores**2) / normalization_scores - important_scores**2)
        _, std_removed_indices = torch.topk(std_scores, k=self.compressed_chunk_size // 2, dim=-1, largest=False)
        
        mask_for_important_scores = torch.zeros_like(important_scores).bool()
        assert not torch.any(mask_for_important_scores)
        important_scores = torch.scatter(
            important_scores, 
            dim=-1, 
            index=std_removed_indices, 
            src=torch.zeros_like(std_removed_indices).type_as(important_scores)) # mask the kv with low variance

        important_indices = torch.topk(important_scores, dim=-1, k=target_len).indices

        important_indices, _ = torch.sort(important_indices, dim=-1) # 方便位置编码

        return important_indices

class PerceiverPrunerLayer(nn.Module):
    def __init__(self, query_len, compressed_chunk_size, d_query, d_model, chunk_size, temperature) -> None:
        super().__init__()
        self.query_len = query_len
        self.compressed_chunk_size = compressed_chunk_size
        assert self.query_len >= self.compressed_chunk_size or self.compressed_chunk_size % self.query_len == 0
        self.query = nn.Parameter(torch.randn(query_len, d_query))
        self.Wq = nn.Linear(d_query, d_query, bias=False)
        self.Wk = nn.Linear(d_model, d_query, bias=False)
        self.chunk_size = chunk_size
        self.temperature = temperature
    
    def get_random_indices(self, attention, target_len, segment_size=1):
        bsz, num_heads, seq_len = attention.shape[0], attention.shape[1], attention.shape[-1]
        assert seq_len % segment_size == 0
        assert target_len % segment_size == 0
        target_segment_num = target_len // segment_size
        segment_num = seq_len // segment_size
        assert seq_len >= target_len
        segment_indices = random.sample(range(segment_num), target_segment_num)
        token_indices = torch.arange(seq_len, device=attention.device).view(segment_num, segment_size)[segment_indices]
        token_indices = token_indices.view(-1)
        # assert len(token_indices) == target_len
        return token_indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, target_len), None
        
    def get_rank_indices(self, attention, target_len,):
        important_scores = attention.mean(dim=2) # (bsz, num_heads, seq_len)
        topk_values, topk_indices = torch.topk(important_scores, dim=-1, k=target_len)
        return topk_indices, topk_values
    
    def get_indices_pos_routing(self, attention, target_len):
        # 为了排除不同query检索的内容重复，需要先对kv做分区，每个query对应单独的区域
        # attention: (bsz, num_heads, query_len, kv_len)
        # TODO: handel the case that target_len is not divided by query_len
        bsz, num_heads, query_len, kv_len = attention.shape
        if kv_len % query_len != 0:
            num_pad = query_len - kv_len % query_len
            # the negative parts are not likely to be selected, just to enable parallel computation
            attention = torch.nn.functional.pad(attention, pad=[0,num_pad, 0,0, 0,0, 0,0], value=-1)
            assert attention.shape == (bsz, num_heads, query_len, kv_len+num_pad)
        else:
            num_pad = 0
        assert target_len % query_len == 0
        num_tokens_per_query = (kv_len+num_pad) // query_len
        attention = attention.reshape(bsz, num_heads, query_len, query_len, num_tokens_per_query)
        attention = torch.gather(attention, dim=3, 
                                 index=torch.arange(query_len, device=attention.device).view(1,1,query_len,1,1).expand(bsz, num_heads, query_len, 1, num_tokens_per_query))
        # attention: (bsz, num_heads, query_len, 1, num_tokens_per_query)
        attention = attention.squeeze(dim=3)
        target_len_per_query = target_len // query_len
        topk_probs, topk_indices = torch.topk(attention, k=target_len_per_query, dim=-1) # (bsz, num_heads, query_len, target_len_per_query)
        topk_indices = topk_indices + torch.arange(query_len, device=topk_indices.device).view(1,1,query_len,1) * num_tokens_per_query # add offset for each query [0, num_tokens_per_query, 2*num_tokens_per_query, ...]
        topk_indices = topk_indices.view(bsz, num_heads, target_len)
        topk_probs = topk_probs.view(bsz, num_heads, target_len)
        if num_pad > 0:
            indices = torch.arange(target_len, device=topk_indices.device)
            non_overflow_mask = (topk_probs[0, 0] >= 0) # 只有当pad数量天多，才会选到pad，而且每个样本、每个head都会选到相同数量的pad
            non_overflow_indices = indices[non_overflow_mask]

            topk_indices = topk_indices[:, :, non_overflow_indices]
            topk_probs = topk_probs[:, :, non_overflow_indices]
        assert torch.all(topk_probs >= 0) and torch.all(topk_indices < kv_len)
        return topk_indices, topk_probs

    def get_indices_random_routing(self, attention, target_len):
        shuffle_indices = list(range(attention.shape[-1]))
        random.shuffle(shuffle_indices)
        shuffle_indices = torch.tensor(shuffle_indices, device=attention.device)
        attention = attention[:,:,:,shuffle_indices]
        topk_indices, topk_probs = self.get_indices_random_routing(attention, target_len)
        bsz, num_heads, _ = topk_indices.shape
        indices_len = len(shuffle_indices)
        topk_indices = torch.gather(shuffle_indices.view(1,1,indices_len).expand(bsz, num_heads, indices_len), 
                                    dim=-1, index=topk_indices)
        return topk_indices, topk_probs
    
    def get_indices_greedy_routing(self, attention, target_len):
        # take each query as an expert
        bsz, num_heads, query_len, kv_len = attention.shape
        token_expert_indices = torch.argmax(attention, dim=-2) # each token select one expert (query), shape: (bsz, num_heads, kv_len)
        mask = nn.functional.one_hot(token_expert_indices, num_classes=query_len).transpose(-1, -2) # (bsz, num_heads, query_len, kv_len)
        masked_attention = attention + mask.type_as(attention) # 优先考虑token选择专家结果，因为这些选择之间没有重合，这些选择都考虑完了，再考虑专家选择token的概率
        # num_tokens_each_query = masked_attention.sum(dim=-1)
        # num_target_tokens_each_query = (num_tokens_each_query / kv_len * target_len).int()
        
        token_per_query = target_len // self.query_len # capacity
        topk_probs, expert_token_indices = torch.topk(masked_attention, dim=-1, k=token_per_query)
        expert_token_indices = expert_token_indices.view(bsz, num_heads, -1)
        topk_probs = topk_probs.view(bsz, num_heads, -1)
                
        return expert_token_indices, topk_probs

    def get_attention_scores(self, key, value):
        bsz, seq_len, num_heads, head_dim = key.shape
        if self.compressed_chunk_size < self.query_len and not self.training:
            query_len = self.compressed_chunk_size
            query = self.query[:query_len]
        else:
            query_len = self.query_len
            query = self.query
        query = query.unsqueeze(dim=0).expand(bsz, query_len, -1)
        projected_query = self.Wq(query) # (bsz, query_len, d_query)
        projected_key = self.Wk(key.view(bsz, seq_len, -1)) # (bsz, seq_len, d_query)
        
        q_heads = projected_query.reshape(bsz, query_len, num_heads, -1)
        k_heads = projected_key.reshape(bsz, seq_len, num_heads, -1)
        
        logits = torch.einsum('bqhd,bkhd->bhqk', q_heads, k_heads)
        
        attention_scores = torch.softmax(logits/self.temperature, dim=-1)
        return attention_scores

    def forward(self, key, value, attention, target_len):
        ################## estimate attention scores ###################
        # key: (bsz, seq, num_heads, head_dim)
        bsz, seq_len, num_heads, head_dim = key.shape
        attention_scores = self.get_attention_scores(key, value)
        ################## gather keys and values ###################
        topk_indices, topk_probs = self.get_random_indices(attention_scores, target_len)
        target_len = topk_indices.shape[-1] # the actual target len may be smaller than the expected
        topk_indices, sort_indices = torch.sort(topk_indices, dim=-1)
        topk_indices = topk_indices.unsqueeze(dim=-1).expand(bsz, num_heads, target_len, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
        topk_indices = topk_indices.transpose(1, 2) # (bsz, num_heads, target_len, head_dim) -> (bsz, target_len, num_heads, head_dim)
        if topk_probs is not None:
            topk_probs = torch.gather(topk_probs, dim=-1, index=sort_indices)
            topk_probs = topk_probs.unsqueeze(dim=-1).expand(bsz, num_heads, target_len, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
            topk_probs = topk_probs.transpose(1, 2) # (bsz, num_heads, target_len, head_dim) -> (bsz, target_len, num_heads, head_dim)

        selected_keys = torch.gather(key, dim=1, index=topk_indices)
        selected_values = torch.gather(value, dim=1, index=topk_indices)
        if topk_probs is not None:
            selected_keys = selected_keys * (1 + topk_probs - topk_probs.detach()) # straight-through trick
        if attention is not None:
            target_scores = attention[:, :, -self.chunk_size:, :].mean(dim=2)
            model_scores = attention_scores.mean(dim=2)
            assert target_scores.shape == model_scores.shape
            mse_loss = torch.mean((attention.shape[-1]*(target_scores - model_scores))**2)
            self.mse_loss = mse_loss
        else:
            self.mse_loss = None
        return selected_keys, selected_values

class PerceiverPruner(TovaPruner):
    def __init__(self, config, chunk_size, query_len, compressed_chunk_size=0, model=None, num_sink_tokens=0, num_layers=0, memory_type=None, d_query=0, d_model=0, temperature=1.0, **kwargs):
        super().__init__(config, chunk_size, compressed_chunk_size, model, num_sink_tokens, num_layers, memory_type, **kwargs)
        self.perceiver_layers = nn.ModuleList([
            self.build_perceiver_layer(query_len, compressed_chunk_size, d_query, d_model, chunk_size, temperature)
            for i in range(num_layers)
        ])
        self.query_len = query_len
        self.num_layers = num_layers
        self.temperature = temperature

    def compress_layer(self, key, value, attention, target_len, layer_idx):
        # print(f'param of W_Q at 1st layer: {self.perceiver_layers[0].Wq.weight.data}')
        selected_keys, selected_values = self.perceiver_layers[layer_idx](key, value, attention, target_len)
        # print(f'key shape before compression: {keys[0].shape}, after compress: {keeped_keys[0].shape}')
        return selected_keys, selected_values
    
    def build_perceiver_layer(self, query_len, compressed_chunk_size, d_query, d_model, chunk_size, temperature):
        return PerceiverPrunerLayer(query_len, compressed_chunk_size, d_query, d_model, chunk_size, temperature)
    
    def forward(self, input_ids: Any | None = None, attention_mask: Any | None = None, past_key_values: Tuple | None = None, **kwargs):
        outputs = super().forward(input_ids, attention_mask, past_key_values, **kwargs)
        avg_mse_loss = 0
        for layer_idx in range(self.num_layers):
            mse_loss = getattr(self.perceiver_layers[layer_idx], 'mse_loss', None)
            if mse_loss is not None:
                avg_mse_loss += mse_loss
                # self.perceiver_layers[layer_idx].mse_loss = None
        avg_mse_loss /= self.num_layers
        if avg_mse_loss != 0:
            setattr(outputs, 'mse_loss', avg_mse_loss)
        return outputs

class PrunerLoss(GPTLMLoss):
    def __init__(self, ignore_index=-100, aux_loss_weight=0.):
        super().__init__(ignore_index)
        self.aux_loss_weight = aux_loss_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mse_loss: torch.Tensor):
        lm_loss = super().forward(logits, labels)
        mse_loss = mse_loss * self.aux_loss_weight
        return lm_loss + mse_loss
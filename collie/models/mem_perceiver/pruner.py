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
from .utils import ChunkSizeScheduler, MemSizeScheduler, IncrementalType, LayerwiseIncrementalType, MemoryStateManager, ReviewScheduler
from functools import partial
from enum import Enum, unique
from collie.module import GPTLMLoss
import torch.nn.functional as F

class MemoryType:
    CHUNK_STREAMING="FM"
    DualMemory="dual_memory"
    FIXED_INCREMENTAL="Incremental_Chunk_Streaming_Fixed_History"
    DYNAMIC_INCREMENTAL="IM"
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
    RANDOM_QUERY='random_query'
    ONLINE_KMEANS_QUERY='online_kms'
    all_pruners = [H2O, STREAMING, CHUNK_PREFIX, TOVA, RANDOM, LOCAL_WINDOW, NO_COMPRESS, PERCEIVER, ROCO, CONV, RANDOM_QUERY, ONLINE_KMEANS_QUERY]

# class IncrementalType:
#     LINEAR="linear"
#     SQUARE="square"
#     HALF_SQUARE_HALF_LINEAR="half_square_half_linear"
#     HALF_SQUARE_HALF_SQRT="half_square_half_sqrt"
#     ADAPTIVE="adaptive"
#     SMOOTH_ADAPTIVE="smooth_adaptive"
    

def build_llm_from_name_or_path(model_name_or_path, config):
    if 'llama' in model_name_or_path.lower():
        pe_config  = {'exp': False, '1d': False, 'imp': False, 'log': False, 'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'pi_lambda': 1,
                      'max_length': config.max_position_embeddings, 'base': config.model_config.rope_theta, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }
        print(f'model: {model_name_or_path}, base: {pe_config["base"]}')
        config.pe_config = pe_config
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
    elif 'internlm' in model_name_or_path.lower() or 'moss' in model_name_or_path.lower():
        model = InternLM2ForCausalLM.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
    else:
        raise NotImplementedError
    return model

class AutoMemory:
    @staticmethod
    def from_pretrained(compresser_type, pretrained_model_name_or_path, config, perceiver_path=None):
        if compresser_type in PrunerType.all_pruners:
            return AutoPruner.from_pretrained(compresser_type, pretrained_model_name_or_path, config, perceiver_path=None)
        elif compresser_type == 'share_kv':
            config.share_kv = True
            return build_llm_from_name_or_path(pretrained_model_name_or_path, config)

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
            config.mem_perceiver_config['memory_size_limit'] = 0
            pruner = TovaPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.NO_COMPRESS:
            pruner = build_llm_from_name_or_path(pretrained_model_name_or_path, config) # no compress
        elif pruner_type == PrunerType.ROCO:
            config.use_flash = False
            print('Warning: the roco pruner requires attention scores, therefore the flash_attention is set to False!')
            pruner = ROCOPruner.from_pretrained(pretrained_model_name_or_path, config)
        elif pruner_type == PrunerType.CONV:
            config.use_flash = False
            print('Warning: the conv pruner requires attention scores, therefore the flash_attention is set to False!')
            pruner = ConvPruner.from_pretrained(pretrained_model_name_or_path, config)
        elif pruner_type == PrunerType.RANDOM_QUERY:
            pruner = RandomQueryPruner.from_pretrained(pretrained_model_name_or_path, config)
        elif pruner_type == PrunerType.ONLINE_KMEANS_QUERY:
            pruner = OnlineKmeansPruner.from_pretrained(pretrained_model_name_or_path, config)
        else:
            raise NotImplementedError(f"the {pruner_type} is not supported")

        return pruner

class TovaPruner(CollieModelForCausalLM):
    def __init__(self, config, chunk_size=None, model=None, num_sink_tokens=None, num_layers=None, memory_type=None, memory_size_limit=None, 
                 incremental_type='linear', decremental_chunk=False, review_scheduler=None, **kwargs):
        super().__init__(config)
        self.model = model
        self.chunk_size = chunk_size
        # assert memory_size_limit is not None or compress_ratio is not None, "memory_size_limit or compress_ratio must be set!"
        # assert memory_size_limit is None or compress_ratio is None, "memory_size_limit and compress_ratio can not be set at the same time!"
        
        if self.model is not None:
            self.frozen_model()
        assert num_layers > 0
        self.num_sink_tokens = num_sink_tokens
        self.num_layers = num_layers
        self.memory_type = memory_type
        self.memory_size_limit = memory_size_limit
        self.decremental_chunk = decremental_chunk
        self.transpose_kv = False
        self._supports_cache_class = True

        assert memory_size_limit is not None
        self.incremental_type = incremental_type
        self.chunksize_scheduler = ChunkSizeScheduler(chunk_size)
        self.memsize_scheduler = MemSizeScheduler(chunk_size, max_mem_size=memory_size_limit, incremental_type=incremental_type, num_layers=num_layers)

        if review_scheduler is not None:
            self.review_scheduler = ReviewScheduler(
                scheduler=review_scheduler['scheduler'], 
                review_times=review_scheduler['times'],
                review_chunks=review_scheduler['length'],
                chunk_id_to_review = math.ceil(self.memory_size_limit / self.chunk_size)
            )
        else:
            self.review_scheduler = None
        self.memory_state = MemoryStateManager(num_layers, chunk_size, num_sink_tokens, self.review_scheduler)
        print('WARNING: The following kwargs are not used' + '!'* 20)
        print(kwargs)

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

    def report_memory_state(self):
        self.memory_state.report_state()

    def get_indices(self, attention, attention_shape, target_len, survive_indices=None):
        assert attention is not None
        chunk_size = min(self.chunk_size, attention.shape[-1])
        new_attention_scores = attention[:, :, -self.chunk_size:, :] # (bsz, num_heads, chunk_size, seq_len)
        accum_attention_scores = new_attention_scores.sum(dim = 2) # (bsz, num_heads, seq_len)

        normalization_scores = torch.ones(attention.shape[-1], device=attention.device) * chunk_size
        normalization_scores[-chunk_size:] = chunk_size - torch.arange(chunk_size, device=attention.device)
        # print(normalization_scores)
        important_scores = accum_attention_scores / normalization_scores.view(1, 1, -1).expand_as(accum_attention_scores)
        assert important_scores.shape[-1] >= target_len, f'{important_scores.shape=}, {target_len=}'
        if survive_indices is not None:
            important_scores = torch.scatter(
                important_scores, dim=-1, index=survive_indices, 
                src=torch.full(survive_indices.shape, 1e6, device=important_scores.device, dtype=important_scores.dtype))
        important_indices = torch.topk(important_scores, dim=-1, k=target_len).indices
        important_indices, _ = torch.sort(important_indices, dim=-1) # 方便位置编码
        
        return important_indices

    def compress_layer(self, key, value, attention, target_len, chunk_step, layer_idx, survive_indices=None):
        if key.shape[1] <= target_len:
            return key, value
        bsz, seq_len, num_heads, head_dim = key.shape
        if attention is None:
            attention_shape = [bsz, num_heads, seq_len, seq_len]
        else:
            attention_shape = list(attention.shape)
            assert attention_shape[-1] == seq_len
        if attention_shape[1] != num_heads:
            # support gqa
            assert attention_shape[1] % num_heads == 0
            num_groups = attention_shape[1] // num_heads
            attention = attention.view(bsz, num_groups, num_heads, attention.shape[-2], attention.shape[-1])
            attention = attention.sum(dim=1)
            assert attention.shape[1] == num_heads
            attention_shape[1] = num_heads
        topk_indices = self.get_indices(attention, attention_shape, target_len, survive_indices)
        assert topk_indices.shape[1] == num_heads, f"{topk_indices.shape=}, {attention.shape=}"
        topk_indices = topk_indices.to(key.device)
        if self.memory_type != MemoryType.FIXED_INCREMENTAL:
            self.memory_state.update_state(seq_len, self.prefill_len, topk_indices, chunk_step, layer_idx, attention)
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
                key=key, value=value, attention=attention, chunk_step=chunk_step, layer_idx=i)
            keeped_keys.append(selected_keys)
            keeped_values.append(selected_values)
        # print(f'before compression, key shape: {keys[0].shape}, value shape: {values[0].shape}, attention shape: {attentions[0].shape}')
        # print(f'after compression, key shape: {keeped_keys[0].shape}, value shape: {keeped_values[0].shape}')

        return keeped_keys, keeped_values

    def read_and_write_memory(self, compress_func, **kwargs_of_compress_func):
        chunk_step = kwargs_of_compress_func.get('chunk_step')
        layer_idx = kwargs_of_compress_func.get('layer_idx')
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
            kwargs_of_compress_func['target_len'] = self.memsize_scheduler.get_fix_memsize() # the compressed memory size is constant
            # 类似rmt读写compressed kv cache,但是每次写入会覆盖之前保存的compressed kv cache
            compressed_key, compressed_value = compress_func(**kwargs_of_compress_func)
            return torch.cat([sink_key, compressed_key], dim=1), torch.cat([sink_value, compressed_value], dim=1)
        
        elif self.memory_type == MemoryType.FIXED_INCREMENTAL:
            # 类似AutoCompresser, 压缩后的kv cache都缓存下来, 且都被读取
            memory_size = self.memory_state.memory_indices[layer_idx].shape[-1] if self.memory_state.memory_indices[layer_idx] is not None else 0
            incremental_key = key[:, :memory_size]
            incremental_value = value[:, :memory_size]
            kwargs_of_compress_func['key'] = key[:, memory_size:]
            kwargs_of_compress_func['value'] = value[:, memory_size:]

            if kwargs_of_compress_func['attention'] is not None:
                kwargs_of_compress_func['attention'] = kwargs_of_compress_func['attention'][:, :, :, memory_size:]
            
            kwargs_of_compress_func['target_len'] = self.memsize_scheduler.get_fix_incremental_memsize(chunk_step, self.prefill_len) # the compressed key size is constant
            compressed_key, compressed_value = compress_func(**kwargs_of_compress_func)
            return torch.cat([sink_key, incremental_key, compressed_key], dim=1), torch.cat([sink_value, incremental_value, compressed_value], dim=1)
        
        elif self.memory_type == MemoryType.DYNAMIC_INCREMENTAL:
            # memory在随着chunk数量的增加而变长，但是每次增长会刷新整个memory，而incremental memory只会在之前的基础上拼接新的memory
            kwargs_of_compress_func['target_len'] = self.memsize_scheduler.get_memsize(chunk_step, self.prefill_len, layer_idx, keep_memory_rate=self.memory_state.memory_rention)
            if kwargs_of_compress_func['key'].shape[1] > kwargs_of_compress_func['target_len']:
                compressed_key, compressed_value = compress_func(**kwargs_of_compress_func)
            else:
                compressed_key, compressed_value = kwargs_of_compress_func['key'], kwargs_of_compress_func['value']
                        
            return torch.cat([sink_key, compressed_key], dim=1), torch.cat([sink_value, compressed_value], dim=1)
        
        elif self.memory_type == MemoryType.DualMemory:
            self.compressed_chunk_size = self.memory_size_limit
            assert kwargs_of_compress_func['attention'] is not None
            kwargs_of_compress_func['target_len'] = self.memsize_scheduler.get_fix_memsize() # the compressed memory size is constant
            global_memory_size = int(kwargs_of_compress_func['target_len'] * 0.05)
            if self.cached_frequences[layer_idx] is not None:
                global_memory_freqs, global_memory_indices = torch.topk(self.cached_frequences[layer_idx], dim=-1, k=global_memory_size)
                global_memory_freqs -= 0.5
                self.cached_frequences[layer_idx] = self.cached_frequences[layer_idx].scatter(index=global_memory_indices, dim=-1, src=global_memory_freqs)
                kwargs_of_compress_func['survive_indices'] = global_memory_indices
            # count the frequence of the memory unit and keep the most frequent memory into long-term memory, which is not involved into pruning 
            compressed_key, compressed_value = compress_func(**kwargs_of_compress_func)
            return torch.cat([sink_key, compressed_key], dim=1), torch.cat([sink_value, compressed_value], dim=1)
        else:
            raise NotImplementedError

    def repeat_first_chunk(lst, positions):
        first_chunk = lst[0]
        new_lst = lst[:]
        for pos in sorted(positions, reverse=True):
            if pos <= len(new_lst):
                new_lst.insert(pos, first_chunk)
            else:
                new_lst.append(first_chunk)
        return new_lst

    def estimate_loss(self, input_ids, logits, labels=None):
        # 使用交叉熵损失函数计算损失
        batch_size, sequence_length, vocab_size = logits.shape

        # 将 input_ids 向左偏移一位，去掉第一个 token，最后一个 token 可以填充为 0 或者其他填充值
        if labels is None:
            shifed_labels = input_ids[:, 1:].contiguous()
        else:
            shifed_labels = labels[:, 1:].contiguous()

        # 对 logits 进行相应的切片，去掉最后一个预测，因为它没有对应的标签
        shifted_logits = logits[:, :-1, :].contiguous()

        # 重新计算形状，以便于交叉熵损失的计算
        shifted_logits = shifted_logits.view(-1, vocab_size)
        shifed_labels = shifed_labels.view(-1)

        # 使用交叉熵损失函数计算损失
        loss = F.cross_entropy(shifted_logits, shifed_labels)

        return loss
        

    def forward(self, input_ids: Any | None = None, attention_mask: Any | None = None, past_key_values: Tuple | None = None, labels = None, update_memory=True, **kwargs):
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
            if self.decremental_chunk and self.memory_type == MemoryType.DYNAMIC_INCREMENTAL:
                chunk_sizes = self.chunksize_scheduler.get_decremental_chunk_sizes(seq_len, self.memsize_scheduler)
            else:
                chunk_sizes = self.chunksize_scheduler.get_std_chunk_sizes(seq_len)
            
            chunked_input_ids = torch.split(input_ids, chunk_sizes, dim=1)
            if labels is not None:
                chunked_labels = torch.split(labels, chunk_sizes, dim=1)
            else:
                chunked_labels = chunked_input_ids
            if self.review_scheduler is not None:
                chunked_input_ids = self.review_scheduler.review(list(chunked_input_ids)) # augment chunks by reviewing the first chunk
                chunked_labels = self.review_scheduler.review(list(chunked_labels)) # augment chunks by reviewing the first chunk
            
            self.prefill_len = seq_len
            # chunked_attention_mask = torch.split(attention_mask, self.chunk_size, dim=1)
            num_chunks = len(chunked_input_ids)
            
            cached_llm_outpus = []

            # print("compress prompt...", flush=True)
            # compress the kv cache of prompt
            cached_compressed_kv = past_key_values
            losses = []
            for i in range(num_chunks):
                model_outputs = self.model(chunked_input_ids[i], None, past_key_values=cached_compressed_kv)
                # compress kv cache and update memory
                kv_cache = model_outputs.past_key_values # kv_cache is list of tuple: [(key of layer0, value of layer0), ...]
                if self.transpose_kv:
                    kv_cache = [(kv[0].transpose(1, 2).contiguous(), kv[1].transpose(1, 2).contiguous()) for kv in kv_cache]
                
                attention_scores = model_outputs.attentions

                keys, values = [kv[0].detach() for kv in kv_cache], [kv[1].detach() for kv in kv_cache]
                if self.memory_size_limit != 0 and update_memory:
                    # print(chunked_input_ids[i].shape, flush=True)
                    compressed_keys, compressed_values = self.compress(keys=keys, values=values, attentions=attention_scores, chunk_step=i)
                    # assert compressed_keys.requires_grad and compressed_values.requires_grad
                    cached_compressed_kv = [(ck,cv) for ck, cv in zip(compressed_keys, compressed_values)]# [num_layers, 2, bsz, seq_len, num_heads, head_dim]
                    if self.transpose_kv:
                        cached_compressed_kv = [(kv[0].transpose(1, 2).contiguous(), kv[1].transpose(1, 2).contiguous()) for kv in cached_compressed_kv]
                
                model_outputs =CausalLMOutputWithPast(
                    logits=model_outputs.logits,
                )
                loss = self.estimate_loss(chunked_input_ids[i], model_outputs.logits, chunked_labels[i])
                # assert not torch.isnan(loss).any()
                if not torch.isnan(loss).any():
                    losses.append(loss.item())
                else:
                    losses.append(0) # zero loss will be ignored
                # print(f'step: {i}, loss:{loss}')
                # cached_llm_outpus.append(model_outputs) # drop the first rank, since the first chunk does not use compressed memory
                cached_llm_outpus = model_outputs
            # accum_logits = torch.cat([o.logits for o in cached_llm_outpus], dim=1) # batch_size, seq_len
            accum_logits = cached_llm_outpus.logits
            # TODO: accumulate hidden states
            # print(f"hidden states shape: {cached_llm_outpus[0][0].shape}, {cached_llm_outpus[0][1].shape}, {cached_llm_outpus[0][2].shape}")
            # accum_hidden_states = torch.cat([o.hidden_states for o in cached_llm_outpus], dim=1) # batch_size, seq_len, d_model
            lm_output = CausalLMOutputWithPast(
                loss=losses,
                logits=accum_logits,
                past_key_values=cached_compressed_kv,
                hidden_states=None,
                attentions=None,
            )
            # self.memory_state.clear_state()
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
        mem_perceiver = cls.from_config(config=config, model=model)
        if 'internlm' in model_path_or_name.lower():
            mem_perceiver.transpose_kv = True
        return mem_perceiver

class StreamingLMPruner(TovaPruner):
    def get_indices(self, attention, attention_shape, target_len, survive_indices=None, *args, **kwargs):
        # indices shape: (bsz, num_heads, target_len)
        bsz, num_heads, seq_len = attention_shape[0], attention_shape[1], attention_shape[-1]
        if survive_indices is not None:
            raise NotImplementedError
            # survive_num = survive_indices.shape[-1]
            # other_num = seq_len-survive_num
            # indices = torch.arange(seq_len)[-other_num:]
            # indices = indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, other_num)
            # indices = torch.cat([survive_indices, indices], dim=-1)
        else:
            indices = torch.arange(seq_len)[-target_len:]
            indices = indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, target_len)
        return indices

class RandomPruner(TovaPruner):
    # sink tokens + random tokens
    def get_indices(self, attention, attention_shape, target_len, survive_indices=None, *args, **kwargs):
        bsz, num_heads, seq_len = attention_shape[0], attention_shape[1], attention_shape[-1]
        assert seq_len >= target_len
        if survive_indices is not None:
            raise NotImplementedError
        else:
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
    def __init__(self, config, kernel_size=5, **kwargs):
        super().__init__(config, **kwargs)
        self.kernel_size = kernel_size
        assert kernel_size % 2 != 0

    def get_indices(self, attention, attention_shape, target_len, survive_indices=None):
        assert attention is not None
        chunk_size = min(self.chunk_size, attention.shape[-1])
        new_attention_scores = attention[:, :, -self.chunk_size:, :] # (bsz, num_heads, chunk_size, seq_len)
        accum_attention_scores = new_attention_scores.sum(dim = 2) # (bsz, num_heads, seq_len)
        
        accum_attention_scores = nn.functional.avg_pool1d(accum_attention_scores, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        assert accum_attention_scores.shape[-1] == attention_shape[-1], f'{accum_attention_scores.shape=}, {attention_shape=}' 
        
        normalization_scores = torch.ones(attention.shape[-1], device=attention.device) * chunk_size
        normalization_scores[-chunk_size:] = chunk_size - torch.arange(chunk_size, device=attention.device)
        important_scores = accum_attention_scores / normalization_scores.view(1, 1, -1).expand_as(accum_attention_scores)

        if survive_indices is not None:
            raise NotImplementedError
        assert important_scores.shape[-1] >= target_len, f"{important_scores.shape[-1]}, {target_len}"
        important_indices = torch.topk(important_scores, dim=-1, k=target_len).indices
        important_indices, _ = torch.sort(important_indices, dim=-1) # 方便位置编码

        return important_indices

class ROCOPruner(TovaPruner):
    def get_indices(self, attention, attention_shape, target_len, survive_indices=None):
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
        _, std_removed_indices = torch.topk(std_scores, k=target_len // 2, dim=-1, largest=False) # remove the 
        
        mask_for_important_scores = torch.zeros_like(important_scores).bool()
        assert not torch.any(mask_for_important_scores)
        important_scores = torch.scatter(
            important_scores, 
            dim=-1, 
            index=std_removed_indices, 
            src=torch.zeros_like(std_removed_indices).type_as(important_scores)) # mask the kv with lowest variance

        if survive_indices is not None:
            raise NotImplementedError
        important_indices = torch.topk(important_scores, dim=-1, k=target_len).indices

        important_indices, _ = torch.sort(important_indices, dim=-1) # 方便位置编码

        return important_indices

class RandomQueryPruner(TovaPruner):
    def __init__(self, config, d_model=None, num_querys=128, **kwargs):
        super().__init__(config, **kwargs)
        self.query = torch.randn([num_querys, d_model])
    
    def compress_layer(self, key, value, attention, target_len, chunk_step, layer_idx, survive_indices=None):
        if key.shape[1] <= target_len:
            return key, value
        bsz, seq_len, num_heads, head_dim = key.shape
        if attention is None:
            attention_shape = [bsz, num_heads, seq_len, seq_len]
        else:
            attention_shape = list(attention.shape)
            assert attention_shape[-1] == seq_len
        if num_heads * head_dim == self.query.shape[-1]:
            # mha
            query = self.query.view(-1, num_heads, head_dim)
        else:
            # gqa
            num_querys = self.query.shape[0]
            query = self.query.view(num_querys, -1, num_heads, head_dim)[:, 0]
        if query.device != key.device or query.dtype != key.dtype:
            query = query.to(key)
        attention_scores = torch.einsum('qhd,bshd -> bqsh', query, key) # (bsz, query_length, key_length, num_heads)  
        attention_scores = attention_scores / (head_dim ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=2) 
        attention_scores = attention_scores.sum(dim=1) # (bsz, key_length, num_heads)  
        topk_indices = torch.topk(attention_scores, dim=1, k=target_len, largest=True).indices # (bsz, target_len, num_heads, )
        topk_indices = topk_indices.unsqueeze(dim=-1).expand(bsz, target_len, num_heads, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
        selected_keys = torch.gather(key, index=topk_indices, dim=1)
        selected_values = torch.gather(value, index=topk_indices, dim=1)

        return selected_keys, selected_values

class OnlineKmeansPruner(TovaPruner): 
    """
    参考sklearn的实现：
    - 初始化：随机从样本中sample簇中心
    - assignment: 计算样本assign的簇中心
    - update: 利用移动滑动平均更新簇中心
    """
    def __init__(self, config, d_model=None, num_querys=128, **kwargs):
        super().__init__(config, **kwargs)
        self.query = None
        self.count = None
        self.num_querys = num_querys
    
    def init_query(self, key):
        bsz, seq_len, num_heads, head_dim = key.shape
        key = key.view(-1, num_heads, head_dim) # bsz和seq_len被认为是一个维度，或者说batch_size应该是1
        query = []
        for head_idx in range(num_heads):
            indices = torch.randperm(bsz * seq_len)[:self.num_querys]
            selected_key = key[indices, head_idx] # (num_querys, head_dim)
            query.append(selected_key)
        self.query = torch.stack(query, dim=1) # (num_querys, num_heads, head_dim)
        self.count = torch.zeros(self.num_querys, num_heads, 1).to(self.query)

    def assign_key(self, key):
        bsz, seq_len, num_heads, head_dim = key.shape
        key = key.view(-1, num_heads, head_dim) # bsz和seq_len被认为是一个维度，或者说batch_size应该是1
        distances = torch.linalg.norm(self.query.unsqueeze(dim=0) - key.unsqueeze(dim=1), axis=-1) # (bsz*seq_len, num_querys, num_heads)
        closest_centroid_idx = torch.argmin(distances, dim=1) # (bsz*seq_len, num_heads)
        return closest_centroid_idx

    def update_query(self, key, labels):
        bsz, seq_len, num_heads, head_dim = key.shape
        key = key.view(-1, num_heads, head_dim)
        # labels: (bsz * seq_len, num_heads)
        # keys: (bsz*seq_len, num_heads, head_dim)
        # query: (num_querys, num_heads, head_dim)
        new_count = (labels.unsqueeze(dim=0) == torch.arange(self.num_querys).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(self.num_querys, bsz*seq_len, num_heads).to(labels)) 
        new_count = new_count.sum(dim=1).unsqueeze(dim=-1)

        labels = labels.unsqueeze(dim=-1).expand(bsz * seq_len, num_heads, head_dim)
        new_query = self.query * self.count
        new_query = torch.scatter_add(dim=0, index=labels, input=new_query, src=key)
        # (1, bsz*seq_len, num_heads) == (num_querys, 1, 1) => (num_querys, bsz*seq_len, num_heads)
        self.query = new_query / new_count
        self.count = new_count

    def compress_layer(self, key, value, attention, target_len, chunk_step, layer_idx, survive_indices=None):
        if key.shape[1] <= target_len:
            return key, value
        bsz, seq_len, num_heads, head_dim = key.shape
        
        if self.query is None:
            self.init_query(key)
        labels = self.assign_key(key)
        self.update_query(key, labels)

        attention_scores = torch.einsum('bshd,qhd -> bsqh', key, self.query) # (bsz, key_length, query_length, num_heads)  
        assert not torch.any(torch.isinf(self.query)) and not torch.any(torch.isnan(self.query))
        attention_scores = attention_scores / (head_dim ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=1) 
        attention_scores = attention_scores.sum(dim=2) # (bsz, key_length, num_heads)  
        topk_indices = torch.topk(attention_scores, dim=1, k=target_len, largest=True).indices # (bsz, target_len, num_heads, )
        topk_indices = topk_indices.unsqueeze(dim=-1).expand(bsz, target_len, num_heads, head_dim) # (bsz, num_heads, target_len, 1) -> (bsz, num_heads, target_len, head_dim)
        selected_keys = torch.gather(key, index=topk_indices, dim=1)
        selected_values = torch.gather(value, index=topk_indices, dim=1)
        return selected_keys, selected_values

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
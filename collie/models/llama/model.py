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
import numpy as np
from torch.nn.parameter import Parameter

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x1 = x[..., :x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2:]
    # return torch.cat((-x2, x1), dim=-1)
    return torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, config) -> None:
        super().__init__()
        self.pe_config = config.pe_config
        self.head_dim = head_dim
        
        # batch_size, seq_len, n_head (fixed for tp), head_dim 
        
        if self.pe_config['imp']:
            order, beta = 3,  1 / math.log(10000.0)  # 0.10856
            start, end = np.power(0.0005, 1 / order), np.power(0.9999, 1 / order)  # 
            if self.pe_config['1d']:
                omega = np.power(np.linspace(start, end, head_dim), order)
            else:
                omega = np.power(np.linspace(start, end, head_dim // 2), order)
                omega = np.stack([omega, omega], axis=-1).reshape((head_dim))
                # omega = np.concatenate([omega, omega], axis=-1)
            omega = omega[None, None, None, ::-1].copy()
            expos = (beta / omega) / (1/order * np.power(omega, 1/order - 1))
        else:
            alpha = self.pe_config['ntk_alpha'] if self.pe_config['ntk_option'] == 'fixed' else 1
            if self.pe_config['1d']:
                base = self.pe_config['base'] * alpha ** (self.head_dim / (self.head_dim - 1))
                omega = 1.0 / (base ** (np.arange(0, head_dim, 1) / head_dim))
            else:
                base = self.pe_config['base'] * alpha ** (self.head_dim / (self.head_dim - 2))
                omega = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
                omega = np.stack([omega, omega], axis=-1).reshape((head_dim))
                # omega = np.concatenate([omega, omega], axis=-1)
            omega = omega[None, None, None, :] / self.pe_config['pi_lambda']
            expos = np.ones_like(omega)
        self.register_buffer("omega", torch.tensor(omega), persistent=False)
        self.register_buffer("expos", torch.tensor(np.sqrt(expos)), persistent=False)

        if self.pe_config['exp']:
            if self.pe_config['imp']:
                scale = - np.log(omega) / np.log(10000.0) * head_dim
            else:
                scale = np.arange(0, head_dim, 1 if self.pe_config['1d'] else 2)
                scale = scale if self.pe_config['1d'] else np.stack([scale, scale], axis=-1).reshape((head_dim))
                # scale = scale if self.pe_config['1d'] else np.concatenate([scale, scale], axis=-1)
                scale = scale[None, None, None, :]
            scale = (scale + 0.4 * head_dim) / (1.4 * head_dim)
            self.register_buffer("scale", torch.tensor(scale), persistent=False)

        # inv_freq = 1.0 / (10000.0 ** (
        #     torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        # self.register_buffer('inv_freq', inv_freq)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                seq_len,
                positions: torch.Tensor = None,
                start_pos: int = 0):
        dtype = query.dtype
        
        # batch_size, seq_len, n_head (fixed for tp), head_dim
        if positions is None:
            positions = torch.arange(seq_len, device=query.device)
        t = positions.reshape(1, -1, 1, 1).float()
        if self.pe_config['ntk_option'] == 'dynamic': 
            if self.pe_config['imp']:
                raise KeyError('ntk for imp is not currently supported')
            # copy from https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py#L379
            base = max(2 ** math.ceil(math.log(seq_len / self.pe_config['max_length'], 2) + 1) - 1, 1)
            alpha = self.pe_config['base'] * base ** (self.head_dim / (self.head_dim - (1 if self.pe_config['1d'] else 2)))
            theta = 1.0 / (alpha ** (torch.arange(0, self.head_dim, 1 if self.pe_config['1d'] else 2, 
                                                  device='cuda') / self.head_dim)).float()
            theta = theta if self.pe_config['1d'] else torch.stack([theta, theta], axis=-1).reshape((self.head_dim))
            theta = theta[None, None, None, :]
            self.scale = (torch.clamp(-torch.log(theta)/math.log(10000.0), max=1) + 0.4) / 1.4
        else:
            theta = self.omega.float()
        theta = theta * t
        sin, cos, expos = torch.sin(theta), torch.cos(theta), self.expos.float()

        q, k = query.float() * expos, key.float() * expos

        if self.pe_config['1d']:
            q_real = torch.cat([q * cos, q * sin], dim=-1)
            k_real = torch.cat([k * cos, k * sin], dim=-1)
            # q_imag = torch.cat([q * sin, -q * cos], dim=-1)
        else:
            q_real = q * cos + rotate_half(q) * sin
            k_real = k * cos + rotate_half(k) * sin
            # q_imag = q * sin - rotate_half(q) * cos
            
        if self.pe_config['exp']:
            scale = self.scale ** ((t - seq_len // 2) / self.pe_config['exp_base'])
            scale = scale if not self.pe_config['1d'] else torch.cat([scale, scale], dim=-1)
            q_real = q_real * scale
            k_real = k_real / scale

        if self.pe_config['log']:
            q_real = q_real * torch.clamp(torch.log(t+1) / math.log(self.pe_config['log_base']), min=self.pe_config['log_clip'])

        # query = torch.view_as_complex(
        #     query.float().reshape(*query.shape[:-1], -1, 2))
        # key = torch.view_as_complex(
        #     key.float().reshape(*key.shape[:-1], -1, 2))
        # freqs = torch.outer(torch.arange(
        #     (2 ** 16) * 2, device=self.inv_freq.device), self.inv_freq).float()
        # freqs_cis = torch.polar(torch.ones_like(freqs), freqs)[
        #     start_pos: start_pos + seq_len]
        # shape = [d if i == 1 or i == query.ndim -
        #          1 else 1 for i, d in enumerate(query.shape)]
        # freqs_cis = freqs_cis.view(*shape)
        # query = torch.view_as_real(query * freqs_cis).flatten(3)
        # key = torch.view_as_real(key * freqs_cis).flatten(3)
        # return query.type(t), key.type(t)
        
        return q_real.type(dtype), k_real.type(dtype)


class RMSNormalize(nn.Module):
    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None):
        super(RMSNormalize, self).__init__()
        self.dtype = dtype
        if weight is not None:
            self.weight = weight
        else:
            self.weight = nn.Parameter(
                torch.ones(
                    dim, dtype=dtype, device=get_accelerator().current_device_name()
                )
            )
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight

    def post_init(self):
        """
        用于 from scratch 初始化时，使用不同于 CollieConfig 里的初始化方法。
        """
        if self.weight.device == torch.device("meta"):
            device = 'cpu'
        else:
            device = self.weight.device
        return torch.ones(
            self.weight.shape, dtype=self.dtype, device=device
        )

class LlamaLayer(nn.Module):
    def __init__(self, config: CollieConfig, layer_idx) -> None:
        super().__init__()
        self.idx = layer_idx
        self.config = config
        if hasattr(config, "num_key_value_heads"):
            # llama2 (transformers >= 4.31.0)
            self.num_key_value_heads = config.num_key_value_heads
        else:
            self.num_key_value_heads = config.num_attention_heads
        self.num_key_value_groups = (
            config.num_attention_heads // self.num_key_value_heads
        )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    self.num_heads * self.head_dim,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "k_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "v_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "o_proj": RowParallelLinearWithoutBias(
                    self.num_heads * self.head_dim,
                    config.hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    init_method=lambda x: x,
                ),
                "rotary_emb": RotaryPositionEmbedding(
                    self.config.hidden_size // self.config.num_attention_heads,
                    config=config
                ),
            }
        )
        self.input_layernorm = RMSNormalize(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    config.intermediate_size,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "up_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    config.intermediate_size,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                ),
                "down_proj": RowParallelLinearWithoutBias(
                    config.intermediate_size,
                    config.hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    init_method=lambda x: x,
                ),
            }
        )
        self.post_attention_layernorm = RMSNormalize(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )
        # 务必保持变量名一致
        self.use_cache = self.config.model_config.use_cache
        self.hidden_states = None

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not self.training:
            self.hidden_states = hidden_states
        else:
            self.hidden_states = None
        layer_past = None
        if past_key_values is not None:
            layer_past = past_key_values
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states.shape must be (B, N, H), but got {hidden_states.shape}"
        batch_size, seq_len, _ = hidden_states.shape
        _hidden_states = self.input_layernorm(hidden_states)
        query, key, value = (
            self.self_attn["q_proj"](_hidden_states),
            self.self_attn["k_proj"](_hidden_states),
            self.self_attn["v_proj"](_hidden_states),
        )
        query, key, value = (
            rearrange(query, "b n (h d) -> b n h d", d=self.head_dim),
            rearrange(key, "b n (h d) -> b n h d", d=self.head_dim),
            rearrange(value, "b n (h d) -> b n h d", d=self.head_dim),
        )
        if layer_past is not None:
            if self.config.peft_config and self.config.peft_config.peft_type == "PREFIX_TUNING":
                start_pos = layer_past[0].shape[2]
            else:
                start_pos = layer_past[0].shape[1]
        else:
            start_pos = 0
        
        if layer_past is not None:
            # past_key: batch_size, num_heads, seq_len, head_dim
            if self.config.peft_config and self.config.peft_config.peft_type == "PREFIX_TUNING":
                past_key = layer_past[0].reshape(*layer_past[0].shape[:-1], 2, -1)\
                                        .permute(0, 2, 1, 4, 3)\
                                        .reshape(batch_size, start_pos, self.num_heads, -1)
                past_value = layer_past[1].permute([0, 2, 1, 3])
            else:
                past_key, past_value = layer_past

            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)

        new_layer_past = None
        if self.use_cache: # TODO: 删除了not self.training的判断，但是有问题，因为这个调整只适合perceiver，而我们改了llama本身的代码
            # 调整成和 hf 兼容的格式，方便 prefix tuning
            if self.config.peft_config and self.config.peft_config.peft_type == "PREFIX_TUNING":
                present_key = key.reshape(*key.shape[:-1], -1, 2)\
                                .permute(0, 2, 1, 4, 3)\
                                .reshape(batch_size, self.num_heads // self.config.tp_size, seq_len + start_pos, -1)
                present_value = value.permute([0, 2, 1, 3])
                new_layer_past = (present_key, present_value)
            else:
                new_layer_past = (key, value)
        
        if self.num_key_value_groups > 1:
            key = torch.repeat_interleave(key, dim=2, repeats=self.num_key_value_groups)
            value = torch.repeat_interleave(
                value, dim=2, repeats=self.num_key_value_groups
            )
        if layer_past is not None:
            query = torch.nn.functional.pad(query, (0, 0, 0, 0, past_key.shape[1], 0))
            assert query.shape == key.shape

        positions = torch.arange(query.shape[1], device=query.device)
        # if layer_past is not None:
        #     previous_key_len = past_key.shape[1]
        #     positions = positions - previous_key_len + 1 # 0, 1 , 2, 3 (previous_key_len=2) => -1, 0, 1, 2
        #     positions[:previous_key_len] = 0
        #     assert torch.all(positions >= 0)
        seq_len_for_rope = positions[-1].item()
        query, key = self.self_attn["rotary_emb"](query, key, seq_len_for_rope, positions, start_pos)
        
        attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones((query.shape[0], query.shape[1])).to(hidden_states.device)
        )
        if self.config.use_flash:
            output = flash_attention(query, key, value, attention_mask)
            attention_score = None
        else:
            query, key, value = (
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
            )
            attention_score = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            if seq_len + start_pos > 1:
                mask = torch.full(
                    (1, 1, seq_len + start_pos, seq_len + start_pos), float("-inf")
                )
                mask = torch.triu(mask, diagonal=1).to(attention_score.device)
                attention_score = attention_score + mask
            key_padding_mask = (
                1.0 - attention_mask.unsqueeze(1).unsqueeze(2)
            ) * torch.finfo(attention_score.dtype).min
            attention_score = F.softmax(
                attention_score + key_padding_mask, dim=-1
            ).type_as(value)
            output = torch.matmul(attention_score, value)
            output = (
                output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len + start_pos, -1)
            )
        output = F.dropout(output, p=self.config.dropout, training=self.training)
        output = output[:, start_pos:, :]
        hidden_states = hidden_states + self.self_attn["o_proj"](output)
        _hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + F.dropout(
            self.mlp["down_proj"](
                F.silu(self.mlp["gate_proj"](_hidden_states))
                * self.mlp["up_proj"](_hidden_states)
            ),
            p=self.config.dropout,
            training=self.training,
        )
        return hidden_states, new_layer_past, attention_score

    def forward(self, inputs: dict):
        layer_past = inputs_to_kv_cache_for_layer(idx=self.idx, inputs=inputs)

        if self.config.checkpointing and self.training:
            # if layer_past is not None:
            #     assert layer_past.requires_grad
            hidden_states, new_layer_past, attention_score = torch.utils.checkpoint.checkpoint(
                self._forward,
                inputs["hidden_states"],
                inputs.get("attention_mask", None),
                layer_past,  # inputs.get("past_key_values", None),
            )
        else:
            hidden_states, new_layer_past, attention_score = self._forward(inputs["hidden_states"],
                                                          inputs.get("attention_mask", None), 
                                                          layer_past)  # **inputs
        inputs["hidden_states"] = hidden_states
        inputs[f"attention_score_{self.idx}"] = attention_score
        inputs.update(kv_cache_to_inputs_for_layer(idx=self.idx, new_layer_past=new_layer_past))
        return inputs


class LlamaModel(nn.Module):
    def __init__(self, config: CollieConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [LlamaLayer(self.config, i) for i in range(self.config.num_hidden_layers)]
        )
        self.norm = RMSNormalize(
            dim=self.config.hidden_size, eps=self.config.rms_norm_eps
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if input_ids == None:
            inputs["hidden_states"] = kwargs["inputs_embeds"]
        else:
            inputs["hidden_states"] = self.embed_tokens(inputs["input_ids"])

        inputs.update(kv_cache_to_inputs_for_model(past_key_values))
            
        all_hidden_states = ()
        for layer in self.layers:
            all_hidden_states += (inputs["hidden_states"],)
            inputs.update(layer(inputs))
        inputs["hidden_states"] = self.norm(inputs["hidden_states"])
        all_hidden_states += (inputs["hidden_states"],)

        past_key_values = inputs_to_kv_cache_for_model(self.config.num_hidden_layers, inputs)
        attention_scores = [inputs[f"attention_score_{i}"] for i,_ in enumerate(self.layers)]

        return BaseModelOutputWithPast(
            last_hidden_state=inputs["hidden_states"],
            hidden_states=all_hidden_states,
            past_key_values=past_key_values,
            attentions=attention_scores
        )

    @classmethod
    def pipeline_layers(cls, config: CollieConfig):
        """
        Get layers of pipeline.

        :return: list
        """
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)

        if config.tie_word_embeddings:
            embed_tokens = TiedLayerSpec(
                "embed_tokens",
                dict_as_params(input_keys="input_ids", output_keys="hidden_states"),
                tensor_parallel.VocabParallelEmbedding,
                config.vocab_size,
                config.hidden_size,
            )
        else:
            embed_tokens = LayerSpec(
                dict_as_params(input_keys="input_ids", output_keys="hidden_states"),
                tensor_parallel.VocabParallelEmbedding,
                config.vocab_size,
                config.hidden_size,
            )

        layers = [
            LayerSpec(LlamaLayer, config, i) for i in range(config.num_hidden_layers)
        ]
        norm = LayerSpec(
            dict_as_params(input_keys="hidden_states", output_keys="hidden_states"),
            RMSNormalize,
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        return [
            ("embed_tokens", embed_tokens),
            ("layers", layers),
            ("norm", norm),
        ]


class LlamaForCausalLM(CollieModelForCausalLM):
    base_model_prefix = "model"

    def __init__(self, config: CollieConfig, **kwargs) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = ColumnParallelLinearWithoutBias(
            self.collie_config.hidden_size, self.collie_config.vocab_size, bias=False
        )
        # GenerationMixin 需要的额外参数
        self.config.is_decoder = True
        if config.model_config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.main_input_name = "input_ids"
        self._supports_cache_class = True

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        logits = self.lm_head(output.last_hidden_state)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def clean_cache(self):
        self._clean_hidden_states([*self.model.layers, self.lm_head])
        self._set_use_cache(self.model.layers, False)

    def set_cache(self, use_cache):
        self._set_use_cache(self.model.layers, use_cache)

    @classmethod
    def pipeline_layers(cls, config: CollieConfig):
        """
        Get layers of pipeline.

        :return: list
        """
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)

        if config.tie_word_embeddings:
            lm_head = TiedLayerSpec(
                "embed_tokens",
                dict_as_params(input_keys="hidden_states", output_keys="logits"),
                ColumnParallelLMHead,
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        else:
            lm_head = LayerSpec(
                dict_as_params(input_keys="hidden_states", output_keys="logits"),
                ColumnParallelLMHead,
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        return [("model", LlamaModel.pipeline_layers(config)), ("lm_head", lm_head)]

    @staticmethod
    def load_parallel_state_dict(
        path: str,
        config: Union[CollieConfig, str],
        process_exclusion: bool = False,
        **kwargs,
    ):
        ...

    @staticmethod
    def load_parallel_state_dict(
        path: str,
        config: Union[CollieConfig, str],
        process_exclusion: bool = False,
        protocol: str = "file",
        **kwargs,
    ):
        """
        Load state_dict from ``path``.

        The format of pretrained model should be the same as that of
        `huggingface`.

        :return: state_dict. Note that the state_dict should be processed
            properly to match the current rank.
        """
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)
        io_driver = IODriver.from_protocol(protocol)
        if not io_driver.exists(path):
            raise FileNotFoundError(f"folder {path} not found.")
        state_dict = OrderedDict()
        weights = []
        parts = None
        # 如果开启了进程互斥，那么每个进程都会显示进度条，否则只显示 RANK0 的
        hide_progress = not process_exclusion and int(os.environ.get("RANK", "0")) != 0
        if dist.is_initialized() and process_exclusion:
            # 如果启动了进程互斥，则要进行 dist.get_world_size() 次循环
            rank_order = range(dist.get_world_size())
        else:
            # 不开启只进行一次循环
            rank_order = range(1)
        for rank in rank_order:
            # 如果开启了进程互斥，那么只有对应 RANK 的能进入循环；不开启进程互斥的话就都可以进
            if int(os.environ.get("RANK", "0")) == rank or not process_exclusion:
                # PP 分层的方法保存在了 os.environ["COLLIE_PP_PARTS"], 格式类似于 [0, 17, 35], 左闭右开
                if env.is_pipeline:
                    # 保存的是 json 格式
                    parts = env.pipeline_parts
                if hasattr(config, "num_key_value_heads"):
                    # llama2 (transformers >= 4.31.0)
                    num_key_value_heads = config.num_key_value_heads
                else:
                    num_key_value_heads = config.num_attention_heads
                head_dim = config.hidden_size // config.num_attention_heads
                # 如果存在 .index.json 文件的话，此时不同的 pp 进程可以按需加载自己需要的权重
                # 优先加载 model.safetensors.index.json 文件中的权重
                if io_driver.exists(os.path.join(path, "model.safetensors.index.json")):
                    index_json_file_path = os.path.join(path, "model.safetensors.index.json")
                elif io_driver.exists(os.path.join(path, "pytorch_model.bin.index.json")):
                    index_json_file_path = os.path.join(path, "pytorch_model.bin.index.json")
                else:
                    index_json_file_path = None
                if (
                    index_json_file_path is not None
                    and "COLLIE_PP_PARTS" in os.environ.keys()
                ):
                    weight_map = json.loads(
                        io_driver.load(
                            index_json_file_path, mode="r"
                        )
                    )["weight_map"]
                    # layers 表示自己需要的层
                    layers = env.pipeline_layers_idx
                    # 筛选出形似 model.layers.0 这样的层。包含两个条件：1. 有数字的层；2. 数字加一要在 layers 里面（因为最开始还有个 embedding 占一层）
                    weights.extend(
                        [
                            value
                            for key, value in weight_map.items()
                            if len(key.split(".")) > 2
                            and key.split(".")[2].isdigit()
                            and (int(key.split(".")[2]) + 1) in layers
                        ]
                    )
                    # 去重
                    weights = list(set(weights))
                    # 继续筛选，如果有 0 层，那么就要加载 embedding；如果有最后一层，那么就要加载 lm_head；如果有倒数第二层，那么就要加载 norm
                    if 0 in layers:
                        weights.append(weight_map["model.embed_tokens.weight"])
                    if max(parts) - 1 in layers:
                        weights.append(weight_map["lm_head.weight"])
                    if max(parts) - 2 in layers:
                        weights.append(weight_map["model.norm.weight"])
                else:
                    # 如果没有 .index.json 文件的话，那么就加载所有的权重
                    # 优先加载 safetensors 存储的权重
                    weights = [
                        weight
                        for weight in io_driver.list(path)
                        if weight.endswith(".safetensors")
                    ]
                    if len(weights) == 0:
                        # 如果没有 safetensors 文件，那么就加载 bin 文件
                        weights = [
                            weight
                            for weight in io_driver.list(path)
                            if weight.endswith(".bin")
                        ]
                with progress(
                    weights,
                    desc="Loading state dict",
                    total=len(weights),
                    disable=hide_progress,
                ) as pbar:
                    for weight in pbar:
                        part_state_dict = io_driver.load(
                            os.path.join(path, weight), mode="rb"
                        )
                        for key in list(part_state_dict.keys()):
                            # 对 q_proj.weight 和 k_proj.weight 进行 reshape
                            if key.endswith("q_proj.weight"):
                                part_state_dict[key] = (
                                    rearrange(
                                        part_state_dict[key],
                                        "(h two t) d -> h two t d",
                                        h=config.num_attention_heads,
                                        two=2,
                                    )
                                    .transpose(1, 2)
                                    .reshape(config.hidden_size, config.hidden_size)
                                )
                            elif key.endswith("k_proj.weight"):
                                part_state_dict[key] = (
                                    rearrange(
                                        part_state_dict[key],
                                        "(h two t) d -> h two t d",
                                        h=num_key_value_heads,
                                        two=2,
                                    )
                                    .transpose(1, 2)
                                    .reshape(
                                        num_key_value_heads * head_dim,
                                        config.hidden_size,
                                    )
                                )
                        state_dict.update(part_state_dict)
                        del part_state_dict
                if parts is not None:
                    # 这一步是 pp 的复筛
                    layers = env.pipeline_layers_idx
                    for key in list(state_dict.keys()):
                        if key.startswith("layers"):
                            layer = int(key.split(".")[1])
                            if layer + 1 not in layers:
                                state_dict.pop(key)
                        if key.endswith("embed_tokens.weight"):
                            if 0 not in layers:
                                state_dict.pop(key)
                        if key == "norm.weight":
                            if max(parts) - 2 not in layers:
                                state_dict.pop(key)
                        if key.endswith("lm_head.weight"):
                            if max(parts) - 1 not in layers:
                                state_dict.pop(key)
                # 根据用户配置的新的 tp size 进行分割
                for key in list(state_dict.keys()):
                    col_filter = [
                        "q_proj.weight",
                        "k_proj.weight",
                        "v_proj.weight",
                        "gate_proj.weight",
                        "up_proj.weight",
                        "embed_tokens.weight",
                        "lm_head.weight",
                    ]
                    col_split = any([key.endswith(filter) for filter in col_filter])

                    if col_split:
                        tensor = (
                            list(torch.chunk(state_dict[key], config.tp_size, dim=0))[
                                env.tp_rank
                            ]
                            .detach()
                            .clone()
                        )
                        del state_dict[key]
                        if process_exclusion:
                            # CPU 内存回收（速度很慢）
                            gc.collect()
                        state_dict[key] = tensor
                    elif key.endswith("o_proj.weight") or key.endswith(
                        "down_proj.weight"
                    ):
                        tensor = (
                            list(torch.chunk(state_dict[key], config.tp_size, dim=1))[
                                env.tp_rank
                            ]
                            .detach()
                            .clone()
                        )
                        del state_dict[key]
                        if process_exclusion:
                            # CPU 内存回收（速度很慢）
                            gc.collect()
                        state_dict[key] = tensor
            if dist.is_initialized() and process_exclusion:
                # 如果选择了进程互斥，那么本次循环中不需要加载权重的进程需等待
                dist.barrier()
        return state_dict

    @staticmethod
    def save_parallel_state_dict(
        state_dict: dict,
        path: str,
        config: CollieConfig,
        process_exclusion: bool = False,
        **kwargs,
    ):
        ...

    @staticmethod
    def save_parallel_state_dict(
        state_dict: dict,
        path: str,
        config: CollieConfig,
        process_exclusion: bool = False,
        protocol: str = "file",
    ):
        """
        Save state_dict to ``path``.

        The format of saved state dict should be the same as that of
        `huggingface`.
        """
        io_driver = IODriver.from_protocol(protocol)

        def reshape_wq_wk(w: torch.Tensor, kv=False):
            if hasattr(config, "num_key_value_heads"):
                # llama2 (transformers >= 4.31.0)
                num_key_value_heads = config.num_key_value_heads
            else:
                num_key_value_heads = config.num_attention_heads
            head_dim = config.hidden_size // config.num_attention_heads
            if kv:
                num_heads = num_key_value_heads
            else:
                num_heads = config.num_attention_heads
            return (
                w.view(num_heads, head_dim // 2, 2, config.hidden_size)
                .transpose(1, 2)
                .reshape(num_heads * head_dim, config.hidden_size)
            )

        # gather to tp rank 0
        if dist.is_initialized() and process_exclusion:
            # 如果启动了进程互斥，则要进行 pp_size 次循环
            rank_order = range(config.pp_size)
        else:
            # 不开启只进行一次循环
            rank_order = range(1)
        dst = parallel_state.get_tensor_model_parallel_src_rank()
        with progress(
            rank_order,
            desc="Saving model",
            disable=int(os.environ.get("RANK", "0")) != 0,
        ) as pbar:
            for rank in pbar:
                if env.dp_rank == 0 and (env.pp_rank == rank or not process_exclusion):
                    for key in sorted(list(state_dict.keys())):
                        tensor_list = None
                        if env.tp_rank == 0:
                            tensor_list = [
                                torch.zeros_like(state_dict[key])
                                .to(state_dict[key].dtype)
                                .cuda()
                                for _ in range(config.tp_size)
                            ]
                        dist.gather(
                            state_dict[key].cuda(),
                            dst=dst,
                            gather_list=tensor_list,
                            group=env.tp_group,
                        )
                        if env.tp_rank == 0:
                            col_filter = [
                                "q_proj.weight",
                                "k_proj.weight",
                                "v_proj.weight",
                                "gate_proj.weight",
                                "up_proj.weight",
                                "embed_tokens.weight",
                                "lm_head.weight",
                            ]
                            col_split = any(
                                [key.endswith(filter) for filter in col_filter]
                            )

                            if col_split:
                                state_dict[key] = concat_tensor(tensor_list, dim=0)

                                if process_exclusion:
                                    # CPU 内存回收（速度很慢）
                                    gc.collect()

                            elif key.endswith("o_proj.weight") or key.endswith(
                                "down_proj.weight"
                            ):
                                state_dict[key] = concat_tensor(tensor_list, dim=1)

                                if process_exclusion:
                                    # CPU 内存回收（速度很慢）
                                    gc.collect()
                            if key.endswith("q_proj.weight"):
                                state_dict[key] = reshape_wq_wk(state_dict[key])
                            if key.endswith("k_proj.weight"):
                                state_dict[key] = reshape_wq_wk(
                                    state_dict[key], kv=True
                                )
                    if env.tp_rank == 0:
                        # Save gathered weights
                        if env.is_pipeline:
                            ckpt_name = f"pytorch_model-{env.pp_rank+1:05d}-of-{config.pp_size:05d}.bin"
                            total_size = 0
                            weight_map = {}
                            for name, weight in state_dict.items():
                                weight_size = weight.numel() * dtype_byte_size(
                                    weight.dtype
                                )
                                weight_map[name] = ckpt_name
                                total_size += weight_size
                            index_dict = dict(
                                total_size=total_size, weight_map=weight_map
                            )
                            index_dicts = [None for _ in range(env.pp_size)]
                            dist.gather_object(
                                index_dict, index_dicts if env.pp_rank == 0 else None, group=env.pp_group
                            )
                            if env.pp_rank == 0:
                                total_size = 0
                                weight_map = {}
                                for _index_dict in index_dicts:
                                    total_size += _index_dict["total_size"]
                                    weight_map.update(_index_dict["weight_map"])
                                merged_dict = {
                                    "metadata": {"total_size": total_size},
                                    "weight_map": weight_map,
                                }
                                io_driver.save(
                                    json.dumps(merged_dict, indent=2, sort_keys=True)
                                    + "\n",
                                    os.path.join(path, "pytorch_model.bin.index.json"),
                                )

                        else:
                            ckpt_name = f"pytorch_model.bin"
                        ckpt_path = os.path.join(path, ckpt_name)
                        io_driver.save(state_dict, ckpt_path)
                if dist.is_initialized() and process_exclusion:
                    dist.barrier()
        if env.rank == 0:
            config.save_pretrained(path, protocol=protocol)
        dist.barrier()

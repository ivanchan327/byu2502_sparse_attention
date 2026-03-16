
import math
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, rotate_half
from transformers.processing_utils import Unpack
from transformers.utils import is_torch_npu_available, is_torch_xpu_available
from transformers.utils.import_utils import is_torch_greater_or_equal
from streaming_llm.utils import load, load_jsonl
import argparse
from dataclasses import dataclass
from datasets import load_dataset
import functools
from typing import Any, Dict, Optional
import os

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()


@dataclass
class TransformersKwargs:
    """Compatibility class for newer transformers versions"""
    task: Optional[str] = None
    tags: Optional[list] = None
    model: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    tokenizer: Optional[Dict[str, Any]] = None
    device: Optional[int] = None
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def use_gqa_in_sdpa(attention_mask: Optional[torch.Tensor], key: torch.Tensor) -> bool:
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

        attn_output, attn_weights = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    attn_output, attn_weights = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# -----------------------------
# Start-Recent KV cache slices
# -----------------------------
import torch
def slice2d(x, start, end):
    return x[:, :, start:end, ...]
def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]
def slice1d(x, start, end):
    return x[:, start:end, ...]
DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]


# -----------------------------
# Multi-layer schema cache (device-safe)
# -----------------------------
class MultiLayersCache():
    def __init__(
        self,
        granularity,
        num_granularity,
        locality,
        num_locality,
        cache_size=128,
        device=None
    ):
        # Avoid meta device at construction (Accelerate/HF may use it)
        if device is None or (isinstance(device, torch.device) and device.type == "meta"):
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.granularity = granularity  # rho
        self.num_granularity = num_granularity
        self.locality = locality        # nu
        self.num_locality = num_locality

        # Initialize on a real device (CPU by default)
        self.time_score_cache = [
            [torch.zeros((2, 0), device=self.device) for _ in range(self.num_locality)]
            for _ in range(self.num_granularity)
        ]
        self.key_value_cache = [
            [torch.zeros((2, 1, 4, 0, 64), device=self.device) for _ in range(self.num_locality)]
            for _ in range(self.num_granularity)
        ]
        self.time_score_buffer = [
            torch.zeros((2, 0), device=self.device) for _ in range(self.num_granularity)
        ]
        self.key_value_buffer = [
            torch.zeros((2, 1, 4, 0, 64), device=self.device) for _ in range(self.num_granularity)
        ]

        self.cache_size = cache_size
        self.cache_time = 0
        self.TIME, self.SCORE = 0, 1

    def time_mask(self, cache, time):
        index = torch.searchsorted(cache[self.TIME], time)
        mask = torch.zeros(cache[self.TIME].shape, dtype=torch.bool, device=cache.device)
        mask[index:] = True
        return mask

    def topk_mask(self, cache, topk, dim):
        topk_min = min(topk, cache[dim].shape[0])
        _, topk_index = torch.topk(cache[dim], topk_min, dim=0)
        mask = torch.zeros(cache[dim].shape, dtype=torch.bool, device=cache.device)
        mask[topk_index] = True
        return mask

    def append(self, key_states, value_states, attention_weights):
        key_value_states = torch.stack((key_states, value_states)).to(self.device)
        seq_len = attention_weights.shape[2]
        score_states = attention_weights.sum(2).sum(1).sum(0).to(self.device)
        time_states = torch.arange(self.cache_time, self.cache_time+seq_len, device=self.device)
        time_score_states = torch.stack((time_states, score_states))

        for i in range(self.num_granularity):
            self.time_score_buffer[i] = torch.cat(
                [self.time_score_buffer[i], time_score_states], dim=1
            )
            self.key_value_buffer[i] = torch.cat(
                [self.key_value_buffer[i], key_value_states], dim=3
            )
            buffer_len = self.key_value_buffer[i].shape[3]
            new_size = (buffer_len // self.granularity[i]) * self.granularity[i]
            time_score_states_append, self.time_score_buffer[i] = (
                self.time_score_buffer[i][:, :new_size], self.time_score_buffer[i][:, new_size:]
            )
            key_value_states_append, self.key_value_buffer[i] = (
                self.key_value_buffer[i][:, :, :, :new_size, :], self.key_value_buffer[i][:, :, :, new_size:, :]
            )
            time_score_states_compress = time_score_states_append.view(
                time_score_states_append.shape[0], -1, self.granularity[i]
            ).mean(dim=2)
            key_value_states_compress = key_value_states_append.view(
                *key_value_states_append.shape[0:3], -1, self.granularity[i], key_value_states_append.shape[4]
            ).mean(dim=4)

            self.time_score_cache[i][0] = torch.cat(
                [self.time_score_cache[i][0], time_score_states_compress.to(self.device)], dim=1
            )
            self.key_value_cache[i][0] = torch.cat(
                [self.key_value_cache[i][0], key_value_states_compress.to(self.device)], dim=3
            )
        self.cache_time += seq_len

    def update(self, mask, attn_weights):
        for i in range(self.num_granularity):
            for ii in range(self.num_locality):
                if mask[i][ii] is not None and mask[i][ii].sum().item() != 0:
                    lengths = [cache[self.SCORE].shape[0] for cache in self.time_score_cache[i][:ii+1]]
                    mask_splits = torch.split(mask[i][ii], lengths, dim=0)
                    attn_weights_partial, attn_weights = (
                        attn_weights[:, :, :, -1 * mask[i][ii].sum().item():],
                        attn_weights[:, :, :, :-1 * mask[i][ii].sum().item()]
                    )
                    attn_weights_splits = torch.split(
                        attn_weights_partial, [mask_split.sum().item() for mask_split in mask_splits], dim=3
                    )
                    for iii, (attn_weights_split, mask_split) in enumerate(zip(attn_weights_splits, mask_splits)):
                        device = self.time_score_cache[i][iii][1].device
                        self.time_score_cache[i][iii][1][mask_split.to(device)] += (
                            attn_weights_split.sum(2).sum(1).sum(0).to(device)
                        )

    def evict(self):
        for i in range(self.num_granularity):
            time_score_carry, key_value_carry = torch.tensor([]), torch.tensor([])
            for ii in range(self.num_locality):
                device = self.time_score_cache[i][ii].device
                self.time_score_cache[i][ii] = torch.cat(
                    [self.time_score_cache[i][ii], time_score_carry.to(device)], dim=1
                )
                self.key_value_cache[i][ii] = torch.cat(
                    [self.key_value_cache[i][ii], key_value_carry.to(self.key_value_cache[i][ii].device)], dim=3
                )
                if self.time_score_cache[i][ii].shape[1] <= self.cache_size:
                    continue
                if self.locality[ii] != 0:
                    mask = self.time_mask(self.time_score_cache[i][ii], (1-self.locality[ii]) * self.cache_time)
                    self.time_score_cache[i][ii], time_score_carry = (
                        self.time_score_cache[i][ii][:, mask], self.time_score_cache[i][ii][:, ~mask]
                    )
                    self.key_value_cache[i][ii], key_value_carry = (
                        self.key_value_cache[i][ii][:, :, :, mask], self.key_value_cache[i][ii][:, :, :, ~mask]
                    )
                    mask = self.topk_mask(self.time_score_cache[i][ii], self.cache_size, dim=self.SCORE)
                    self.time_score_cache[i][ii], _ = (
                        self.time_score_cache[i][ii][:, mask], self.time_score_cache[i][ii][:, ~mask]
                    )
                    self.key_value_cache[i][ii], _ = (
                        self.key_value_cache[i][ii][:, :, :, mask], self.key_value_cache[i][ii][:, :, :, ~mask]
                    )
                else:
                    mask = self.topk_mask(self.time_score_cache[i][ii], self.cache_size, dim=self.TIME)
                    self.time_score_cache[i][ii], time_score_carry = (
                        self.time_score_cache[i][ii][:, mask], self.time_score_cache[i][ii][:, ~mask]
                    )
                    self.key_value_cache[i][ii], key_value_carry = (
                        self.key_value_cache[i][ii][:, :, :, mask], self.key_value_cache[i][ii][:, :, :, ~mask]
                    )

    def partial_evict(self):
        for i in range(self.num_granularity):
            time_score_carry, key_value_carry = torch.tensor([]), torch.tensor([])
            for ii in range(self.num_locality):
                device = self.time_score_cache[i][ii].device
                self.time_score_cache[i][ii] = torch.cat(
                    [self.time_score_cache[i][ii], time_score_carry.to(device)], dim=1
                )
                self.key_value_cache[i][ii] = torch.cat(
                    [self.key_value_cache[i][ii], key_value_carry.to(self.key_value_cache[i][ii].device)], dim=3
                )
                if self.time_score_cache[i][ii].shape[1] <= self.cache_size:
                    continue
                if self.locality[ii] != 0:
                    mask = self.time_mask(self.time_score_cache[i][ii], (1-self.locality[ii]) * self.cache_time)
                    self.time_score_cache[i][ii], time_score_carry = (
                        self.time_score_cache[i][ii][:, mask], self.time_score_cache[i][ii][:, ~mask]
                    )
                    self.key_value_cache[i][ii], key_value_carry = (
                        self.key_value_cache[i][ii][:, :, :, mask], self.key_value_cache[i][ii][:, :, :, ~mask]
                    )
                else:
                    mask = self.topk_mask(self.time_score_cache[i][ii], self.cache_size, dim=self.TIME)
                    self.time_score_cache[i][ii], time_score_carry = (
                        self.time_score_cache[i][ii][:, mask], self.time_score_cache[i][ii][:, ~mask]
                    )
                    self.key_value_cache[i][ii], key_value_carry = (
                        self.key_value_cache[i][ii][:, :, :, mask], self.key_value_cache[i][ii][:, :, :, ~mask]
                    )

    def flush(self):
        self.time_score_buffer = [torch.zeros((2,0), device=self.device) for _ in range(self.num_granularity)]
        self.key_value_buffer = [torch.zeros((2,1,4,0,64), device=self.device) for _ in range(self.num_granularity)]
        for i in range(self.num_granularity):
            time_score_carry, key_value_carry = torch.tensor([]), torch.tensor([])
            for ii in range(self.num_locality):
                device = self.time_score_cache[i][ii].device
                self.time_score_cache[i][ii] = torch.cat(
                    [self.time_score_cache[i][ii], time_score_carry.to(device)], dim=1
                )
                self.key_value_cache[i][ii] = torch.cat(
                    [self.key_value_cache[i][ii], key_value_carry.to(self.key_value_cache[i][ii].device)], dim=3
                )
                if self.locality[ii] != 0:
                    if self.time_score_cache[i][ii].shape[1] <= self.cache_size:
                        continue
                    mask = self.time_mask(self.time_score_cache[i][ii], (1-self.locality[ii]) * self.cache_time)
                    self.time_score_cache[i][ii], time_score_carry = (
                        self.time_score_cache[i][ii][:, mask], self.time_score_cache[i][ii][:, ~mask]
                    )
                    self.key_value_cache[i][ii], key_value_carry = (
                        self.key_value_cache[i][ii][:, :, :, mask], self.key_value_cache[i][ii][:, :, :, ~mask]
                    )
                    mask = self.topk_mask(self.time_score_cache[i][ii], self.cache_size, dim=self.SCORE)
                    self.time_score_cache[i][ii], _ = (
                        self.time_score_cache[i][ii][:, mask], self.time_score_cache[i][ii][:, ~mask]
                    )
                    self.key_value_cache[i][ii], _ = (
                        self.key_value_cache[i][ii][:, :, :, mask], self.key_value_cache[i][ii][:, :, :, ~mask]
                    )
                else:
                    self.time_score_cache[i][ii], time_score_carry = torch.zeros((2,0), device=self.device), self.time_score_cache[i][ii]
                    self.key_value_cache[i][ii], key_value_carry = torch.zeros((2,1,4,0,64), device=self.device), self.key_value_cache[i][ii]

    def clear(self):
        self.time_score_cache = [
            [torch.zeros((2, 0), device=self.device) for _ in range(self.num_locality)]
            for _ in range(self.num_granularity)
        ]
        self.key_value_cache = [
            [torch.zeros((2, 1, 4, 0, 64), device=self.device) for _ in range(self.num_locality)]
            for _ in range(self.num_granularity)
        ]
        self.time_score_buffer = [torch.zeros((2, 0), device=self.device) for _ in range(self.num_granularity)]
        self.key_value_buffer = [torch.zeros((2, 1, 4, 0, 64), device=self.device) for _ in range(self.num_granularity)]
        self.cache_time = 0


# -----------------------------
# Adapter + attention with schema mixing (device-safe)
# -----------------------------
class QueryDependentAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2):
        super().__init__()
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adapter_down = nn.Linear(input_dim, hidden_dim, bias=False)
        self.adapter_up = nn.Linear(hidden_dim, output_dim1 * output_dim2, bias=False)
        self.adapter_act1 = nn.GELU()
        self.adapter_act2 = nn.GELU()
        self.adapter_dropout = nn.Dropout(p=0.1)

    def init_weights(self):
        nn.init.uniform_(self.adapter_down.weight, b=1e-3)
        nn.init.uniform_(self.adapter_up.weight, b=1e-3)

    def forward(self, query):
        x = query.mean(dim=(0, 1, 2))
        x = self.adapter_dropout(x)
        x = self.adapter_down(x)
        x = self.adapter_act1(x)
        x = self.adapter_up(x)
        x = self.adapter_act2(x)
        x = x.view(self.output_dim1, self.output_dim2)
        return x


class CacheLlamaAttention(LlamaAttention):
    def __init__(
        self, config, granularity, locality, prefill_base_weights, base_weights, cache_size,
        mix_alpha=1.0, mix_beta=0.0, normalize_rows=True, clamp_min=None, clamp_max=None, *args, **kwargs
    ):
        parent_kwargs = {}
        if "layer_idx" in kwargs:
            parent_kwargs["layer_idx"] = kwargs.pop("layer_idx")
        super().__init__(config, *args, **parent_kwargs)

        self.config = config
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.granularity = granularity
        self.locality = locality
        self.num_granularity = len(granularity)
        self.num_locality = len(locality)

        # Validate shapes (robust against any G/L)
        assert prefill_base_weights.shape == (self.num_granularity, self.num_locality), \
            f"prefill_base_weights must be [{self.num_granularity} x {self.num_locality}], got {tuple(prefill_base_weights.shape)}"
        assert base_weights.shape == (self.num_granularity, self.num_locality), \
            f"base_weights must be [{self.num_granularity} x {self.num_locality}], got {tuple(base_weights.shape)}"

        self.prefill_base_weights = prefill_base_weights
        self.base_weights = base_weights
        self.cache_size = cache_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        # Cache on a real device (avoid meta at __init__)
        try:
            param_dev = next(self.parameters()).device
        except StopIteration:
            param_dev = torch.device("cpu")
        cache_dev = torch.device("cpu") if param_dev.type == "meta" else param_dev
        self.cache = MultiLayersCache(
            self.granularity, self.num_granularity,
            self.locality, self.num_locality,
            self.cache_size, device=cache_dev
        )

        self.adapter = QueryDependentAdapter(
            input_dim=64, hidden_dim=64,
            output_dim1=self.num_granularity, output_dim2=self.num_locality
        )

        # Mixing knobs; mix_beta=0 keeps adapter neutral
        self.mix_alpha = mix_alpha
        self.mix_beta = mix_beta
        self.normalize_rows = normalize_rows
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _postprocess_weights(self, weights):
        if self.clamp_min is not None or self.clamp_max is not None:
            lo = self.clamp_min if self.clamp_min is not None else float("-inf")
            hi = self.clamp_max if self.clamp_max is not None else float("inf")
            weights = torch.clamp(weights, lo, hi)
        if self.normalize_rows:
            weights = torch.softmax(weights, dim=1)  # normalize across locality per granularity row
        return weights

    def get_prefill_schema_weights(self, query_states):
        base = self.prefill_base_weights
        adapt = self.adapter(query_states).to(base.device)
        weights = self.mix_alpha * base + self.mix_beta * adapt
        return self._postprocess_weights(weights)

    def get_schema_weights(self, query_states):
        base = self.base_weights
        adapt = self.adapter(query_states).to(base.device)
        weights = self.mix_alpha * base + self.mix_beta * adapt
        return self._postprocess_weights(weights)

    # ---- Schema KV selection (device-safe; boolean mask -> indices) ----
    def get_schema_key_value_states(self, granu, local):
        dev = next(self.parameters()).device
        key_value_states_select = torch.cat(
            [kv.to(dev) for kv in self.cache.key_value_cache[granu][:local+1]], dim=3
        )
        time_score_states_select = torch.cat(
            [ts.to(dev) for ts in self.cache.time_score_cache[granu][:local+1]], dim=1
        )
        mask = self.cache.topk_mask(time_score_states_select, self.cache.cache_size, dim=self.cache.SCORE).to(dev)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            empty = key_value_states_select[:, :, :, :0]
            return empty, mask
        topk_key_value_states = key_value_states_select[:, :, :, idx]
        return topk_key_value_states, mask

    def partial_get_schema_key_value_states(self, granu, local):
        dev = next(self.parameters()).device
        key_value_states_select = torch.cat(
            [kv.to(dev) for kv in self.cache.key_value_cache[granu][:local+1]], dim=3
        )
        time_score_states_select = torch.cat(
            [ts.to(dev) for ts in self.cache.time_score_cache[granu][:local+1]], dim=1
        )
        mask = self.cache.topk_mask(
            time_score_states_select, time_score_states_select.shape[1], dim=self.cache.SCORE
        ).to(dev)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            empty = key_value_states_select[:, :, :, :0]
            return empty, mask
        topk_key_value_states = key_value_states_select[:, :, :, idx]
        return topk_key_value_states, mask

    # ---- Training / Prefill ----
    def forward_training(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = torch.tensor(hidden_states.shape[:-1])
        hidden_shape = (*input_shape, -1, self.head_dim)

        self.cache.clear()

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_values.update(torch.zeros(key_states.shape), torch.zeros(value_states.shape), self.layer_idx, cache_kwargs)

        schema_attn_output, schema_attn_weights = sdpa_attention_forward(
            self,
            query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        with torch.no_grad():
            self.cache.append(key_states, value_states, schema_attn_weights[:, :, :, -1 * key_states.shape[2]:])
            self.cache.evict()

        schema_weights = self.get_prefill_schema_weights(query_states)
        schema_mask = [[None for _ in range(self.cache.num_locality)] for _ in range(self.cache.num_granularity)]

        for i in range(self.cache.num_granularity):
            for ii in range(self.cache.num_locality):
                if schema_weights[i][ii] != 0:
                    schema_key_value_states, schema_mask[i][ii] = self.get_schema_key_value_states(i, ii)
                    if schema_key_value_states.shape[3] == 0:
                        continue
                    schema_key_states  = schema_key_value_states[0, :, :, :].to(query_states.device, dtype=query_states.dtype)
                    schema_value_states = schema_key_value_states[1, :, :, :].to(query_states.device, dtype=query_states.dtype)
                    w = schema_weights[i][ii].to(schema_value_states.device)
                    schema_value_states = w * schema_value_states

                    key_states   = torch.cat([schema_key_states,  key_states],  dim=2)
                    value_states = torch.cat([schema_value_states, value_states], dim=2)

        schema_attn_output, schema_attn_weights = sdpa_attention_forward(
            self,
            query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = schema_attn_output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    # ---- Inference / Streaming ----
    def forward_inference(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        prefill = (input_shape[1] > 1)

        if prefill:
            self.cache.clear()

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_values.update(torch.zeros(key_states.shape), torch.zeros(value_states.shape), self.layer_idx, cache_kwargs)

        local_key_states, local_value_states = key_states, value_states

        schema_weights = self.get_schema_weights(query_states) if not prefill else self.get_prefill_schema_weights(query_states)
        schema_mask = [[None for _ in range(self.cache.num_locality)] for _ in range(self.cache.num_granularity)]

        for i in range(self.cache.num_granularity):
            for ii in range(self.cache.num_locality):
                if schema_weights[i][ii] != 0:
                    schema_key_value_states, schema_mask[i][ii] = self.get_schema_key_value_states(i, ii)
                    if schema_key_value_states.shape[3] == 0:
                        continue
                    schema_key_states  = schema_key_value_states[0, :, :, :].to(query_states.device, dtype=query_states.dtype)
                    schema_value_states = schema_key_value_states[1, :, :, :].to(query_states.device, dtype=query_states.dtype)
                    w = schema_weights[i][ii].to(schema_value_states.device)
                    schema_value_states = w * schema_value_states

                    key_states   = torch.cat([schema_key_states,  key_states],  dim=2)
                    value_states = torch.cat([schema_value_states, value_states], dim=2)

        schema_attn_output, schema_attn_weights = sdpa_attention_forward(
            self,
            query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = schema_attn_output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        with torch.no_grad():
            self.cache.update(schema_mask, schema_attn_weights[:, :, :, :-1 * local_key_states.shape[2]])
            self.cache.append(local_key_states, local_value_states, schema_attn_weights[:, :, :, -1 * local_key_states.shape[2]:])
            self.cache.evict()

        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inference = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not inference:
            return self.forward_training(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )
        else:
            return self.forward_inference(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )


# -----------------------------
# Alpaca dataset & training
# -----------------------------
class AlpacaDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        item = self.dataset[index]
        if 'input' in item and item['input']:
            text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        else:
            text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': encoding['input_ids'].copy()
        }


def train_cache_llama(args, model, tokenizer, device):
    dataset_name = "yahma/alpaca-cleaned"
    # dataset_name = "roneneldan/TinyStories"
    dataset = load_dataset(dataset_name)
    train_dataset = AlpacaDataset(dataset['train'], tokenizer)
    eval_dataset = AlpacaDataset(dataset['test'] if 'test' in dataset else dataset['train'].select(range(1000)), tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    training_args = TrainingArguments(
        output_dir="./cache-llama-alpaca",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Reduced for testing
        per_device_train_batch_size=1,  # Reduced for testing
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=100,  # Reduced for testing
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=5,
        eval_steps=50,
        save_steps=1000,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,  # Disabled for simplicity
        fp16=False,  # torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        # CRITICAL: Disable all integrations
        report_to="none",  # This disables all loggers
        disable_tqdm=False,  # Keep tqdm for progress bars
        max_grad_norm=1.0,
    )
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained("./cache-llama-alpaca")
    print("Training completed!")


def main_training(args, model, tokenizer, device):
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")
        else:
            print(f"Frozen: {name}")
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")
    train_cache_llama(args, model, tokenizer, device)


def main_inference(args, model, tokenizer, device):
    for name, param in model.named_parameters():
        param.requires_grad = False

    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        past_key_values = None

        outputs = model(input_ids=input_ids, past_key_values=past_key_values, inference=True)
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [pred_token_idx.item()]
        pos = 0
        max_gen_len = args.max_gen_len

        for _ in range(max_gen_len - 1):
            outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, inference=True)
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids.append(pred_token_idx.item())

            generated_text = (
                tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
            )
            now = len(generated_text) - 1
            if now > pos:
                print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                pos = now
            if pred_token_idx == tokenizer.eos_token_id:
                break

        print(" ".join(generated_text[pos:]), flush=True)


def main(args):
    transformers.models.llama.modeling_llama.LlamaAttention = \
        functools.partial(
            CacheLlamaAttention,
            granularity=args.granularity,
            locality=args.locality,
            prefill_base_weights=args.prefill_base_weights,
            base_weights=args.base_weights,
            cache_size=args.cache_size
        )
    model, tokenizer = load(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == 'cuda':
        print("Model is using GPU.")
    else:
        print("Model is using CPU.")

    for name, module in model.named_modules():
        if hasattr(module, '__class__') and 'CacheLlamaAttention' in module.__class__.__name__:
            print(f"Initializing adapter: {name}")
            module.adapter.init_weights()

    model.to(device)

    if args.train:
        main_training(args, model, tokenizer, device)
    else:
        main_inference(args, model, tokenizer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--granularity", type=int, default=[1])
    parser.add_argument("--locality", type=float, default=[0.0, 1.0])
    parser.add_argument("--cache_size", type=int, default=10000)
    parser.add_argument("--prefill_base_weights", default = torch.tensor(
        [[1, 0]],
        dtype=torch.float32
    ))
    parser.add_argument("--base_weights", default = torch.tensor(
        [[1, 0]],
        dtype=torch.float32
    ))
    parser.add_argument("--max_gen_len", type=int, default=1000)
    parser.add_argument("--data_root", type=str, default="data/")
    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="data/alpaca_data", help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="The output directory where the model will be saved")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Floating point precision")
    # Training arguments
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--init_adapter", type=bool, default=True)
    parser.add_argument("--init_cache", type=bool, default=True)
    args = parser.parse_args()
    main(args)

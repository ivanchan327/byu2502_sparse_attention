import math
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, rotate_half
from streaming_llm.utils import load, load_jsonl

import argparse
import functools
import os, inspect
from typing import Optional, Tuple

from collections.abc import Callable
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from transformers.utils import is_torch_npu_available, is_torch_xpu_available, logging
from transformers.utils.import_utils import is_torch_greater_or_equal


def forward(
        self, 
        hidden_states: torch.Tensor,  # (batch_size, sequence_length, hidden_size)
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        *args, **kwargs
    ):

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (batch_size, num_attention_heads, seq_len, head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (batch_size, num_key_value_heads, seq_len, head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # for i in range(self.cache.num_granularity):
        #     self.cache.key_value_cache[i][0] = torch.stack((key_states, value_states))
        #     self.cache.time_score_cache[i][0] = torch.Tensor([[0, 0]] * 38)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_states(key_states, self.num_key_value_groups)
        value_states = repeat_states(value_states, self.num_key_value_groups)

        local_attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        local_attn_weights = nn.functional.softmax(local_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        local_attn_output = torch.matmul(local_attn_weights, value_states)

        # attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # local_attn_output, local_attn_weights = attention_interface(
        #     self, query_states, key_states, value_states,
        #     attention_mask, dropout=0.0 if not self.training else self.attention_dropout,
        #     scaling=self.scaling,
        #     **kwargs,
        # )

        print("-", end='', flush=True)
        
        schema_weights = self.get_schema_weights(query_states, local_attn_output)
        attn_output = local_attn_output
        
        if self.cache.cache_time > 0:
            for i in range(self.cache.num_granularity):
                for ii in range(self.cache.num_locality):
                    schema_key_value_states = self.get_schema_key_value_states(i, ii)
                    schema_key_states, schema_value_states = schema_key_value_states[0,:,:,:], schema_key_value_states[1,:,:,:]

                    schema_key_states = repeat_states(schema_key_states, self.num_key_value_groups)
                    schema_value_states = repeat_states(schema_value_states, self.num_key_value_groups)

                    schema_attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                    schema_attn_weights = nn.functional.softmax(schema_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                    schema_attn_output = torch.matmul(schema_attn_weights, schema_value_states)
                    attn_output += schema_weights[i][ii] * schema_attn_output

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        # attn_output = local_attn_output.transpose(1, 2).reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        if local_attn_weights != None:
            print("=", end='')
            self.cache.update(key_states, value_states, local_attn_weights)
            self.cache.evict()
        
        return attn_output, local_attn_weights



    def forward():
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

def use_gqa_in_sdpa(attention_mask: Optional[torch.Tensor], key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda or Ascend NPU
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    # 2.xpu
    #   - torch version >= 2.8
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def repeat_states(states, num_repeat):
    batch_size, num_key_value_heads, seq_len, head_dim = states.shape
    if num_repeat == 1:
        return states
    states = states[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_repeat, seq_len, head_dim)
    return states.reshape(batch_size, num_key_value_heads * num_repeat, seq_len, head_dim)


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
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

    # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
    # - Not in decoding phase (otherwise we want full attention on the single query token)
    # - Attention mask is not to be provided (even if it is a causal pattern)
    # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
    #
    # Quirks on the conditionals:
    # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
    #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
    # - It is important to check first for the shape, otherwise compile will fail with
    #   `argument 'is_causal' must be bool, not SymBool`.
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

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
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

    # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
    # - Not in decoding phase (otherwise we want full attention on the single query token)
    # - Attention mask is not to be provided (even if it is a causal pattern)
    # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
    #
    # Quirks on the conditionals:
    # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
    #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
    # - It is important to check first for the shape, otherwise compile will fail with
    #   `argument 'is_causal' must be bool, not SymBool`.
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
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
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
import types

__all__ = ["enable_llama_pos_shift_attention"]


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def llama_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # HF uses tuple of tuples
    cache_position: Optional[torch.LongTensor] = None,  # Added for compatibility
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs  # Future-proof
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Custom LLaMA attention forward with position shift and KV cache handling.
    """

    # Silence Pylance warnings for unused args
    _ = cache_position
    _ = kwargs

    bsz, q_len, _ = hidden_states.size()

    # Project QKV
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape to [batch, heads, seq_len, head_dim]
    query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Compute sequence length including past if available
    kv_seq_len = key_states.shape[-2]
    if past_key_values and len(past_key_values) > 0 and past_key_values[0] and past_key_values[0][0] is not None:
        kv_seq_len += past_key_values[0][0].shape[-2]

    # Rotary embeddings
    
    # Hugging Face passes cos and sin in kwargs or position_embeddings
    cos = kwargs.get("cos", None)
    sin = kwargs.get("sin", None)

    if cos is None or sin is None:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        rotary_emb = LlamaRotaryEmbedding(self.config)
        position_ids = torch.arange(kv_seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = rotary_emb(hidden_states, position_ids)


    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)


    # Append past key/value if available
    if past_key_values and len(past_key_values) > 0 and past_key_values[0] and past_key_values[0][0] is not None:
        key_states = torch.cat([past_key_values[0][0], key_states], dim=2)
        value_states = torch.cat([past_key_values[0][1], value_states], dim=2)

    # Update past_key_values for caching
    past_key_values = (key_states, value_states) if use_cache else None

    # Apply rotary to keys with updated positions
    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
    key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

    # Repeat KV heads if needed
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Attention weights
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Softmax in fp32 for stability
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.config.hidden_size)

    # Output projection
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, past_key_values


def enable_llama_pos_shift_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_pos_shift_attention(
                module,
            )

        
        if isinstance(module, LlamaAttention):
            # Patch forward
            module.forward = types.MethodType(llama_pos_shift_attention_forward, module)


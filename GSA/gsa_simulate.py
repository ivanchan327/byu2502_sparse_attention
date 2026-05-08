# region Imports
import argparse
import functools
import json
import math
import os
import random
import re

from datasets import load_dataset
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.models.llama.modeling_llama import LlamaAttention

from gsa_dataset import *
from gsa_gpu import *
from gsa_transformer import *
from streaming_llm.utils import load
# endregion

class VanillaCache():
    def __init__(self, granu, local, cache_size, vector_size, dtype, device):
        self.granu, self.local = granu, local
        self.num_granu, self.num_local = len(self.granu), len(self.local)
        self.cache_size = cache_size
        self.vector_size = vector_size
        self.dtype, self.device = dtype, device
        assert self.device == torch.device("cuda")

        self.kv_cache = [[torch.zeros([2, *self.vector_size[0:2], 0, self.vector_size[2]], dtype=self.dtype, device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.ts_cache = [[torch.zeros([2, 0], dtype=self.dtype, device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.kv_buffer = [torch.zeros([2, *self.vector_size[0:2], 0, self.vector_size[2]], dtype=self.dtype, device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros([2, 0], dtype=self.dtype, device=self.device) for _ in range(self.num_granu)]

        self.cache_time = 0
        self.TIME, self.SCORE = 0, 1

    def time_mask(self, cache, time):
        idx = torch.searchsorted(cache[self.TIME], time)
        mask = torch.zeros(cache[self.TIME].shape, dtype=torch.bool, device=cache.device)
        mask[idx:] = True
        return mask
    
    def topk_mask(self, cache, topk, dim):
        topk = min(topk, cache.shape[1])
        _, topk_idx = torch.topk(cache[dim], topk, dim=0)
        mask = torch.zeros(cache[dim].shape, dtype=torch.bool, device=cache.device)
        mask[topk_idx] = True
        return mask
    
    def retrieval(self, granu, local):
        kv_select = self.kv_cache[granu][local]
        return kv_select, None
    
    def append(self, key_states, value_states, attn_weights):
        seq_len = attn_weights.shape[3]
        kv_states = torch.stack((key_states, value_states))
        time_states = torch.arange(self.cache_time, self.cache_time + seq_len, device=self.device)
        score_states = attn_weights.sum(dim=(0, 1, 2))
        ts_states = torch.stack((time_states, score_states))

        for i in range(self.num_granu):
            self.kv_buffer[i] = torch.cat([self.kv_buffer[i], kv_states], dim=3)
            self.ts_buffer[i] = torch.cat([self.ts_buffer[i], ts_states], dim=1)
            buffer_len = self.ts_buffer[i].shape[1]

            if buffer_len >= self.granu[i]:
                app_len = buffer_len // self.granu[i]
                use_len = app_len * self.granu[i]

                kv_append, self.kv_buffer[i] = self.kv_buffer[i][..., :use_len, :], self.kv_buffer[i][..., use_len:, :]
                ts_append, self.ts_buffer[i] = (self.ts_buffer[i][:, :use_len], self.ts_buffer[i][:, use_len:])
                kv_compress = kv_append.view(2, *self.vector_size[0:2], -1, self.granu[i], self.vector_size[2]).mean(dim=4)
                ts_compress = ts_append.view(2, -1, self.granu[i]).mean(dim=2)

                for ii in range(self.num_local):
                    self.kv_cache[i][ii] = torch.cat([self.kv_cache[i][ii], kv_compress], dim=3)
                    self.ts_cache[i][ii] = torch.cat([self.ts_cache[i][ii], ts_compress], dim=1)

        self.cache_time += seq_len
        
    def update(self, mask_list, attn_weights):
        attn_scores = attn_weights.sum(dim=(0, 1, 2))
        for i in range(self.num_granu):
            for ii in range(self.num_local):
                cache_len = self.ts_cache[i][ii].shape[1]
                if cache_len == 0: continue
                attach = cache_len
                attn_scores_partial, attn_scores = attn_scores[-1 * attach:], attn_scores[:-1 * attach]
                self.ts_cache[i][ii][self.SCORE, :cache_len] += attn_scores_partial

    def evict(self):
        for i in range(self.num_granu):
            for ii in range(self.num_local):
                if self.ts_cache[i][ii].shape[1] <= self.cache_size:
                    continue
                if self.local[ii] != 0:
                    mask = self.time_mask(self.ts_cache[i][ii], (1-self.local[ii]) * self.cache_time)
                    self.kv_cache[i][ii], _ = (self.kv_cache[i][ii][..., mask, :], self.kv_cache[i][ii][..., ~mask, :])
                    self.ts_cache[i][ii], _ = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size, dim=self.SCORE)                  
                    self.kv_cache[i][ii], _ = (self.kv_cache[i][ii][..., mask, :], self.kv_cache[i][ii][..., ~mask, :])
                    self.ts_cache[i][ii], _ = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                else:
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size, dim=self.TIME) 
                    self.kv_cache[i][ii], _ = (self.kv_cache[i][ii][..., mask, :], self.kv_cache[i][ii][..., ~mask, :])
                    self.ts_cache[i][ii], _ = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])

    def evict_internal(self):
        # This is the correct implementation
        pass

    def clear(self):
        self.kv_cache = [[torch.zeros([2, *self.vector_size[0:2], 0, self.vector_size[2]], dtype=self.dtype, device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.ts_cache = [[torch.zeros([2, 0], dtype=self.dtype, device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.kv_buffer = [torch.zeros([2, *self.vector_size[0:2], 0, self.vector_size[2]], dtype=self.dtype, device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros([2, 0], dtype=self.dtype, device=self.device) for _ in range(self.num_granu)]
        self.cache_time = 0

class ResidualCachewithBlock():
    def __init__(self, granu, local, cache_size, vector_size, dtype, device):
        self.granu, self.local = granu, local 
        self.num_granu, self.num_local = len(self.granu), len(self.local)
        self.cache_size, self.max_size = cache_size, cache_size + 7000   # Chnage when Prompt Length INCREASE
        self.vector_size = vector_size
        self.dtype, self.device = dtype, device
        assert self.device == torch.device("cuda")

        self.kv_cache = [[torch.zeros([2, *self.vector_size[0:2], self.max_size // self.granu[i], self.granu[i], self.vector_size[2]], dtype=self.dtype, device=self.device) 
                          for _ in range(self.num_local)] for i in range(self.num_granu)]
        self.ts_cache = [[torch.zeros((2, self.max_size // self.granu[i]), dtype=self.dtype, device=self.device) for _ in range(self.num_local)] for i in range(self.num_granu)]
        self.cache_len = [[0 for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.kv_buffer = [torch.zeros([2, *self.vector_size[0:2], self.max_size, self.vector_size[2]], dtype=self.dtype, device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros((2, self.max_size), dtype=self.dtype, device=self.device) for _ in range(self.num_granu)]
        self.buffer_len = [0 for _ in range(self.num_granu)]

        self.cache_time = 0
        self.TIME, self.SCORE = 0, 1

    def time_mask(self, cache, time):
        idx = torch.searchsorted(cache[self.TIME], time)
        mask = torch.zeros(cache[self.TIME].shape, dtype=torch.bool, device=cache.device)
        mask[idx:] = True
        return mask
    
    def topk_mask(self, cache, topk, dim):
        topk = min(topk, cache.shape[1])
        _, topk_idx = torch.topk(cache[dim], topk, dim=0)
        mask = torch.zeros(cache[dim].shape, dtype=torch.bool, device=cache.device)
        mask[topk_idx] = True
        return mask
    
    def retrieval(self, granu, local):
        len = self.cache_len[granu][local]
        kv_select = self.kv_cache[granu][local][..., :len, :, :]
        kv_select = kv_select.reshape(2, *self.vector_size[0:2], -1, self.vector_size[2])
        return kv_select, None
    
    def append(self, key_states, value_states, attn_weights):
        seq_len = attn_weights.shape[3]
        time_states = torch.arange(self.cache_time, self.cache_time + seq_len, device=self.device)
        score_states = attn_weights.sum(dim=(0, 1, 2))

        for i in range(self.num_granu):
            buffer_idx = self.buffer_len[i]
            self.kv_buffer[i][0, ..., buffer_idx:buffer_idx + seq_len, :] = key_states
            self.kv_buffer[i][1, ..., buffer_idx:buffer_idx + seq_len, :] = value_states
            self.ts_buffer[i][0, buffer_idx:buffer_idx + seq_len] = time_states
            self.ts_buffer[i][0, buffer_idx:buffer_idx + seq_len] = score_states
            self.buffer_len[i] += seq_len

            if self.buffer_len[i] >= self.granu[i]:
                cache_len = self.cache_len[i][0]
                app_len = self.buffer_len[i] // self.granu[i]
                use_len = app_len * self.granu[i]
                rem_len = self.buffer_len[i] - use_len

                kv_append = self.kv_buffer[i][..., :use_len, :].view(2, *self.vector_size[0:2], -1, self.granu[i], self.vector_size[2])
                ts_append = self.ts_buffer[i][:, :use_len].view(2, -1, self.granu[i]).mean(dim=2)
                self.kv_cache[i][0][..., cache_len:cache_len + app_len, :, :] = kv_append
                self.ts_cache[i][0][:, cache_len:cache_len + app_len] = ts_append
                self.cache_len[i][0] += app_len
                self.kv_buffer[i][..., :rem_len, :] = self.kv_buffer[i][..., use_len:self.buffer_len[i], :]
                self.ts_buffer[i][:, :rem_len] = self.ts_buffer[i][:, use_len:self.buffer_len[i]]
                self.buffer_len[i] = rem_len

        self.cache_time += seq_len

    def update(self, mask_list, attn_weights):
        attn_scores = attn_weights.sum(dim=(0, 1, 2))
        for i in range(self.num_granu):
            for ii in range(self.num_local):
                cache_len = self.cache_len[i][ii]
                if cache_len == 0: continue
                attach = cache_len * self.granu[i]
                attn_scores_partial, attn_scores = attn_scores[-1 * attach:], attn_scores[:-1 * attach]
                self.ts_cache[i][ii][self.SCORE, :cache_len] += attn_scores_partial.view(-1, self.granu[i]).mean(dim=1)


    def evict(self):
        for i in range(self.num_granu):
            kv_carry, ts_carry = None, None
            for ii in range(self.num_local):
                if kv_carry is not None:
                    cache_len = self.cache_len[i][ii]
                    carry_len = kv_carry.shape[3]
                    self.kv_cache[i][ii][..., cache_len:cache_len + carry_len, :, :] = kv_carry
                    self.ts_cache[i][ii][:, cache_len:cache_len + carry_len] = ts_carry
                    self.cache_len[i][ii] += carry_len
                if self.cache_len[i][ii] <= self.cache_size // self.granu[i]:
                    kv_carry, ts_carry = None, None
                    continue

                if self.local[ii] != 0:
                    cache_len = self.cache_len[i][ii]
                    kv_full = self.kv_cache[i][ii][..., :cache_len, :, :]
                    ts_full = self.ts_cache[i][ii][:, :cache_len]
                    mask = self.time_mask(ts_full, (1 - self.local[ii]) * self.cache_time)
                    pend_len = torch.count_nonzero(mask)
                    self.kv_cache[i][ii][..., :pend_len, :, :], kv_carry = kv_full[..., mask, :, :], kv_full[..., ~mask, :, :]
                    self.ts_cache[i][ii][:, :pend_len], ts_carry = ts_full[:, mask], ts_full[:, ~mask]
                    self.cache_len[i][ii] = pend_len

                    cache_len = self.cache_len[i][ii]
                    kv_full = self.kv_cache[i][ii][..., :cache_len, :, :]
                    ts_full = self.ts_cache[i][ii][:, :cache_len]
                    mask = self.topk_mask(ts_full, self.cache_size // self.granu[i], dim=self.SCORE)
                    pend_len = torch.count_nonzero(mask)
                    self.kv_cache[i][ii][..., :pend_len, :, :] = kv_full[..., mask, :, :]
                    self.ts_cache[i][ii][:, :pend_len] = ts_full[:, mask]
                    self.cache_len[i][ii] = pend_len
                else:
                    cache_len = self.cache_len[i][ii]
                    kv_full = self.kv_cache[i][ii][..., :cache_len, :, :]
                    ts_full = self.ts_cache[i][ii][:, :cache_len]
                    mask = self.topk_mask(ts_full, self.cache_size // self.granu[i], dim=self.TIME)
                    pend_len = torch.count_nonzero(mask)
                    self.kv_cache[i][ii][..., :pend_len, :, :], kv_carry = kv_full[..., mask, :, :], kv_full[..., ~mask, :, :]
                    self.ts_cache[i][ii][:, :pend_len], ts_carry = ts_full[:, mask], ts_full[:, ~mask]
                    self.cache_len[i][ii] = pend_len

    def evict_internal(self):
        for i in range(self.num_granu):
            kv_carry, ts_carry = None, None
            for ii in range(self.num_local):
                if kv_carry is not None:
                    cache_len = self.cache_len[i][ii]
                    carry_len = kv_carry.shape[3]
                    self.kv_cache[i][ii][..., cache_len:cache_len + carry_len, :, :] = kv_carry
                    self.ts_cache[i][ii][:, cache_len:cache_len + carry_len] = ts_carry
                    self.cache_len[i][ii] += carry_len
                if self.cache_len[i][ii] <= self.cache_size // self.granu[i]:
                    kv_carry, ts_carry = None, None
                    continue

                if self.local[ii] != 0:
                    cache_len = self.cache_len[i][ii]
                    kv_full = self.kv_cache[i][ii][..., :cache_len, :, :]
                    ts_full = self.ts_cache[i][ii][:, :cache_len]
                    mask = self.time_mask(ts_full, (1 - self.local[ii]) * self.cache_time)
                    pend_len = torch.count_nonzero(mask)
                    self.kv_cache[i][ii][..., :pend_len, :, :], kv_carry = kv_full[..., mask, :, :], kv_full[..., ~mask, :, :]
                    self.ts_cache[i][ii][:, :pend_len], ts_carry = ts_full[:, mask], ts_full[:, ~mask]
                    self.cache_len[i][ii] = pend_len

                else:
                    cache_len = self.cache_len[i][ii]
                    kv_full = self.kv_cache[i][ii][..., :cache_len, :, :]
                    ts_full = self.ts_cache[i][ii][:, :cache_len]
                    mask = self.topk_mask(ts_full, self.cache_size // self.granu[i], dim=self.TIME)
                    pend_len = torch.count_nonzero(mask)
                    self.kv_cache[i][ii][..., :pend_len, :, :], kv_carry = kv_full[..., mask, :, :], kv_full[..., ~mask, :, :]
                    self.ts_cache[i][ii][:, :pend_len], ts_carry = ts_full[:, mask], ts_full[:, ~mask]
                    self.cache_len[i][ii] = pend_len

    def clear(self):
        self.kv_cache = [[torch.zeros([2, *self.vector_size[0:2], self.max_size // self.granu[i], self.granu[i], self.vector_size[2]], dtype=self.dtype, device=self.device) 
                          for _ in range(self.num_local)] for i in range(self.num_granu)]
        self.ts_cache = [[torch.zeros((2, self.max_size // self.granu[i]), dtype=self.dtype, device=self.device) for _ in range(self.num_local)] for i in range(self.num_granu)]
        self.cache_len = [[0 for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.kv_buffer = [torch.zeros([2, *self.vector_size[0:2], self.max_size, self.vector_size[2]], dtype=self.dtype, device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros((2, self.max_size), dtype=self.dtype, device=self.device) for _ in range(self.num_granu)]
        self.buffer_len = [0 for _ in range(self.num_granu)]
        self.cache_time = 0


class GSA1_Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.device = torch.device("cuda")
        
        self.adapter_down = nn.Linear(input_dim, hidden_dim, bias=False, dtype=dtype)
        self.adapter_act1 = nn.GELU()
        self.adapter_up = nn.Linear(hidden_dim, output_dim, bias=False, dtype=dtype)
        self.adapter_act2 = nn.GELU()
        self.adapter_dropout = nn.Dropout(p=0.1)
        self.scale = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.to_empty(device=self.device)

    def init_weights(self, para):
        nn.init.uniform_(self.adapter_down.weight, b=para)
        nn.init.uniform_(self.adapter_up.weight, b=para)

    def forward(self, x):
        x = x[:, :, -1, :].mean(dim=1)
        x = self.adapter_dropout(x)
        x = self.adapter_down(x)
        x = self.adapter_act1(x)
        x = self.adapter_up(x)
        x = self.adapter_act2(x)
        return x
    
class GSA_LlamaAttention(LlamaAttention):
    def __init__(
        self, config, granu, local, base_weights, cache_type, cache_size, vector_size, dtype,
        *args, **kwargs
    ):
        parent_kwargs = {}
        if "layer_idx" in kwargs:
            parent_kwargs["layer_idx"] = kwargs.pop("layer_idx")
        super().__init__(config, *args, **parent_kwargs)

        self.config = config
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        self.dtype, self.device = dtype, torch.device("cuda")
        self.granu, self.local = granu, local
        self.num_granu, self.num_local = len(self.granu), len(self.local)
        self.base_weights = base_weights.to(self.device)
        self.cache_size, self.vector_size = cache_size, vector_size   

        if cache_type == "Vanilla":
            self.cache = VanillaCache(self.granu, self.local, self.cache_size, self.vector_size, self.dtype, self.device)
            self.evict_cycle = 1
        elif cache_type == "ResidualwithBlock":
            self.cache = ResidualCachewithBlock(self.granu, self.local, self.cache_size, self.vector_size, self.dtype, self.device)
            self.evict_cycle = int(math.log2(self.cache_size))
        else:
            raise ValueError(f"Unregistered Cache Type: {cache_type}.")
        # self.evict_cycle = 1
        

class GSA1_LlamaAttention(GSA_LlamaAttention):
    def __init__(
        self, config, granu, local, base_weights, cache_type, cache_size, vector_size, dtype, 
        *args, **kwargs
    ):
        super().__init__(config, granu, local, base_weights, cache_type, cache_size, vector_size, dtype, *args, **kwargs)
        self.adapter = GSA1_Adapter(
            input_dim=vector_size[2], hidden_dim=vector_size[2], output_dim=self.num_granu * self.num_local,
            dtype=dtype, layer_idx=self.layer_idx
        )

    def adapted_weights(self, query_states):
        base = self.base_weights.flatten()
        adapt = self.adapter(query_states)
        weights = base + adapt
        weights = F.softmax(weights, dim=1).view(-1, self.num_granu, self.num_local) * (self.num_granu * self.num_local)
        return weights
    
    def forward(
        self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, cache_position=None, **kwargs
    ):
        input_shape = torch.tensor(hidden_states.shape[:-1])
        hidden_shape = (*input_shape, -1, self.head_dim)
        prefill = (input_shape[1] > 1)
        if prefill: self.cache.clear()

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_values.update(torch.zeros(key_states.shape), torch.zeros(value_states.shape), self.layer_idx, cache_kwargs)

        if self.training:
            _, attn_weights = sdpa_attention_forward(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )
            with torch.no_grad():   
                self.cache.append(key_states, value_states, attn_weights)
                self.cache.evict_internal()

        adapted_weights = self.adapted_weights(query_states)
        all_keys, all_values, all_masks = [], [], []

        for i in range(self.num_granu):
            for ii in range(self.num_local):
                kv_states, mask = self.cache.retrieval(i, ii)
                weights = adapted_weights[:, i, ii].view(-1, 1, 1, 1)
                all_keys.append(kv_states[0])
                all_values.append(kv_states[1] * weights)
                all_masks.append(mask)

        all_keys.reverse(); all_values.reverse()
        key_concat = torch.cat(all_keys + [key_states], dim=2)
        value_concat = torch.cat(all_values + [value_states], dim=2)
        attn_output, attn_weights = sdpa_attention_forward(
            self, query_states, key_concat, value_concat, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        if not self.training:
            with torch.no_grad():
                mask_list = [[all_masks[i * self.cache.num_local + ii] 
                            for ii in range(self.cache.num_local)] for i in range(self.cache.num_granu)]
                split_idx = key_concat.shape[2] - key_states.shape[2]
                self.cache.update(mask_list, attn_weights[..., :split_idx])
                self.cache.append(key_states, value_states, attn_weights[..., split_idx:])
                # if prefill or self.cache.cache_time % self.evict_cycle == 0:
                if prefill or (self.cache.cache_time + self.layer_idx) % self.evict_cycle == 0:
                    self.cache.evict()
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None
    

def main_train(args, model, tokenizer, device):
    for name, param in model.named_parameters():
        param.requires_grad = ("adapter" in name)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")
    
    config_dict = vars(args)
    output_dir_path = f"./{args.model_abbrev}/{args.model_type}_{args.cache_type}_{args.cache_size}/Ver_{args.version}"
    os.makedirs(output_dir_path, exist_ok=True)
    config_file = f"{output_dir_path}/config.json"
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Configuration saved to: {config_file}.")

    dataset_name = "yahma/alpaca-cleaned"
    dataset = load_dataset(dataset_name)
    train_dataset_raw = dataset['train'].select(range(min(10000, len(dataset['train']))))
    train_dataset = AlpacaDataset(train_dataset_raw, tokenizer)
    if 'validation' in dataset:
        eval_dataset_raw = dataset['validation'].select(range(min(500, len(dataset['validation']))))
    else:
        eval_dataset_raw = dataset['train'].select(range(10000, min(10500, len(dataset['train']))))
    eval_dataset = AlpacaDataset(eval_dataset_raw, tokenizer)
    eval_dataset = AlpacaDataset(eval_dataset_raw, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_callback = GenerationMonitorCallback(
        tokenizer=tokenizer, raw_dataset=eval_dataset_raw, device=device, log_steps=500
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8 if args.train_batch_size > 1 else None, return_tensors="pt"
    )

    is_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        # --- 1. Paths & State ---
        output_dir=output_dir_path,
        overwrite_output_dir=True,
        resume_from_checkpoint=args.checkpoint,
        seed=args.seed,        
        # --- 2. Hardware & Batching ---
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=is_cuda, # Automatically enable if GPU is present
        fp16=False, 
        dataloader_pin_memory=is_cuda, # Speed up data transfer to GPU
        # --- 3. Optimization & Schedule ---
        num_train_epochs=1,
        learning_rate=args.learning_rate,
        warmup_steps=1000,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # --- 4. Strategy & Frequency ---
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=1000,
        logging_steps=10,
        load_best_model_at_end=False,
        # --- 5. System & Logging ---
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False,
        ignore_data_skip=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[generation_callback],
    )

    if is_cuda:
        torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=args.checkpoint)
    trainer.save_model()
    print("Training completed!")

def main_eval(args, model, tokenizer, device):
    for name, param in model.named_parameters():
        param.requires_grad = False

    dataset = load_dataset("tatsu-lab/alpaca")
    list_data = dataset['train'] 

    prompts = []
    for item in list_data:
        instruction = item['instruction']
        input_text = item['input']
        prompt = f"{instruction}\n\n{input_text}" if input_text and len(input_text.strip()) > 0 else instruction
        prompts.append(prompt)
    
    for idx, prompt in enumerate(prompts):      
        print(f"\n=== Sample {idx + 1} ===")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print("\n" + "="*50)
        print("Generating response (streaming)...")
        print("="*50)

        response, _ = generate_response(
            model, tokenizer, prompt, 
            device, args.max_gen_len,
            stream_output=True 
        )
        
        ground_truth = list_data[idx]['output']
        print(f"\n{'='*50}")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'='*50}")

        if args.pause_between:
            input("\nPress Enter to continue to next sample...")

def main(args):
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    args.dtype = dtype_map[args.dtype]

    if args.model_type == "Llama":
        pass
    elif args.model_type == "GSA1":
        transformers.models.llama.modeling_llama.LlamaAttention = \
            functools.partial(
                GSA1_LlamaAttention,
                granu=args.granu, local=args.local,
                base_weights=torch.tensor(args.base_weights, dtype=args.dtype),
                cache_type=args.cache_type, cache_size=args.cache_size,
                vector_size=args.vector_size, attn_size=args.attn_size,
                dtype=args.dtype
            )
    else:
        raise ValueError("Unregistered Model Type")
    
    if args.checkpoint is not None:
        print(f"Loading Weights from {args.checkpoint}.")
        checkpoint_path = args.checkpoint
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True, torch_dtype=args.dtype)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
    else:
        print(f"No checkpoint is loaded. Initializing adapter.")
        model, tokenizer = load(args.model_name_or_path)
        for name, module in model.named_modules():
            if hasattr(module, '__class__') and re.search(r'GSA[12]_LlamaAttention', module.__class__.__name__):
                module.adapter.init_weights(args.adapter_init_weights)
    print(f"Initializing cache.")
    for name, module in model.named_modules():
        if hasattr(module, '__class__') and re.search(r'GSA[12]_LlamaAttention', module.__class__.__name__):
            module.cache.clear()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model is using GPU." if device.type == 'cuda' else "Model is using CPU.")

    if tokenizer.chat_template is None:
        tokenizer.chat_template = tokenizer.chat_template = (
            "{% for message in messages %}\n"
            "{% if message['role'] == 'user' %}\n"
            "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}\n"
            "{% elif message['role'] == 'system' %}\n"
            "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}\n"
            "{% elif message['role'] == 'assistant' %}\n"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}\n"
            "{% endif %}\n"
            "{% if loop.last and add_generation_prompt %}\n"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}\n"
            "{% endif %}\n"
            "{% endfor %}"
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    if args.train: main_train(args, model, tokenizer, device)
    else: main_eval(args, model, tokenizer, device)

def fix_seed(seed):
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using Seed: {seed}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model arguments
    # parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B")
    # parser.add_argument("--model_abbrev", type=str, default="Llama3B")
    # parser.add_argument("--vector_size", default=(1, 8, 128))
    # parser.add_argument("--attn_size", type=int, default=2048)
    parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--model_abbrev", type=str, default="TinyLlama_fp32")
    parser.add_argument("--vector_size", default=(1, 4, 64))
    parser.add_argument("--attn_size", type=int, default=2048)
    
    # GSA arguments
    # parser.add_argument("--model_type", type=str, default="Llama")
    parser.add_argument("--model_type", type=str, default="GSA1")
    parser.add_argument("--cache_type", type=str, default="ResidualwithBlock")
    parser.add_argument("--cache_size", type=int, default=64)
    parser.add_argument("--granu", type=int, default=[1, 16])
    parser.add_argument("--local", type=float, default=[0.0, 1.0])
    parser.add_argument("--base_weights", default=[[0]])
    parser.add_argument("--adapter_init_weights", type=float, default=1e-6)
    
    # Training / Inference arguments
    parser.add_argument("--train", type=bool, default=True)
    # parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="TinyLlama_fp32/GSA1_ResidualwithBlock_64/Ver_2/checkpoint-4000")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Other arguments
    parser.add_argument("--dataset_path", type=str, default="data/alpaca_data", help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="The output directory where the model will be saved")
    parser.add_argument("--dtype", type=str, default="float32", help="Floating point precision")
    parser.add_argument("--max_gen_len", type=int, default=1000)
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--pause_between", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=random.randint(0, 10000))
    
    args = parser.parse_args()
    fix_seed(args.seed)
    main(args)
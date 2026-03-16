# region External Import
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, GenerationConfig, TrainingArguments, Trainer
from transformers import set_seed
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, rotate_half
from transformers.processing_utils import Unpack

import argparse
from datasets import load_dataset
import functools
import json
from streaming_llm.utils import load, load_jsonl
from typing import Any, Dict, Optional
import os
import random
import re
# endregion

# region Internal Import
from gsa_transformer import *
from gsa_dataset import *
from gsa_gpu import *
# endregion

# -----------------------------
# Vanilla cache (device-safe)
# -----------------------------
class VanillaCache():
    def __init__(self, granu, local, cache_size, vector_size, device=None):
        self.granu = granu
        self.local = local
        self.num_granu = len(self.granu)
        self.num_local = len(self.local)
        self.cache_size = cache_size
        self.device = torch.device("cpu") if device is None or (isinstance(device, torch.device) and device.type == "meta") else device
        
        self.kv_cache = [[torch.zeros((2, 1, 4, 0, 64), device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.ts_cache = [[torch.zeros((2, 0), device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.kv_buffer = [torch.zeros((2, 1, 4, 0, 64), device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros((2, 0), device=self.device) for _ in range(self.num_granu)]

        self.cache_time = 0
        self.TIME, self.SCORE = 0, 1

    def time_mask(self, cache, time):
        idx = torch.searchsorted(cache[self.TIME], time)
        mask = torch.zeros(cache[self.TIME].shape, dtype=torch.bool, device=cache.device)
        mask[idx:] = True
        return mask

    def topk_mask(self, cache, topk, dim):
        topk_min = min(topk, cache[dim].shape[0])
        _, topk_idx = torch.topk(cache[dim], topk_min, dim=0)
        mask = torch.zeros(cache[dim].shape, dtype=torch.bool, device=cache.device)
        mask[topk_idx] = True
        return mask

    def kv_topk(self, granu, local, device):
        kv_select = self.kv_cache[granu][local]
        ts_select = self.ts_cache[granu][local]
        mask = self.topk_mask(ts_select, self.cache_size, dim=self.SCORE).to(device)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        kv_topk = kv_select[:, :, :, idx] if idx.numel() > 0 else kv_select[:, :, :, :0]
        return kv_topk, mask
    
    def kv_full(self, granu, local, device):
        kv_select = self.kv_cache[granu][local]
        ts_select = self.ts_cache[granu][local]
        mask = torch.ones(ts_select.shape[1], dtype=torch.bool, device=self.device)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        kv_full = kv_select[:, :, :, idx] if idx.numel() > 0 else kv_select[:, :, :, :0]
        return kv_full, mask

    def append(self, key_states, value_states, attention_weights):
        kv_states = torch.stack((key_states, value_states)).to(self.device)
        seq_len = attention_weights.shape[3]
        score_states = attention_weights.sum(2).sum(1).sum(0).to(self.device)
        time_states = torch.arange(self.cache_time, self.cache_time+seq_len, device=self.device)
        ts_states = torch.stack((time_states, score_states))

        for i in range(self.num_granu):
            self.kv_buffer[i] = torch.cat([self.kv_buffer[i], kv_states], dim=3)
            self.ts_buffer[i] = torch.cat([self.ts_buffer[i], ts_states], dim=1)

            buffer_len = self.kv_buffer[i].shape[3]
            new_size = (buffer_len // self.granu[i]) * self.granu[i]
            kv_append, self.kv_buffer[i] = (self.kv_buffer[i][:, :, :, :new_size, :], self.kv_buffer[i][:, :, :, new_size:, :])
            ts_append, self.ts_buffer[i] = (self.ts_buffer[i][:, :new_size], self.ts_buffer[i][:, new_size:])
            kv_compress = kv_append.view(*kv_append.shape[0:3], -1, self.granu[i], kv_append.shape[4]).mean(dim=4)
            ts_compress = ts_append.view(ts_append.shape[0], -1, self.granu[i]).mean(dim=2)

            for ii in range(self.local):
                self.kv_cache[i][ii] = torch.cat([self.kv_cache[i][ii], kv_compress.to(self.device)], dim=3)
                self.ts_cache[i][ii] = torch.cat([self.ts_cache[i][ii], ts_compress.to(self.device)], dim=1)

    def update(self, mask, attn_weights):
        for i in range(self.num_granu):
            for ii in range(self.num_local):
                device = self.ts_cache[i][ii][self.SCORE].device
                self.ts_cache[i][ii][self.SCORE][mask] += attn_weights.sum(2).sum(1).sum(0).to(device)

    def evict(self):
        for i in range(self.num_granu):
            for ii in range(self.num_local):
                if self.ts_cache[i][ii].shape[1] <= self.cache_size:
                    continue
                if self.local[ii] != 0:
                    mask = self.time_mask(self.ts_cache[i][ii], (1-self.local[ii]) * self.cache_time)
                    self.kv_cache[i][ii], _ = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], _ = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size, dim=self.SCORE)                  
                    self.kv_cache[i][ii], _ = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], _ = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                else:
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size, dim=self.TIME) 
                    self.kv_cache[i][ii], _ = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], _ = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])

    def evict_internal(self):
        # This is the correct implementation
        pass

    def clear(self):
        self.kv_cache = [[torch.zeros((2, 1, 4, 0, 64), device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.ts_cache = [[torch.zeros((2, 0), device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.kv_buffer = [torch.zeros((2, 1, 4, 0, 64), device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros((2, 0), device=self.device) for _ in range(self.num_granu)]        
        self.cache_time = 0

# -----------------------------
# Residual cache (device-safe)
# -----------------------------
class ResidualCache():
    def __init__(self, granu, local, cache_size, vector_size, device=None):
        self.granu = granu
        self.local = local
        self.num_granu = len(self.granu)
        self.num_local = len(self.local)
        self.cache_size = cache_size
        self.vector_size = vector_size
        # Avoid meta device at construction (Accelerate/HF may use it)
        self.device = torch.device("cpu") if device is None or (isinstance(device, torch.device) and device.type == "meta") else device

        self.kv_cache = [[torch.zeros(self.vector_size, device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.ts_cache = [[torch.zeros((2, 0), device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.kv_buffer = [torch.zeros(self.vector_size, device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros((2, 0), device=self.device) for _ in range(self.num_granu)]
        
        self.cache_time = 0
        self.TIME, self.SCORE = 0, 1

    def time_mask(self, cache, time):
        idx = torch.searchsorted(cache[self.TIME], time)
        mask = torch.zeros(cache[self.TIME].shape, dtype=torch.bool, device=cache.device)
        mask[idx:] = True
        return mask

    def topk_mask(self, cache, topk, dim):
        topk_min = min(topk, cache[dim].shape[0])
        _, topk_idx = torch.topk(cache[dim], topk_min, dim=0)
        mask = torch.zeros(cache[dim].shape, dtype=torch.bool, device=cache.device)
        mask[topk_idx] = True
        return mask
    
    # ---- KV selection (device-safe; boolean mask -> indices) ----
    def kv_topk(self, granu, local, device):
        kv_select = torch.cat([kv.to(device) for kv in self.kv_cache[granu][:local+1]], dim=3)
        ts_select = torch.cat([ts.to(device) for ts in self.ts_cache[granu][:local+1]], dim=1)
        mask = self.topk_mask(ts_select, self.cache_size, dim=self.SCORE).to(device)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        kv_topk = kv_select[:, :, :, idx] if idx.numel() > 0 else kv_select[:, :, :, :0]
        return kv_topk, mask
    
    # Not identical to full attention
    def kv_full(self, granu, local, device):
        kv_select = torch.cat([kv.to(device) for kv in self.kv_cache[granu][:local+1]], dim=3)
        ts_select = torch.cat([ts.to(device) for ts in self.ts_cache[granu][:local+1]], dim=1)
        mask = torch.ones(ts_select.shape[1], dtype=torch.bool, device=self.device)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        kv_full = kv_select[:, :, :, idx] if idx.numel() > 0 else kv_select[:, :, :, :0]
        return kv_full, mask

    def kv_residual(self, granu, local, device):
        kv_select = self.kv_cache[granu][local].to(device)
        ts_select = self.ts_cache[granu][local].to(device)
        mask = torch.ones(ts_select.shape[1], dtype=torch.bool, device=self.device)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        kv_residual = kv_select[:, :, :, idx] if idx.numel() > 0 else kv_select[:, :, :, :0]
        return kv_residual, mask

    def append(self, key_states, value_states, attn_weights):
        kv_states = torch.stack((key_states, value_states)).to(self.device)
        seq_len = attn_weights.shape[3]
        score_states = attn_weights.sum(2).sum(1).sum(0).to(self.device)
        time_states = torch.arange(self.cache_time, self.cache_time+seq_len, device=self.device)
        ts_states = torch.stack((time_states, score_states))

        for i in range(self.num_granu):
            self.kv_buffer[i] = torch.cat([self.kv_buffer[i], kv_states], dim=3) 
            self.ts_buffer[i] = torch.cat([self.ts_buffer[i], ts_states], dim=1) 

            buffer_len = self.kv_buffer[i].shape[3]
            new_size = (buffer_len // self.granu[i]) * self.granu[i]
            kv_append, self.kv_buffer[i] = (self.kv_buffer[i][:, :, :, :new_size, :], self.kv_buffer[i][:, :, :, new_size:, :])
            ts_append, self.ts_buffer[i] = (self.ts_buffer[i][:, :new_size], self.ts_buffer[i][:, new_size:])
            kv_compress = kv_append.view(*kv_append.shape[0:3], -1, self.granu[i], kv_append.shape[4]).mean(dim=4)
            ts_compress = ts_append.view(ts_append.shape[0], -1, self.granu[i]).mean(dim=2)
            
            self.kv_cache[i][0] = torch.cat([self.kv_cache[i][0], kv_compress.to(self.device)], dim=3) 
            self.ts_cache[i][0] = torch.cat([self.ts_cache[i][0], ts_compress.to(self.device)], dim=1)

        self.cache_time += seq_len

    def update(self, mask, attn_weights):
        for i in range(self.num_granu):
            for ii in range(self.num_local):
                if mask[i][ii] is not None and mask[i][ii].sum().item() != 0:
                    lengths = [cache[self.SCORE].shape[0] for cache in self.ts_cache[i][:ii+1]]
                    mask_splits = torch.split(mask[i][ii], lengths, dim=0)
                    attn_weights_partial, attn_weights = (attn_weights[:, :, :, -1 * mask[i][ii].sum().item():], attn_weights[:, :, :, :-1 * mask[i][ii].sum().item()])
                    attn_weights_splits = torch.split(attn_weights_partial, [mask_split.sum().item() for mask_split in mask_splits], dim=3)
                    for iii, (attn_weights_split, mask_split) in enumerate(zip(attn_weights_splits, mask_splits)):
                        if self.ts_cache[i][iii].ndim > 1:
                            self.ts_cache[i][iii][self.SCORE][mask_split.to(self.device)] += attn_weights_split.sum(2).sum(1).sum(0).to(self.device)
                        
    def update_residual(self, mask, attn_weights):
        for i in range(self.num_granu):
            for ii in range(self.num_local):
                if mask[i][ii] is not None and mask[i][ii].sum().item() != 0:
                    attn_weights_partial, attn_weights = (attn_weights[:, :, :, -1 * mask[i][ii].sum().item():], attn_weights[:, :, :, :-1 * mask[i][ii].sum().item()])
                    device = self.ts_cache[i][ii][self.SCORE].device
                    self.ts_cache[i][ii][self.SCORE][mask[i][ii].to(device)] += attn_weights_partial.sum(2).sum(1).sum(0).to(device)

    def evict(self):
        for i in range(self.num_granu):
            kv_carry, ts_carry = torch.tensor([]), torch.tensor([])
            for ii in range(self.num_local):
                self.kv_cache[i][ii] = torch.cat([self.kv_cache[i][ii], kv_carry.to(self.device)], dim=3) 
                self.ts_cache[i][ii] = torch.cat([self.ts_cache[i][ii], ts_carry.to(self.device)], dim=1) 
                if self.ts_cache[i][ii].ndim == 1 or self.ts_cache[i][ii].shape[1] <= self.cache_size:
                    continue
                if self.local[ii] != 0:
                    mask = self.time_mask(self.ts_cache[i][ii], (1-self.local[ii]) * self.cache_time)
                    self.kv_cache[i][ii], kv_carry = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], ts_carry = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size, dim=self.SCORE)
                    self.kv_cache[i][ii], _ = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], _ = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                else:
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size, dim=self.TIME)
                    self.kv_cache[i][ii], kv_carry = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], ts_carry = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])

    def evict_internal(self):
        for i in range(self.num_granu):
            kv_carry, ts_carry = torch.tensor([]), torch.tensor([])
            for ii in range(self.num_local):  
                self.kv_cache[i][ii] = torch.cat([self.kv_cache[i][ii], kv_carry.to(self.device)], dim=3)
                self.ts_cache[i][ii] = torch.cat([self.ts_cache[i][ii], ts_carry.to(self.device)], dim=1)
                if self.ts_cache[i][ii].ndim == 1 or self.ts_cache[i][ii].shape[1] <= self.cache_size:
                    continue
                if self.local[ii] != 0:
                    mask = self.time_mask(self.ts_cache[i][ii], (1-self.local[ii]) * self.cache_time)
                    self.kv_cache[i][ii], kv_carry = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], ts_carry = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                else:
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size, dim=self.TIME)
                    self.kv_cache[i][ii], kv_carry = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], ts_carry = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])

    def flush(self):
        self.kv_buffer = [torch.zeros(self.vector_size, device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros((2, 0), device=self.device) for _ in range(self.num_granu)]

        for i in range(self.num_granu):
            kv_carry, ts_carry = torch.tensor([]), torch.tensor([])
            for ii in range(self.num_local):
                self.kv_cache[i][ii] = torch.cat([self.kv_cache[i][ii], kv_carry.to(self.device)], dim=3)
                self.ts_cache[i][ii] = torch.cat([self.ts_cache[i][ii], ts_carry.to(self.device)], dim=1)
                if self.local[ii] != 0:
                    if self.ts_cache[i][ii].ndim == 1 or self.ts_cache[i][ii].shape[1] <= self.cache_size:
                        continue
                    mask = self.time_mask(self.ts_cache[i][ii], (1-self.local[ii]) * self.cache_time)
                    self.kv_cache[i][ii], kv_carry = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], ts_carry = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size, dim=self.SCORE)
                    self.kv_cache[i][ii], _ = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], _ = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                else:
                    self.kv_cache[i][ii], kv_carry = torch.zeros(self.vector_size, device=self.device), self.kv_cache[i][ii]
                    self.ts_cache[i][ii], ts_carry = torch.zeros((2, 0), device=self.device), self.ts_cache[i][ii]

    def decay(self, alpha=0.9):
        for i in range(self.num_granu):
            for ii in range(self.num_local):
                self.ts_cache[i][ii][self.SCORE] = alpha * self.ts_cache[i][ii][self.SCORE]

    def clear(self):
        self.kv_cache = [[torch.zeros(self.vector_size, device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.ts_cache = [[torch.zeros((2, 0), device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.kv_buffer = [torch.zeros(self.vector_size, device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros((2, 0), device=self.device) for _ in range(self.num_granu)]
        self.cache_time = 0

class ResidualCachewithBlock():
    def __init__(self, granu, local, cache_size, vector_size, device=None):
        self.granu = granu
        self.local = local
        self.num_granu = len(self.granu)
        self.num_local = len(self.local)
        self.cache_size = cache_size
        self.vector_size = vector_size
        self.device = torch.device("cpu") if device is None or (isinstance(device, torch.device) and device.type == "meta") else device

        self.kv_cache = [[torch.zeros([*self.vector_size[0:4], self.granu[i], self.vector_size[4]], device=self.device) for _ in range(self.num_local)] for i in range(self.num_granu)]
        self.ts_cache = [[torch.zeros((2, 0), device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.kv_buffer = [torch.zeros(self.vector_size, device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros((2, 0), device=self.device) for _ in range(self.num_granu)]
        
        self.cache_time = 0
        self.TIME, self.SCORE = 0, 1

    def time_mask(self, cache, time):
        idx = torch.searchsorted(cache[self.TIME], time)
        mask = torch.zeros(cache[self.TIME].shape, dtype=torch.bool, device=cache.device)
        mask[idx:] = True
        return mask

    def topk_mask(self, cache, topk, dim):
        topk_min = min(topk, cache[dim].shape[0])
        _, topk_idx = torch.topk(cache[dim], topk_min, dim=0)
        mask = torch.zeros(cache[dim].shape, dtype=torch.bool, device=cache.device)
        mask[topk_idx] = True
        return mask
    
    def kv_topk(self, granu, local, device):
        kv_select = torch.cat([kv.to(device) for kv in self.kv_cache[granu][:local+1]], dim=3)
        ts_select = torch.cat([ts.to(device) for ts in self.ts_cache[granu][:local+1]], dim=1)
        mask = self.topk_mask(ts_select, self.cache_size, dim=self.SCORE).to(device)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        kv_topk = kv_select[:, :, :, idx] if idx.numel() > 0 else kv_select[:, :, :, :0]
        kv_topk = kv_topk.reshape(*kv_topk.shape[0:3], -1, kv_topk.shape[5])
        return kv_topk, mask
    
    def kv_full(self, granu, local, device):
        kv_select = torch.cat([kv.to(device) for kv in self.kv_cache[granu][:local+1]], dim=3)
        ts_select = torch.cat([ts.to(device) for ts in self.ts_cache[granu][:local+1]], dim=1)
        mask = torch.ones(ts_select.shape[1], dtype=torch.bool, device=self.device)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        kv_full = kv_select[:, :, :, idx] if idx.numel() > 0 else kv_select[:, :, :, :0]
        kv_full = kv_full.reshape(*kv_full.shape[0:3], -1, kv_full.shape[5])
        return kv_full, mask

    def kv_residual(self, granu, local, device):
        kv_select = self.kv_cache[granu][local].to(device)
        ts_select = self.ts_cache[granu][local].to(device)
        mask = torch.ones(ts_select.shape[1], dtype=torch.bool, device=self.device)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        kv_residual = kv_select[:, :, :, idx] if idx.numel() > 0 else kv_select[:, :, :, :0]
        kv_residual = kv_residual.reshape(*kv_residual.shape[0:3], -1, kv_residual.shape[5])
        return kv_residual, mask

    def append(self, key_states, value_states, attn_weights):
        kv_states = torch.stack((key_states, value_states)).to(self.device)
        seq_len = attn_weights.shape[3]
        score_states = attn_weights.sum(2).sum(1).sum(0).to(self.device)
        time_states = torch.arange(self.cache_time, self.cache_time+seq_len, device=self.device)
        ts_states = torch.stack((time_states, score_states))

        for i in range(self.num_granu):
            self.kv_buffer[i] = torch.cat([self.kv_buffer[i], kv_states], dim=3) 
            self.ts_buffer[i] = torch.cat([self.ts_buffer[i], ts_states], dim=1) 

            buffer_len = self.kv_buffer[i].shape[3]
            new_size = (buffer_len // self.granu[i]) * self.granu[i]
            kv_append, self.kv_buffer[i] = (self.kv_buffer[i][:, :, :, :new_size, :], self.kv_buffer[i][:, :, :, new_size:, :])
            ts_append, self.ts_buffer[i] = (self.ts_buffer[i][:, :new_size], self.ts_buffer[i][:, new_size:])
            kv_compress = kv_append.view(*kv_append.shape[0:3], -1, self.granu[i], kv_append.shape[4])   
                  # torch.Size([2, 1, 4, 1, 16, 64]) -x-> torch.Size([2, 1, 4, 1, 64])          
            ts_compress = ts_append.view(ts_append.shape[0], -1, self.granu[i]).mean(dim=2)
            
            self.kv_cache[i][0] = torch.cat([self.kv_cache[i][0], kv_compress.to(self.device)], dim=3) 
            self.ts_cache[i][0] = torch.cat([self.ts_cache[i][0], ts_compress.to(self.device)], dim=1)

        self.cache_time += seq_len

    def update(self, mask, attn_weights):
        for i in range(self.num_granu):
            for ii in range(self.num_local):
                if mask[i][ii] is not None and mask[i][ii].sum().item() != 0:
                    lengths = [cache[self.SCORE].shape[0] for cache in self.ts_cache[i][:ii+1]]
                    mask_splits = torch.split(mask[i][ii], lengths, dim=0)
                    attn_weights_partial, attn_weights = (attn_weights[:, :, :, -1 * mask[i][ii].sum().item():], attn_weights[:, :, :, :-1 * mask[i][ii].sum().item()])
                    attn_weights_splits = torch.split(attn_weights_partial, [mask_split.sum().item() for mask_split in mask_splits], dim=3)
                    for iii, (attn_weights_split, mask_split) in enumerate(zip(attn_weights_splits, mask_splits)):
                        if self.ts_cache[i][iii].ndim > 1:
                            self.ts_cache[i][iii][self.SCORE][mask_split.to(self.device)] += attn_weights_split.sum(2).sum(1).sum(0).to(self.device)
                        
    def update_residual(self, mask, attn_weights):
        for ii in range(self.num_local):
            for i in range(self.num_granu):
                if mask[i][ii] is not None and mask[i][ii].sum().item() != 0:
                    attn_weights_partial, attn_weights = (attn_weights[:, :, :, -1 * mask[i][ii].sum().item():], attn_weights[:, :, :, :-1 * mask[i][ii].sum().item()])
                    device = self.ts_cache[i][ii][self.SCORE].device
                    self.ts_cache[i][ii][self.SCORE][mask[i][ii].to(device)] += attn_weights_partial.sum(2).sum(1).sum(0).to(device)

    def evict(self):
        for i in range(self.num_granu):
            kv_carry, ts_carry = torch.tensor([]), torch.tensor([])
            for ii in range(self.num_local):
                self.kv_cache[i][ii] = torch.cat([self.kv_cache[i][ii], kv_carry.to(self.device)], dim=3) 
                self.ts_cache[i][ii] = torch.cat([self.ts_cache[i][ii], ts_carry.to(self.device)], dim=1) 
                if self.ts_cache[i][ii].ndim == 1 or self.ts_cache[i][ii].shape[1] <= self.cache_size:
                    continue
                if self.local[ii] != 0:
                    mask = self.time_mask(self.ts_cache[i][ii], (1-self.local[ii]) * self.cache_time)
                    self.kv_cache[i][ii], kv_carry = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], ts_carry = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size//self.granu[i], dim=self.SCORE)
                    self.kv_cache[i][ii], _ = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], _ = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                else:
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size//self.granu[i], dim=self.TIME)
                    self.kv_cache[i][ii], kv_carry = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], ts_carry = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])

    def evict_internal(self):
        for i in range(self.num_granu):
            kv_carry, ts_carry = torch.tensor([]), torch.tensor([])
            for ii in range(self.num_local):  
                self.kv_cache[i][ii] = torch.cat([self.kv_cache[i][ii], kv_carry.to(self.device)], dim=3)
                self.ts_cache[i][ii] = torch.cat([self.ts_cache[i][ii], ts_carry.to(self.device)], dim=1)
                if self.ts_cache[i][ii].ndim == 1 or self.ts_cache[i][ii].shape[1] <= self.cache_size:
                    continue
                if self.local[ii] != 0:
                    mask = self.time_mask(self.ts_cache[i][ii], (1-self.local[ii]) * self.cache_time)
                    self.kv_cache[i][ii], kv_carry = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], ts_carry = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                else:
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size//self.granu[i], dim=self.TIME)
                    self.kv_cache[i][ii], kv_carry = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], ts_carry = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])

    def flush(self):
        self.kv_buffer = [torch.zeros(self.vector_size, device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros((2, 0), device=self.device) for _ in range(self.num_granu)]

        for i in range(self.num_granu):
            kv_carry, ts_carry = torch.tensor([]), torch.tensor([])
            for ii in range(self.num_local):
                self.kv_cache[i][ii] = torch.cat([self.kv_cache[i][ii], kv_carry.to(self.device)], dim=3)
                self.ts_cache[i][ii] = torch.cat([self.ts_cache[i][ii], ts_carry.to(self.device)], dim=1)
                if self.local[ii] != 0:
                    if self.ts_cache[i][ii].ndim == 1 or self.ts_cache[i][ii].shape[1] <= self.cache_size:
                        continue
                    mask = self.time_mask(self.ts_cache[i][ii], (1-self.local[ii]) * self.cache_time)
                    self.kv_cache[i][ii], kv_carry = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], ts_carry = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                    mask = self.topk_mask(self.ts_cache[i][ii], self.cache_size//self.granu[i], dim=self.SCORE)
                    self.kv_cache[i][ii], _ = (self.kv_cache[i][ii][:, :, :, mask], self.kv_cache[i][ii][:, :, :, ~mask])
                    self.ts_cache[i][ii], _ = (self.ts_cache[i][ii][:, mask], self.ts_cache[i][ii][:, ~mask])
                else:
                    self.kv_cache[i][ii], kv_carry = torch.zeros([*self.vector_size[0:4], self.granu[i], self.vector_size[4]], device=self.device), self.kv_cache[i][ii]
                    self.ts_cache[i][ii], ts_carry = torch.zeros((2, 0), device=self.device), self.ts_cache[i][ii]

    def decay(self, alpha=0.9):
        for i in range(self.num_granu):
            for ii in range(self.num_local):
                self.ts_cache[i][ii][self.SCORE] = alpha * self.ts_cache[i][ii][self.SCORE]

    def clear(self):
        self.kv_cache = [[torch.zeros([*self.vector_size[0:4], self.granu[i], self.vector_size[4]], device=self.device) for _ in range(self.num_local)] for i in range(self.num_granu)]
        self.ts_cache = [[torch.zeros((2, 0), device=self.device) for _ in range(self.num_local)] for _ in range(self.num_granu)]
        self.kv_buffer = [torch.zeros(self.vector_size, device=self.device) for _ in range(self.num_granu)]
        self.ts_buffer = [torch.zeros((2, 0), device=self.device) for _ in range(self.num_granu)]
        self.cache_time = 0

# -----------------------------
# Adapter + attention with schema mixing (device-safe)
# -----------------------------
# class GSA1_Adapter(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2):
#         super().__init__()
#         self.output_dim1 = output_dim1
#         self.output_dim2 = output_dim2
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         self.adapter_down = nn.Linear(input_dim, hidden_dim, bias=False)
#         self.adapter_act1 = nn.GELU()
#         self.adapter_up = nn.Linear(hidden_dim, output_dim1 * output_dim2, bias=False)
#         self.adapter_act2 = nn.GELU()
#         self.adapter_dropout = nn.Dropout(p=0.1)

#     def init_weights(self, weights):
#         nn.init.uniform_(self.adapter_down.weight, b=weights)
#         nn.init.uniform_(self.adapter_up.weight, b=weights)

#     def forward(self, x):
#         x = x[:, :, -1, :]
#         x = x.mean(dim=(0, 1))
#         x = self.adapter_dropout(x)
#         x = self.adapter_down(x)
#         x = self.adapter_act1(x)
#         x = self.adapter_up(x)
#         x = self.adapter_act2(x)
#         x = x.view(self.output_dim1, self.output_dim2)
#         return x
    
class GSA1_Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2, layer_idx):
        super().__init__()
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.layer_idx = layer_idx

        self.adapter_down = nn.Linear(input_dim, hidden_dim, bias=False)
        self.adapter_act1 = nn.GELU()
        self.adapter_up = nn.Linear(hidden_dim, output_dim1 * output_dim2, bias=False)
        self.adapter_act2 = nn.GELU()

        self.adapter_dropout = nn.Dropout(p=0.1)
        self.scale = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

    def init_weights(self, weights):
        nn.init.uniform_(self.adapter_down.weight, b=weights)
        nn.init.uniform_(self.adapter_up.weight, b=weights)
    
    def forward(self, x):
        batch = x.shape[0]
        x = x[:, :, -1, :]
        x = x.mean(dim=1)
        x = self.adapter_dropout(x)

        x = self.adapter_down(x)
        x = self.adapter_act1(x)
        x = self.adapter_up(x)
        x = self.adapter_act2(x)

        # if self.layer_idx == 0:
        #     print("                                    ", x.tolist())
        x = x.view(batch, self.output_dim1, self.output_dim2)
        return x

# class GSA1_Adapter(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2, layer_idx):
#         super().__init__()
#         self.output_dim1 = output_dim1
#         self.output_dim2 = output_dim2
#         self.layer_idx = layer_idx

#         self.adapter_down = nn.Linear(input_dim, hidden_dim, bias=False)
#         self.adapter_act1 = nn.GELU()
#         self.adapter_up = nn.Linear(hidden_dim, output_dim1 * output_dim2, bias=False)
#         self.adapter_act2 = nn.GELU()

#         self.adapter_dropout = nn.Dropout(p=0.1)
        
#         # FIX: Proper initialization
#         self.scale = nn.Parameter(torch.full((1,), 0.1))  # Shape: [1]
        

#     def init_weights(self, _):
#         # Initialize adapter weights
#         nn.init.uniform_(self.adapter_down.weight, a=-0.1, b=0.1)
#         nn.init.uniform_(self.adapter_up.weight, a=-0.1, b=0.1)
        
#         # Initialize scale
#         nn.init.constant_(self.scale, 0.1)

#     def forward(self, x):
#         batch = x.shape[0]
        
#         # Take last token and average over heads
#         x = x[:, :, -1, :]  # [batch, heads, dim]
#         x = x.mean(dim=1)    # [batch, dim]
        
#         # Apply adapter
#         x = self.adapter_dropout(x)
#         x = self.adapter_down(x)
#         x = self.adapter_act1(x)
#         x = self.adapter_up(x)
#         x = self.adapter_act2(x)

#         # Reshape to output matrix
#         x = x.view(batch, self.output_dim1, self.output_dim2)
        
#         # Apply scale - now gradients will flow
#         x = x * self.scale  # Broadcasting works with shape [1]
        
#         return x

class GSA2_Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adapter_down = nn.Linear(input_dim, hidden_dim, bias=False)
        self.adapter_act1 = nn.GELU()
        self.adapter_up = nn.Linear(hidden_dim, self.output_dim, bias=False)
        self.adapter_act2 = nn.GELU()
        self.adapter_dropout = nn.Dropout(p=0.1)

    def init_weights(self, weights):
        nn.init.uniform_(self.adapter_down.weight, b=weights)
        nn.init.uniform_(self.adapter_up.weight, b=weights)

    def forward(self, x):
        res = x[:,:,:self.output_dim]
        x = self.adapter_dropout(x)
        x = self.adapter_down(x)
        x = self.adapter_act1(x)
        x = self.adapter_up(x)
        x = self.adapter_act2(x)
        x = x + res
        return x

# -----------------------------
# GSA Llama attention
# -----------------------------
class GSA_LlamaAttention(LlamaAttention):
    def __init__(
        self, config, granu, local, prefill_weights, decode_weights, cache_type, cache_size, vector_size,
        clamp_min=None, clamp_max=None, normalize_rows=True, *args, **kwargs
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

        self.granu = granu
        self.local = local
        self.num_granu = len(granu)
        self.num_local = len(local)

        # Validate shapes (robust against any G/L)
        assert prefill_weights.shape == (self.num_granu, self.num_local), \
            f"prefill_weights must be [{self.num_granu} x {self.num_local}], got {tuple(prefill_weights.shape)}"
        assert decode_weights.shape == (self.num_granu, self.num_local), \
            f"decode_weights must be [{self.num_granu} x {self.num_local}], got {tuple(decode_weights.shape)}"

        self.prefill_weights = prefill_weights
        self.decode_weights = decode_weights
        self.cache_size = cache_size
        self.vector_size = vector_size

        # Cache on a real device (avoid meta at __init__)
        try:
            param_device = next(self.parameters()).device
        except StopIteration:
            param_device = torch.device("cpu")
        cache_device = torch.device("cpu") #if param_device.type == "meta" else param_device

        if cache_type == "Vanilla":
            self.cache = VanillaCache(
                self.granu, self.local, self.cache_size, self.vector_size,
                device=cache_device
            )
        elif cache_type == "Residual":
            self.cache = ResidualCache(
                self.granu, self.local, self.cache_size, self.vector_size,
                device=cache_device
            )
        elif cache_type == "ResidualwithBlock":
            self.cache = ResidualCachewithBlock(
                self.granu, self.local, self.cache_size, self.vector_size,
                device=cache_device
            )
        else:
            raise ValueError(f"Unregistered Cache Type: {cache_type}.")     
        
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.normalize_rows = normalize_rows

    def forward_train(self):
        pass

    def forward_eval(self):
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mode = "train",
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "train":
            return self.forward_train(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )
        else:
            return self.forward_eval(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )

class GSA1_LlamaAttention(GSA_LlamaAttention):
    def __init__(
        self, config, granu, local, prefill_weights, decode_weights, cache_type, cache_size, vector_size,
        clamp_min=None, clamp_max=None, normalize_rows=True, *args, **kwargs
    ):
        super().__init__(config, granu, local, prefill_weights, decode_weights, cache_type, cache_size, vector_size,
            clamp_min, clamp_max, normalize_rows, *args, **kwargs)
        self.adapter = GSA1_Adapter(
            input_dim=vector_size[4], hidden_dim=vector_size[4],
            output_dim1=self.num_granu, output_dim2=self.num_local,
            layer_idx=self.layer_idx
        )
        self.evict_cycle = int(math.log2(cache_size))

    def _postprocess_weights(self, weights):
        if self.clamp_min is not None or self.clamp_max is not None:
            lo = self.clamp_min if self.clamp_min is not None else float("-inf")
            hi = self.clamp_max if self.clamp_max is not None else float("inf")
            weights = torch.clamp(weights, lo, hi)
        if self.normalize_rows:
            weights = torch.softmax(weights, dim=1)  # normalize across local per granu row
        return weights

    def prefill_schema_weights(self, query_states, mode=None):
        base = self.prefill_weights
        adapt = self.adapter(query_states).to(base.device)
        weights = base + adapt
        weights_softmax = F.softmax(weights.flatten(), dim=0).view(weights.shape) * weights.numel()
        return weights_softmax
        # return self._postprocess_weights(weights)
        
    
    def decode_schema_weights(self, query_states):
        base = self.decode_weights
        adapt = self.adapter(query_states).to(base.device)
        weights = base + adapt
        weights_softmax = F.softmax(weights.flatten(), dim=0).view(weights.shape) * weights.numel()
        return weights_softmax
        # return self._postprocess_weights(weights)

    def forward_train(
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

        if prefill:
            attn_output, attn_weights = sdpa_attention_forward(
                self,
                query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )

            with torch.no_grad():   
                self.cache.append(key_states, value_states, attn_weights)
                self.cache.evict_internal()

        schema_weights = self.prefill_schema_weights(query_states) if prefill else self.decode_schema_weights(query_states)
        schema_mask = [[None for _ in range(self.cache.num_local)] for _ in range(self.cache.num_granu)]
        key_concat, value_concat = key_states, value_states

        for ii in range(self.cache.num_local):
            for i in range(self.cache.num_granu):  
                schema_kv_states, schema_mask[i][ii] = self.cache.kv_residual(i, ii, device=next(self.parameters()).device)
                schema_key_states = schema_kv_states[0, :, :, :].to(query_states.device, dtype=query_states.dtype)
                schema_value_states = schema_kv_states[1, :, :, :].to(query_states.device, dtype=query_states.dtype)
                
                w = schema_weights[:,i,ii].to(schema_value_states.device, dtype=query_states.dtype)
                schema_value_states = w * schema_value_states
                key_concat   = torch.cat([schema_key_states, key_concat],  dim=2)
                value_concat = torch.cat([schema_value_states, value_concat], dim=2)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states, key_concat, value_concat, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        with torch.no_grad():
            self.cache.update_residual(schema_mask, attn_weights[:, :, :, :-1 * key_states.shape[2]])  
            self.cache.append(key_states, value_states, attn_weights[:, :, :, -1 * key_states.shape[2]:])
            self.cache.evict()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output) 
        return attn_output, None

    def forward_eval(
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

        schema_weights = self.prefill_schema_weights(query_states, mode="eval") if prefill else self.decode_schema_weights(query_states)
        schema_mask = [[None for _ in range(self.cache.num_local)] for _ in range(self.cache.num_granu)]
        key_concat, value_concat = key_states, value_states

        # if self.layer_idx == 0:
        #     print("                                    ", schema_weights.tolist())

        for ii in range(self.cache.num_local):
            for i in range(self.cache.num_granu):
                schema_kv_states, schema_mask[i][ii] = self.cache.kv_residual(i, ii, device=next(self.parameters()).device)
                schema_key_states = schema_kv_states[0, :, :, :].to(query_states.device, dtype=query_states.dtype)
                schema_value_states = schema_kv_states[1, :, :, :].to(query_states.device, dtype=query_states.dtype)
                
                w = schema_weights[:,i,ii].to(schema_value_states.device, dtype=query_states.dtype)
                schema_value_states = w * schema_value_states
                key_concat   = torch.cat([schema_key_states, key_concat],  dim=2)
                value_concat = torch.cat([schema_value_states, value_concat], dim=2)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states, key_concat, value_concat, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        with torch.no_grad():   
            self.cache.update_residual(schema_mask, attn_weights[:, :, :, :-1 * key_states.shape[2]])
            self.cache.append(key_states, value_states, attn_weights[:, :, :, -1 * key_states.shape[2]:])
            if prefill or self.cache.cache_time % self.evict_cycle == 0:
                self.cache.evict()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output) 
        return attn_output, None

class GSA1a_LlamaAttention(GSA1_LlamaAttention):
    def __init__(
        self, config, granu, local, prefill_weights, decode_weights, cache_type, cache_size, vector_size,
        clamp_min=None, clamp_max=None, normalize_rows=True, *args, **kwargs
    ):
        super().__init__(config, granu, local, prefill_weights, decode_weights, cache_type, cache_size, vector_size,
            clamp_min, clamp_max, normalize_rows, *args, **kwargs)
        self.adapter = GSA1_Adapter(
            input_dim=vector_size[4], hidden_dim=self.num_granu * self.num_local,
            output_dim1=self.num_granu, output_dim2=self.num_local
        )

class GSA2_LlamaAttention(GSA_LlamaAttention):
    def __init__(
        self, config, granu, local, prefill_weights, decode_weights, cache_type, cache_size, vector_size, attn_size,
        clamp_min=None, clamp_max=None, normalize_rows=True, *args, **kwargs
    ):
        super().__init__(config, granu, local, prefill_weights, decode_weights, cache_type, cache_size, vector_size,
            clamp_min, clamp_max, normalize_rows, *args, **kwargs)
        self.adapter = GSA2_Adapter(
            input_dim=attn_size * self.num_granu * self.num_local, hidden_dim=attn_size,
            output_dim=attn_size
        )
        self.evict_cycle = int(math.log2(cache_size))

    def forward_train(self,
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

        if prefill:
            attn_output, attn_weights = sdpa_attention_forward(
                self,
                query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )

            with torch.no_grad():   
                self.cache.append(key_states, value_states, attn_weights)
                self.cache.evict_internal()

        schema_weights = self.prefill_weights if prefill else self.decode_weights
        schema_mask = [[None for _ in range(self.cache.num_local)] for _ in range(self.cache.num_granu)]
        attn_output, attn_weights1, attn_weights2 = None, None, None

        for i in range(self.cache.num_granu):
            for ii in range(self.cache.num_local):
                schema_kv_states, schema_mask[i][ii] = self.cache.kv_full(i, ii, device=next(self.parameters()).device)
                schema_key_states = schema_kv_states[0, :, :, :].to(query_states.device, dtype=query_states.dtype)
                schema_value_states = schema_kv_states[1, :, :, :].to(query_states.device, dtype=query_states.dtype)
                
                schema_key_states = torch.cat([schema_key_states, key_states], dim=2)
                schema_value_states = torch.cat([schema_value_states, value_states], dim=2)
                schema_attn_output, schema_attn_weights = sdpa_attention_forward(
                    self,
                    query_states, schema_key_states, schema_value_states, attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                )
                schema_attn_output = schema_attn_output.reshape(*input_shape, -1).contiguous()

                schema_attn_weights1, schema_attn_weights2 = schema_attn_weights[:, :, :, :-1 * key_states.shape[2]], schema_attn_weights[:, :, :, -1 * key_states.shape[2]:]
                attn_output = torch.cat([attn_output, schema_weights[i][ii] * schema_attn_output], dim=2)  if attn_output is not None else schema_weights[i][ii] * schema_attn_output
                attn_weights1 = torch.cat([attn_weights1, schema_attn_weights1], dim=3) if attn_weights1 is not None else schema_attn_weights1
                attn_weights2 = torch.cat([attn_weights2, schema_attn_weights2], dim=2) if attn_weights2 is not None else schema_attn_weights2 

        with torch.no_grad():
            self.cache.update(schema_mask, attn_weights1)  
            self.cache.append(key_states, value_states, attn_weights2)
            self.cache.evict()

        attn_output = self.adapter(attn_output)
        attn_output = self.o_proj(attn_output)    
        return attn_output, None

    def forward_eval(
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

        schema_weights = self.prefill_weights if prefill else self.decode_weights
        schema_mask = [[None for _ in range(self.cache.num_local)] for _ in range(self.cache.num_granu)]
        attn_output, attn_weights1, attn_weights2 = None, None, None

        for i in range(self.cache.num_granu):
            for ii in range(self.cache.num_local):
                schema_kv_states, schema_mask[i][ii] = self.cache.kv_full(i, ii, device=next(self.parameters()).device)
                schema_key_states = schema_kv_states[0, :, :, :].to(query_states.device, dtype=query_states.dtype)
                schema_value_states = schema_kv_states[1, :, :, :].to(query_states.device, dtype=query_states.dtype)
                
                schema_key_states = torch.cat([schema_key_states, key_states], dim=2)
                schema_value_states = torch.cat([schema_value_states, value_states], dim=2)
                schema_attn_output, schema_attn_weights = sdpa_attention_forward(
                    self,
                    query_states, schema_key_states, schema_value_states, attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                )
                schema_attn_output = schema_attn_output.reshape(*input_shape, -1).contiguous()

                schema_attn_weights1, schema_attn_weights2 = schema_attn_weights[:, :, :, :-1 * key_states.shape[2]], schema_attn_weights[:, :, :, -1 * key_states.shape[2]:]
                attn_output = torch.cat([attn_output, schema_weights[i][ii] * schema_attn_output], dim=2)  if attn_output is not None else schema_weights[i][ii] * schema_attn_output
                attn_weights1 = torch.cat([attn_weights1, schema_attn_weights1], dim=3) if attn_weights1 is not None else schema_attn_weights1
                attn_weights2 = torch.cat([attn_weights2, schema_attn_weights2], dim=2) if attn_weights2 is not None else schema_attn_weights2 

        with torch.no_grad():
            self.cache.update_residual(schema_mask, attn_weights1)  
            self.cache.append(key_states, value_states, attn_weights2)
            if prefill or self.cache.cache_time % self.evict_cycle == 0:
                self.cache.evict()

        attn_output = self.adapter(attn_output)
        attn_output = self.o_proj(attn_output)    
        return attn_output, None

# -----------------------------
# Main function + arguments
# -----------------------------
def train_gsa(args, model, tokenizer, device):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps * num_gpus
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Per device batch size: {args.train_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Number of GPUs: {num_gpus}")

    config_dict = vars(args)
    config_dict.update({
        "device": str(device),
        "num_gpus": num_gpus,
        "effective_batch_size": effective_batch_size,
        "torch_version": torch.__version__,
        "seed": args.seed
    })

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

    # dataset_name = "allenai/Dolci-Instruct-SFT"
    # dataset = load_dataset(dataset_name)
    # train_dataset_raw = dataset['train'].select(range(min(10000, len(dataset['train']))))
    # train_dataset = DolciDataset(train_dataset_raw, tokenizer)
    
    if 'validation' in dataset:
        eval_dataset_raw = dataset['validation'].select(range(min(500, len(dataset['validation']))))
    else:
        eval_dataset_raw = dataset['train'].select(range(10000, min(10500, len(dataset['train']))))
    eval_dataset = AlpacaDataset(eval_dataset_raw, tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_callback = GenerationMonitorCallback(
        tokenizer=tokenizer,
        raw_dataset=eval_dataset_raw,
        device=device,
        log_steps=100
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if args.train_batch_size > 1 else None,
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir=output_dir_path,
        overwrite_output_dir=True,
        num_train_epochs=1, 
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,  
        ### effective_batch_size = per_device_train_batch_size × gradient_accumulation_steps × number_of_gpus
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=1000,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
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
        resume_from_checkpoint=args.checkpoint,
        ignore_data_skip=False,
        seed=args.seed,
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
    
    trainer.train(resume_from_checkpoint=args.checkpoint)
    trainer.save_model()
    # tokenizer.save_pretrained(output_dir_path)
    print("Training completed!")

def main_train(args, model, tokenizer, device):
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")
    # model.gradient_checkpointing_enable()

    train_gsa(args, model, tokenizer, device)

def main_eval(args, model, tokenizer, device):
    for name, param in model.named_parameters():
        param.requires_grad = False

    dataset = load_dataset("tatsu-lab/alpaca")
    list_data = dataset['train'] 

    prompts = []
    for item in list_data:
        instruction = item['instruction']
        input_text = item['input']

        if input_text and len(input_text.strip()) > 0:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction
        prompts.append(prompt)
    
    for idx, prompt in enumerate(prompts):      
        print(f"\n=== Sample {idx + 1} ===")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print("\n" + "="*50)
        print("Generating response (streaming)...")
        print("="*50)

        # Generate with streaming output
        response, _ = generate_response(
            model, tokenizer, prompt, 
            device, args.max_gen_len,
            stream_output=True  # STREAMING - shows tokens as they're generated
        )
        
        ground_truth = list_data[idx]['output']
        print(f"\n{'='*50}")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'='*50}")

        if args.pause_between:
            input("\nPress Enter to continue to next sample...")

def main(args):
    if args.model_type == "Llama":
        pass
    elif args.model_type == "GSA1":
        transformers.models.llama.modeling_llama.LlamaAttention = \
            functools.partial(
                GSA1_LlamaAttention,
                granu=args.granu,
                local=args.local,
                prefill_weights=torch.tensor(args.prefill_weights, dtype=torch.float32),
                decode_weights=torch.tensor(args.decode_weights, dtype=torch.float32),
                cache_type=args.cache_type,
                cache_size=args.cache_size,
                vector_size=args.vector_size,
                attn_size=args.attn_size
            )
    elif args.model_type == "GSA1a":
        transformers.models.llama.modeling_llama.LlamaAttention = \
            functools.partial(
                GSA1a_LlamaAttention,
                granu=args.granu,
                local=args.local,
                prefill_weights=torch.tensor(args.prefill_weights, dtype=torch.float32),
                decode_weights=torch.tensor(args.decode_weights, dtype=torch.float32),
                cache_type=args.cache_type,
                cache_size=args.cache_size,
                vector_size=args.vector_size,
                attn_size=args.attn_size
            )
    elif args.model_type == "GSA2":
        transformers.models.llama.modeling_llama.LlamaAttention = \
            functools.partial(
                GSA2_LlamaAttention,
                granu=args.granu,
                local=args.local,
                prefill_weights=torch.tensor(args.prefill_weights, dtype=torch.float32),
                decode_weights=torch.tensor(args.decode_weights, dtype=torch.float32),
                cache_type=args.cache_type,
                cache_size=args.cache_size,
                vector_size=args.vector_size,
                attn_size=args.attn_size
            )
    else:
        raise ValueError("Unregistered Model Type")

    if args.checkpoint is not None:
        print(f"Loading Weights from {args.checkpoint}.")
        input("="*50)
        checkpoint_path = args.checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False,
            trust_remote_code=True
        )
    else:
        model, tokenizer = load(args.model_name_or_path)
        print(f"Initializing adapter.")
        for name, module in model.named_modules():
            if hasattr(module, '__class__') and re.search(r'GSA[12]_LlamaAttention', module.__class__.__name__):
                module.adapter.init_weights(args.adapter_init_weights)

    print(f"Initializing cache.")
    for name, module in model.named_modules():
        if hasattr(module, '__class__') and re.search(r'GSA[12]_LlamaAttention', module.__class__.__name__):
            module.cache.clear()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Model is using GPU." if device.type == 'cuda' else "Model is using CPU.")
    model.to(device)
    
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
    print(tokenizer.chat_template)

    if args.train:
        main_train(args, model, tokenizer, device)
    else:
        main_eval(args, model, tokenizer, device)

def fix_seed(seed):
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--model_abbrev", type=str, default="Llmam3B")
    parser.add_argument("--vector_size", default=(2, 1, 8, 0, 128))
    # parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # parser.add_argument("--model_abbrev", type=str, default="TinyLlama")
    # parser.add_argument("--vector_size", default=(2, 1, 4, 0, 64))
    parser.add_argument("--attn_size", type=int, default=2048)
    
    # GSA arguments
    # parser.add_argument("--model_type", type=str, default="Llama")
    parser.add_argument("--model_type", type=str, default="GSA1")
    parser.add_argument("--cache_type", type=str, default="ResidualwithBlock")
    parser.add_argument("--cache_size", type=int, default=64)
    parser.add_argument("--granu", type=int, default=[1, 16])
    parser.add_argument("--local", type=float, default=[0.0, 1.0])
    parser.add_argument("--prefill_weights", default=[[0, 0], [0, 0]])
    parser.add_argument("--decode_weights", default=[[0, 0], [0, 0]])
    parser.add_argument("--adapter_init_weights", type=float, default=1e-6)
    
    # Training / Inference arguments
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    # parser.add_argument("--checkpoint", type=str, default="TinyLlama/GSA1_ResidualwithBlock_64/Ver_1/checkpoint-5000")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--learning_rate", type=int, default=1e-6)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Other arguments
    parser.add_argument("--dataset_path", type=str, default="data/alpaca_data", help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="The output directory where the model will be saved")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Floating point precision")
    parser.add_argument("--max_gen_len", type=int, default=500)
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--pause_between", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=random.randint(0, 10000))
    
    args = parser.parse_args()
    fix_seed(args.seed)
    main(args)
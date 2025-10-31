import math
import torch
from torch import nn
import transformers
from transformers.models.llama.modeling_llama import LlamaAttention, rotate_half
from streaming_llm.utils import load, load_jsonl

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils import is_torch_npu_available, is_torch_xpu_available
from transformers.utils.import_utils import is_torch_greater_or_equal

import argparse, functools
import os, inspect
from typing import Optional, Tuple




_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()


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
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

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




class MultiLayersCache():
    def __init__(self, granularity, num_granularity, locality, num_locality, cache_size=128):
        self.granularity = granularity
        self.num_granularity = num_granularity
        self.locality = locality
        self.num_locality = num_locality
        self.time_score_cache = [[ torch.zeros((2,0)) for _ in range(num_locality) ] for _ in range(num_granularity)]
        self.key_value_cache = [[ torch.zeros((2,1,4,0,64)) for _ in range(num_locality) ] for _ in range(num_granularity)]
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
    

    def evict(self):
        for i in range(self.num_granularity):
            time_score_carry, key_value_carry = torch.tensor([]), torch.tensor([])
            for ii in range(self.num_locality):
                self.time_score_cache[i][ii] = torch.cat([self.time_score_cache[i][ii], time_score_carry], dim=1)
                self.key_value_cache[i][ii] = torch.cat([self.key_value_cache[i][ii], key_value_carry], dim=3)
                if self.time_score_cache[i][ii].shape[1] <= self.cache_size:
                    continue

                if self.locality[ii] != 0:
                    mask = self.time_mask(self.time_score_cache[i][ii], (1-self.locality[ii]) * self.cache_time)
                    self.time_score_cache[i][ii], time_score_carry = self.time_score_cache[i][ii][:,mask], self.time_score_cache[i][ii][:,~mask]
                    self.key_value_cache[i][ii], key_value_carry = self.key_value_cache[i][ii][:,:,:,mask], self.key_value_cache[i][ii][:,:,:,~mask]

                    mask = self.topk_mask(self.time_score_cache[i][ii], self.cache_size, dim=self.SCORE)
                    self.time_score_cache[i][ii], _ = self.time_score_cache[i][ii][:,mask], self.time_score_cache[i][ii][:,~mask]
                    self.key_value_cache[i][ii], _ = self.key_value_cache[i][ii][:,:,:,mask], self.key_value_cache[i][ii][:,:,:,~mask]
                else:
                    mask = self.topk_mask(self.time_score_cache[i][ii], self.cache_size, dim=self.TIME)
                    self.time_score_cache[i][ii], time_score_carry = self.time_score_cache[i][ii][:,mask], self.time_score_cache[i][ii][:,~mask]
                    self.key_value_cache[i][ii], key_value_carry = self.key_value_cache[i][ii][:,:,:,mask], self.key_value_cache[i][ii][:,:,:,~mask]


    def append(self, key_states, value_states, attention_weights):
        for i in range(self.num_granularity):
            seq_len = attention_weights.shape[2]
            time_states = torch.Tensor(range(self.cache_time, self.cache_time+seq_len))
            score_states = attention_weights.sum(2).sum(1).sum(0)

            new_size = (seq_len // self.granularity[i]) * self.granularity[i]   
            time_states, score_states = time_states[:new_size], score_states[:new_size]
            key_states, value_states = key_states[:,:,:new_size,:], value_states[:,:,:new_size,:]

            compressed_key_states = key_states.view(*key_states.shape[0:2], -1, self.granularity[i], key_states.shape[3]).mean(dim=3)
            compressed_value_states = value_states.view(*value_states.shape[0:2], -1, self.granularity[i], value_states.shape[3]).mean(dim=3)
            compressed_time_states = time_states.view(-1, self.granularity[i]).mean(dim=1)
            compressed_score_states = score_states.view(-1, self.granularity[i]).mean(dim=1)

            try:
                compressed_key_value_states = torch.stack((compressed_key_states, compressed_value_states))
                compressed_time_score_states = torch.stack((compressed_time_states, compressed_score_states))
                self.key_value_cache[i][0] = torch.cat([self.key_value_cache[i][0], compressed_key_value_states], dim=3)
                self.time_score_cache[i][0] = torch.cat([self.time_score_cache[i][0], compressed_time_score_states], dim=1)
            except:
                input("Keyboard Interpret Guardrail")

        self.cache_time += seq_len


    def update(self, mask, attn_weights):
        for i in range(self.num_granularity):
            for ii in range(self.num_locality):
                if mask[i][ii] is not None:   
                    lengths = [cache[self.SCORE].shape[0] for cache in self.time_score_cache[i][:ii+1]]
                    mask_splits = torch.split(mask[i][ii], lengths, dim=0)
                    attn_weights_partial, attn_weights = attn_weights[:,:,:,-1 * mask[i][ii].sum().item():], attn_weights[:,:,:,:-1 * mask[i][ii].sum().item()]
                    attn_weights_splits = torch.split(attn_weights_partial, [mask_split.sum().item() for mask_split in mask_splits], dim=3)
                    for iii, (attn_weights_split, mask_split) in enumerate(zip(attn_weights_splits, mask_splits)):
                            self.time_score_cache[i][iii][1][mask_split] += attn_weights_split.sum(2).sum(1).sum(0)


    def clear(self):
        self.time_score_cache = [[ torch.zeros((2,0)) for _ in range(self.num_locality) ] for _ in range(self.num_granularity)]
        self.key_value_cache = [[ torch.zeros((2,1,4,0,64)) for _ in range(self.num_locality) ] for _ in range(self.num_granularity)]
        self.cache_time = 0


class CacheLlamaAttention(LlamaAttention):
    def __init__(self, config, granularity, locality, cache_size, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.granularity = granularity
        self.num_graularity = len(granularity)
        self.locality = locality
        self.num_locality = len(locality)
        self.cache_size = cache_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.cache = MultiLayersCache(self.granularity, self.num_graularity, self.locality, self.num_locality, self.cache_size)


    def get_prefill_schema_weights(self, query_states, local_attn_weights, topk=2):
        # Attention Weights during the Prefill Phrase
        return torch.tensor(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        )


    def get_schema_weights(self, query_states, local_attn_weights, topk=2):
        # Attention Weights during the Inference Phrase
        return torch.tensor(
            [[1, 0, 1],
             [0, 0, 1],
             [0, 0, 1]]
        )


    def get_schema_key_value_states(self, granu, local):
        select_key_value_states = torch.cat(self.cache.key_value_cache[granu][:local+1], dim=3)
        select_time_score_states = torch.cat(self.cache.time_score_cache[granu][:local+1], dim=1)
        mask = self.cache.topk_mask(select_time_score_states, self.cache.cache_size, dim=self.cache.SCORE)
        topk_key_value_states = select_key_value_states[:,:,:,mask]
        return topk_key_value_states, mask


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        prefill = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if prefill:
            self.cache.clear()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_values.update(torch.zeros(key_states.shape), torch.zeros(key_states.shape), self.layer_idx, cache_kwargs)

        local_key_states, local_value_states = key_states, value_states
        local_attn_output, local_attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        
        schema_weights = self.get_schema_weights(query_states, local_attn_output) if not prefill else self.get_prefill_schema_weights(query_states, local_attn_output)
        schema_mask = [[ None for _ in range(self.cache.num_locality) ] for _ in range(self.cache.num_granularity)]

        if not prefill:
            for i in range(self.cache.num_granularity):
                for ii in range(self.cache.num_locality):
                    if schema_weights[i][ii] != 0:
                        schema_key_value_states, schema_mask[i][ii] = self.get_schema_key_value_states(i, ii)
                        schema_key_states, schema_value_states = schema_key_value_states[0,:,:,:].to(query_states.dtype), schema_key_value_states[1,:,:,:].to(query_states.dtype)
                        schema_key_states, schema_value_states = schema_key_states, schema_weights[i][ii] * schema_value_states
                        key_states = torch.cat([schema_key_states, key_states], dim=2)
                        value_states = torch.cat([schema_value_states, value_states], dim=2)

        schema_attn_output, schema_attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = schema_attn_output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        self.cache.update(schema_mask, schema_attn_weights[:,:,:,:-1*local_key_states.shape[2]])
        self.cache.append(local_key_states, local_value_states, schema_attn_weights[:,:,:,-1*local_key_states.shape[2]:])
        self.cache.evict()

        return attn_output, None




def main(args):
    transformers.models.llama.modeling_llama.LlamaAttention = \
         functools.partial(CacheLlamaAttention, granularity=args.granularity, locality=args.locality, cache_size=args.cache_size)
    model, tokenizer = load(args.model_name_or_path)

    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]
   
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)

        past_key_values = None
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, prefill=True)    
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [pred_token_idx.item()]

        pos = 0
        max_gen_len = 1000  # Maximum Generation Length
        for _ in range(max_gen_len - 1):
            outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values)
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--granularity", type=int, default=[1, 2, 4])  # Granularity / Compression Ratio
    parser.add_argument("--locality", type=float, default=[0.0, 0.5, 1])  # locality / Reception FIeld
    parser.add_argument("--cache_size", type=int, default=128)  # Cache Size
    parser.add_argument("--data_root", type=str, default="data/")
    args = parser.parse_args()
    main(args)
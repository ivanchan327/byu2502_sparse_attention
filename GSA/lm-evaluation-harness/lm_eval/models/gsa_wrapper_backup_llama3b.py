#
# lm_eval/models/gsa_wrapper.py (FINAL, FLEXIBLE VERSION)
#

# ==================== PATH CORRECTION BLOCK ====================
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_gsa_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
if _gsa_dir not in sys.path:
    sys.path.append(_gsa_dir)
# =================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows

from .gsa import GSA1_LlamaAttention

# --- HELPER FUNCTION TO PARSE BOOLEAN ARGUMENTS ---
def _parse_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).lower() in ["true", "1"]

@register_model("gsa")
class GSAModel(LM):
    def __init__(self, **kwargs):
        super().__init__()

        base_model_path = kwargs.get("base_model_path")
        if base_model_path is None:
            raise ValueError("Must provide 'base_model_path' (e.g., 'meta-llama/Llama-3.2-3B').")

        self._base_model_path = base_model_path
        
        # --- THIS IS THE LOGIC THAT WAS MISSING ---
        # It correctly checks if we should load a checkpoint or run a baseline.
        load_checkpoint_val = kwargs.get("load_checkpoint", True)
        self.load_checkpoint = _parse_bool(load_checkpoint_val)

        # --- HARDCODED COMPLEX PARAMETERS ---
        gsa_granu_list = [1]
        gsa_local_list = [0.0, 1.0]
        gsa_prefill_weights_list = [[1, 1]]
        gsa_decode_weights_list = [[1, 1]]
        gsa_cache_type = "Residual"
        gsa_cache_size = 256
        gsa_vector_size = 128 # Using the corrected size

        gsa_granu = gsa_granu_list
        gsa_local = torch.tensor(gsa_local_list, dtype=torch.float)
        gsa_prefill_weights = torch.tensor(gsa_prefill_weights_list, dtype=torch.float)
        gsa_decode_weights = torch.tensor(gsa_decode_weights_list, dtype=torch.float)
        
        self._device = torch.device(kwargs.get("device", "cuda"))
        
        # 1. Load the base model
        print(f"\n[INFO] Loading base model architecture from: {self._base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self._base_model_path,
            torch_dtype=torch.float32 
        )

        # 2. Patch the architecture
        print("[INFO] Patching base model with GSA modules...")
        attention_class_to_replace = self.model.model.layers[0].self_attn.__class__
        print(f"[INFO] Detected attention class to replace: {attention_class_to_replace.__name__}")

        for i, layer in enumerate(tqdm(self.model.model.layers, desc="Patching Layers")):
            if isinstance(layer.self_attn, attention_class_to_replace):
                new_attn = GSA1_LlamaAttention(
                    config=layer.self_attn.config, layer_idx=i, granu=gsa_granu,
                    local=gsa_local, prefill_weights=gsa_prefill_weights,
                    decode_weights=gsa_decode_weights, cache_type=gsa_cache_type,
                    cache_size=gsa_cache_size, vector_size=gsa_vector_size
                )
                layer.self_attn = new_attn

        # 3. Conditionally load the checkpoint
        if self.load_checkpoint:
            checkpoint_path = kwargs.get("model_name_or_path")
            if checkpoint_path is None:
                raise ValueError("Must provide 'model_name_or_path' when 'load_checkpoint' is not false.")
            
            print(f"[INFO] Loading ADAPTER weights from sharded checkpoint: {checkpoint_path}")
            load_sharded_checkpoint(self.model, checkpoint_path, strict=False)
        else:
            print("[INFO] Skipping checkpoint loading. Using randomly initialized adapter weights for baseline.")

        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self._base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        torch.manual_seed(int(kwargs.get("torch_seed", 1234)))
        print("[INFO] Model loading complete.\n")

    # --- The rest of the file is identical and correct ---
    @property
    def tokenizer_name(self):
        return self._base_model_path
    
    def apply_chat_template(self, chat_history, add_generation_prompt):
        return self.tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return self.model(inps, **kwargs).logits

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []
        for context, continuation in tqdm(requests, disable=disable_tqdm):
            context_enc = torch.tensor([context], device=self.device)
            continuation_enc = torch.tensor([continuation], device=self.device)
            full_ids = torch.cat([context_enc, continuation_enc], dim=1)
            logits = self._model_call(full_ids)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            continuation_logits = log_probs[:, context_enc.shape[1]-1:-1]
            log_likelihood = torch.gather(continuation_logits, 2, continuation_enc.unsqueeze(-1)).squeeze(-1).sum()
            res.append(log_likelihood.item())
        return res

    @torch.no_grad()
    def generate_until(self, requests, **kwargs):
        res = []
        for req in tqdm(requests, desc="Running generate_until"):
            context, gen_kwargs = req.args
            until = gen_kwargs.pop("until")
            context_ids = self.tok_encode(context, add_special_tokens=False)
            input_ids = torch.tensor([context_ids], device=self.device)
            gen_tokens = self.model.generate(input_ids=input_ids, **gen_kwargs)
            text_output = self.tok_decode(gen_tokens[0][len(context_ids):])
            for stop_seq in sorted(until, key=len, reverse=True):
                if stop_seq in text_output:
                    text_output = text_output[:text_output.find(stop_seq)]
            res.append(text_output)
        return res

    @torch.no_grad()
    def loglikelihood(self, requests, **kwargs):
        res = []
        for context, continuation in tqdm([req.args for req in requests], desc="Running loglikelihood"):
            context_ids = self.tok_encode(context, add_special_tokens=False)
            continuation_ids = self.tok_encode(continuation, add_special_tokens=False)
            full_ids = torch.tensor([context_ids + continuation_ids], device=self.device)
            logits = self._model_call(full_ids)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)
            continuation_log_probs = log_probs[len(context_ids)-1:-1].gather(1, torch.tensor(continuation_ids, device=self.device).unsqueeze(-1)).squeeze(-1)
            log_likelihood = continuation_log_probs.sum().item()
            is_greedy = torch.argmax(log_probs[len(context_ids)-1:-1], dim=-1).eq(torch.tensor(continuation_ids, device=self.device)).all()
            res.append((log_likelihood, bool(is_greedy)))
        return res

    def loglikelihood_rolling(self, requests, **kwargs):
        loglikelihoods = []
        for (string,) in tqdm([req.args for req in requests], desc="Running loglikelihood_rolling"):
            rolling_token_windows = get_rolling_token_windows(
                token_list=self.tok_encode(string),
                prefix_token=self.eot_token_id,
                max_seq_len=self.max_length,
                context_len=1,
            )
            string_loglikelihoods = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True
            )
            loglikelihoods.append(sum(string_loglikelihoods))
        return loglikelihoods

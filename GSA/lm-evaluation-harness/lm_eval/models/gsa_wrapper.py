import os
import sys
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows

# Ensure project root is importable

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../../"))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)


from gsa import GSA1_LlamaAttention

# Utility helpers

def _load_checkpoint_flexible(model, ckpt_path: str):
    """
    Robust loader for:
      - Trainer checkpoint directories
      - pytorch_model.bin
      - model.safetensors
      - sharded HF checkpoints
    """
    if os.path.isdir(ckpt_path):
        files = os.listdir(ckpt_path)

        # Safetensors
        st_files = [f for f in files if f.endswith(".safetensors")]
        if len(st_files) == 1:
            from safetensors.torch import load_file
            state = load_file(os.path.join(ckpt_path, st_files[0]))
            model.load_state_dict(state, strict=False)
            return

        # Single bin
        if "pytorch_model.bin" in files:
            state = torch.load(
                os.path.join(ckpt_path, "pytorch_model.bin"),
                map_location="cpu"
            )
            model.load_state_dict(state, strict=False)
            return

        # Sharded
        if "pytorch_model.bin.index.json" in files:
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(model, ckpt_path, strict=False)
            return

        raise FileNotFoundError(f"No model weights found in {ckpt_path}")

    else:
        # Direct file path
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path)
        else:
            state = torch.load(ckpt_path, map_location="cpu")

        model.load_state_dict(state, strict=False)

def _parse_bool(val):
    """Parse boolean arguments from lm-eval model_args."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).lower() in ("1", "true", "yes")

def _parse_list(val, cast=float):
    """Parse list arguments like '1|4|64' → [1,4,64]."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return [cast(val)]
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [cast(x) for x in val.split("|")]
    raise TypeError(f"Cannot parse list from {val}")

def _parse_2d_list(val, cast=float):
    """Parse 2D list like '0|1;1|0' → [[0,1],[1,0]]."""
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [[cast(x) for x in row.split("|")] for row in val.split(";")]
    raise TypeError(f"Cannot parse 2D list from {val}")

# Main GSA wrapper class

@register_model("gsa")
class GSAModel(LM):
    """
    lm-evaluation-harness compatible model wrapper.

    Behavior:
      - model_type=Llama → vanilla LLaMA attention
      - model_type=GSA1  → every attention layer is GSA1_LlamaAttention

    This file NEVER monkey-patches HuggingFace and never restores anything.
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Basic arguments

        self._device = torch.device(kwargs.get("device", "cuda"))
        self._base_model_path = kwargs.get("base_model_path")

        if self._base_model_path is None:
            raise ValueError("base_model_path must be provided")

        model_type = kwargs.get("model_type", "Llama")
        load_checkpoint = _parse_bool(kwargs.get("load_checkpoint", True))
        self._checkpoint_path = kwargs.get("model_name_or_path")

        # Load pretrained base model first

        print(f"[GSA] Loading base model: {self._base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self._base_model_path,
            torch_dtype=torch.float32,
        )

        # Tokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self._base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Replace attention layers ONLY IF model_type == GSA1

        if model_type == "GSA1":
            print("[GSA] Replacing LLaMA attention with GSA1 attention")

            # Parse required GSA hyperparameters
            granu = _parse_list(kwargs.get("granu"), int)
            local = _parse_list(kwargs.get("local"), float)
            vector_size = _parse_list(kwargs.get("vector_size"), int)
            base_weights = _parse_2d_list(kwargs.get("base_weights"), int)

            cache_type = kwargs.get("cache_type", "ResidualwithBlock")
            cache_size = int(kwargs.get("cache_size", 128))
            attn_size = kwargs.get("attn_size")
            attn_size = int(attn_size) if attn_size is not None else None
            dtype = torch.float16

            # Safety checks
            assert granu and local and vector_size and base_weights, \
                "All GSA parameters must be provided"

            # Explicit, layer-by-layer replacement
            for layer_idx, layer in enumerate(self.model.model.layers):
                layer.self_attn = GSA1_LlamaAttention(
                    config=self.model.config,
                    layer_idx=layer_idx,
                    granu=granu,
                    local=local,
                    base_weights=torch.tensor(base_weights, dtype=dtype),
                    cache_type=cache_type,
                    cache_size=cache_size,
                    vector_size=vector_size,
                    attn_size=attn_size,
                    dtype=dtype,
                )

        else:
            print("[GSA] Using vanilla LLaMA attention")

        # Load trained GSA checkpoint

        if model_type == "GSA1" and load_checkpoint and self._base_model_path == "meta-llama/Llama-3.2-3B":
            if self._checkpoint_path is None:
                raise ValueError("model_name_or_path required for GSA1 + checkpoint")
            print(f"[GSA] Loading checkpoint: {self._checkpoint_path}")
            load_sharded_checkpoint(self.model, self._checkpoint_path, strict=False)
        elif model_type == "GSA1" and load_checkpoint and self._base_model_path == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
            print(f"[GSA] Loading checkpoint: {self._checkpoint_path}")
            _load_checkpoint_flexible(self.model, self._checkpoint_path)
        else:
            print("[GSA] No checkpoint loaded")

        # Finalize model

        self.model.to(self._device)
        self.model.eval()

        # HARD sanity check: print attention class
        for i, layer in enumerate(self.model.model.layers[:3]):
            print(f"[DEBUG] layer {i} attention class =", type(layer.self_attn))

        print("[GSA] Model ready\n")

    # Required LM API properties

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self):
        return self._base_model_path

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    # Tokenization helpers

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)

    # Forward helper

    def _model_call(self, inps, **kwargs):
        with torch.no_grad():
            return self.model(inps, use_cache=True, **kwargs).logits

    # loglikelihood

    def loglikelihood(self, requests, **kwargs):
        results = []
        for context, continuation in tqdm(
            (r.args for r in requests),
            total=len(requests),
            desc="[GSA] loglikelihood",
            leave=False,
        ):
            ctx_ids = self.tok_encode(context, add_special_tokens=False)
            cont_ids = self.tok_encode(continuation, add_special_tokens=False)

            input_ids = torch.tensor([ctx_ids + cont_ids], device=self.device)
            logits = self._model_call(input_ids).log_softmax(dim=-1)

            cont_logits = logits[0, len(ctx_ids) - 1:-1]
            cont_ids_t = torch.tensor(cont_ids, device=self.device)

            ll = cont_logits.gather(1, cont_ids_t.unsqueeze(1)).sum().item()
            greedy = cont_logits.argmax(dim=-1).eq(cont_ids_t).all().item()
            results.append((ll, greedy))

        return results

    # loglikelihood_rolling

    def loglikelihood_rolling(self, requests, **kwargs):
        results = []

        for (text,) in tqdm(
            (r.args for r in requests),
            total=len(requests),
            desc="[GSA] loglikelihood_rolling",
            leave=False,
        ):
            tokens = self.tok_encode(text)

            windows = get_rolling_token_windows(
                token_list=tokens,
                prefix_token=self.eot_token_id,
                max_seq_len=self.max_length,
                context_len=1,
            )

            total_ll = 0.0
            for _, window in windows:
                input_ids = torch.tensor([window], device=self.device)
                logits = self.model(input_ids, use_cache=True).log_softmax(dim=-1)

                targets = input_ids[:, 1:]
                preds = logits[:, :-1, :]
                total_ll += (
                    preds.gather(dim=-1, index=targets.unsqueeze(-1))
                    .squeeze(-1)
                    .sum()
                    .item()
                )

            results.append(total_ll)

        return results

    # generate_until

    def generate_until(self, requests, **kwargs):
        outputs = []

        for context, gen_kwargs in tqdm(
            (r.args for r in requests),
            total=len(requests),
            desc="[GSA] generate",
            leave=False,
        ):
            until = gen_kwargs.pop("until")

            ctx_ids = self.tok_encode(context, add_special_tokens=False)
            input_ids = torch.tensor([ctx_ids], device=self.device)

            max_ctx = self.model.config.max_position_embeddings
            if input_ids.shape[1] > max_ctx:
                input_ids = input_ids[:, -max_ctx:]

            gen = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                do_sample=False,
                temperature=0,
                use_cache=True,
            )

            prompt_len = input_ids.shape[1]
            text = self.tok_decode(gen[0][prompt_len:])

            for stop in sorted(until, key=len, reverse=True):
                if stop in text:
                    text = text[:text.index(stop)]

            outputs.append(text)

        return outputs
    
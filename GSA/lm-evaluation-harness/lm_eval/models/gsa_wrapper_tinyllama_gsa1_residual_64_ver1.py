# lm_eval/models/gsa_wrapper.py
# ============================================================
# GSA wrapper for lm-evaluation-harness (TinyLlama compatible)
# ============================================================

import os
import sys
import torch
from tqdm import tqdm
from typing import List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows


# ------------------------------------------------------------
# Ensure gsa.py is importable
# ------------------------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../../"))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from gsa import GSA1_LlamaAttention


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _parse_bool(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).lower() in ("1", "true", "yes")


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


# ------------------------------------------------------------
# GSA Model Wrapper
# ------------------------------------------------------------
@register_model("gsa")
class GSAModel(LM):

    def __init__(self, **kwargs):
        super().__init__()

        # ----------------------------------------------------
        # Arguments
        # ----------------------------------------------------
        self._device = torch.device(kwargs.get("device", "cuda"))
        self._base_model_path = kwargs.get("base_model_path")
        self._load_checkpoint = _parse_bool(kwargs.get("load_checkpoint", True))
        self._checkpoint_path = kwargs.get("model_name_or_path")

        if self._base_model_path is None:
            raise ValueError("base_model_path must be provided")

        # ----------------------------------------------------
        # Load base model
        # ----------------------------------------------------
        print(f"[GSA] Loading base model: {self._base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self._base_model_path,
            torch_dtype=torch.float32
        )

        cfg = self.model.config

        # ----------------------------------------------------
        # Derive TinyLlama-safe attention sizes
        # ----------------------------------------------------
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = cfg.hidden_size // cfg.num_attention_heads

        gsa_vector_size = (2, 1, num_kv_heads, 0, head_dim)

        gsa_granu = [1]
        gsa_local = torch.tensor([0.0, 1.0], dtype=torch.float32)
        gsa_prefill_weights = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        gsa_decode_weights = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

        gsa_cache_type = "Residual"
        gsa_cache_size = 64

        # ----------------------------------------------------
        # Patch attention layers
        # ----------------------------------------------------
        print("[GSA] Patching attention layers...")
        attn_cls = self.model.model.layers[0].self_attn.__class__

        for i, layer in enumerate(tqdm(self.model.model.layers, desc="Patching")):
            if isinstance(layer.self_attn, attn_cls):
                layer.self_attn = GSA1_LlamaAttention(
                    config=layer.self_attn.config,
                    layer_idx=i,
                    granu=gsa_granu,
                    local=gsa_local,
                    prefill_weights=gsa_prefill_weights,
                    decode_weights=gsa_decode_weights,
                    cache_type=gsa_cache_type,
                    cache_size=gsa_cache_size,
                    vector_size=gsa_vector_size,
                )

        # ----------------------------------------------------
        # Load checkpoint (adapter weights)
        # ----------------------------------------------------
        if self._load_checkpoint:
            if self._checkpoint_path is None:
                raise ValueError("model_name_or_path must be set when load_checkpoint=True")

            print(f"[GSA] Loading checkpoint: {self._checkpoint_path}")
            _load_checkpoint_flexible(self.model, self._checkpoint_path)
        else:
            print("[GSA] Skipping checkpoint loading")

        # ----------------------------------------------------
        # Finalize
        # ----------------------------------------------------
        self.model.to(self._device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self._base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[GSA] Model ready\n")

    # --------------------------------------------------------
    # Required LM API
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------
    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)

    # --------------------------------------------------------
    # Model forward
    # --------------------------------------------------------
    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return self.model(inps, **kwargs).logits

    # --------------------------------------------------------
    # Loglikelihood
    # --------------------------------------------------------
    def loglikelihood(self, requests, **kwargs):
        results = []

        for context, continuation in [r.args for r in requests]:
            ctx_ids = self.tok_encode(context, add_special_tokens=False)
            cont_ids = self.tok_encode(continuation, add_special_tokens=False)

            input_ids = torch.tensor(
                [ctx_ids + cont_ids],
                device=self.device
            )

            logits = self._model_call(input_ids).log_softmax(dim=-1)

            cont_logits = logits[0, len(ctx_ids) - 1:-1]
            cont_ids_t = torch.tensor(cont_ids, device=self.device)

            ll = cont_logits.gather(1, cont_ids_t.unsqueeze(1)).sum().item()
            greedy = cont_logits.argmax(dim=-1).eq(cont_ids_t).all().item()

            results.append((ll, greedy))

        return results

    # --------------------------------------------------------
    # Rolling loglikelihood
    # --------------------------------------------------------
    def loglikelihood_rolling(self, requests, **kwargs):
        totals = []

        for (text,) in [r.args for r in requests]:
            windows = get_rolling_token_windows(
                token_list=self.tok_encode(text),
                prefix_token=self.eot_token_id,
                max_seq_len=self.max_length,
                context_len=1,
            )
            totals.append(sum(self._loglikelihood_tokens(windows)))

        return totals

    def _loglikelihood_tokens(self, windows):
        res = []
        for ctx, cont in windows:
            ids = torch.tensor([ctx + cont], device=self.device)
            logits = self._model_call(ids).log_softmax(dim=-1)
            res.append(
                logits[0, len(ctx)-1:-1]
                .gather(1, torch.tensor(cont, device=self.device).unsqueeze(1))
                .sum()
                .item()
            )
        return res

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------
    def generate_until(self, requests, **kwargs):
        outputs = []

        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.pop("until")

            ctx_ids = self.tok_encode(context, add_special_tokens=False)
            input_ids = torch.tensor([ctx_ids], device=self.device)

            gen = self.model.generate(input_ids=input_ids, **gen_kwargs)
            text = self.tok_decode(gen[0][len(ctx_ids):])

            for stop in sorted(until, key=len, reverse=True):
                if stop in text:
                    text = text[:text.index(stop)]

            outputs.append(text)

        return outputs
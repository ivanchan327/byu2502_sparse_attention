# streaming_llm/enable_streaming_llm.py
from streaming_llm.kv_cache import StartRecentKVCache


def enable_streaming_llm(model, start_size=4, recent_size=512):
    # Detect model type
    model_type = model.config.model_type

    if "llama" in model_type:
        # We are using our own CacheLlamaAttention → no need for pos-shift!
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_gpt_neox import enable_gpt_neox_pos_shift_attention
        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from streaming_llm.pos_shift.modify_falcon import enable_falcon_pos_shift_attention
        enable_falcon_pos_shift_attention(model)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Always return the StartRecentKVCache
    kv_cache = StartRecentKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache
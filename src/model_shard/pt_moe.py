"""Phase 7-B + 7-C-1: PyTorch MoE primitives for Gemma 4 split layers.

Mirror of moe.py. When the four functions are composed in order
(run_attention_and_route -> run_shared_expert + run_selected_experts ->
aggregate_experts), they replicate HF Gemma4TextDecoderLayer.forward's
FFN sub-block up to (but NOT including) the outer post_feedforward_layernorm
and outer residual — those are applied by ExpertOrchestrator.run_split_layer
(Task 4). The layer_scalar multiply is also orchestrator-side.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from model_shard.pytorch_engine import _resolve_layer_type, _text_model

HeatObserver = Callable[[int, int, float], None] | None


def _layer(model: Any, layer_idx: int) -> Any:
    return _text_model(model).layers[layer_idx]


def _run_one_expert(
    h: torch.Tensor, gate_up_k: torch.Tensor, down_k: torch.Tensor,
) -> torch.Tensor:
    """Per-expert gated MLP.
    h:         [..., H]
    gate_up_k: [2*I, H]
    down_k:    [H, I]
    """
    gu = F.linear(h, gate_up_k)
    g, u = gu.chunk(2, dim=-1)
    mid = F.silu(g) * u
    return F.linear(mid, down_k)


def run_attention_and_route(
    model: Any,
    h: torch.Tensor,
    layer_idx: int,
    cache: Any,
    masks: tuple[Any, Any],
    heat_observer: HeatObserver = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention sub-block + router (matches HF Gemma 4 exact sequence).

    Returns ``(post_attn_residual, top_k_index, top_k_weights)``.
    - post_attn_residual: [B, S, H]
    - top_k_index: [B*S, K]  (integer expert ids, FLAT)
    - top_k_weights: [B*S, K]
    """
    layer = _layer(model, layer_idx)
    layer_type = _resolve_layer_type(model, layer_idx)
    rotary_dict, attn_mask_dict = masks
    cos, sin = rotary_dict[layer_type]
    attention_mask = (
        attn_mask_dict.get(layer_type) if attn_mask_dict is not None else None
    )
    cache_len = cache.get_seq_length() if cache is not None else 0
    seq_len = h.shape[1]
    device = h.device
    position_ids = torch.arange(
        cache_len, cache_len + seq_len, dtype=torch.long, device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        # Attention sub-block
        residual = h
        x = layer.input_layernorm(h)
        # Gemma4TextAttention.forward requires shared_kv_states (no default);
        # empty dict is correct for non-kv-shared models.
        attn_out = layer.self_attn(
            hidden_states=x,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            shared_kv_states={},
        )
        # HF attention returns (attn_output, attn_weights_or_None)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = layer.post_attention_layernorm(attn_out)
        post_attn_residual = residual + x
        # Router on FLAT RAW post-attn-residual
        flat = post_attn_residual.reshape(-1, post_attn_residual.shape[-1])
        router_out = layer.router(flat)
        # HF returns 3-tuple (probs, weights, index); discard probs.
        _, top_k_weights, top_k_index = router_out
    if heat_observer is not None:
        idx_rows = top_k_index.tolist()
        weight_rows = top_k_weights.tolist()
        for idx_row, w_row in zip(idx_rows, weight_rows, strict=True):
            for eid, w in zip(idx_row, w_row, strict=True):
                heat_observer(layer_idx, int(eid), float(w))
    return post_attn_residual, top_k_index, top_k_weights


def run_shared_expert(
    model: Any, h: torch.Tensor, layer_idx: int,
) -> torch.Tensor:
    """Dense MLP branch: pre_feedforward_layernorm(h) -> mlp(). Does NOT
    apply post_feedforward_layernorm_1 (aggregate_experts does)."""
    layer = _layer(model, layer_idx)
    with torch.no_grad():
        normed = layer.pre_feedforward_layernorm(h)
        out = layer.mlp(normed)
    return out  # type: ignore[no-any-return]


def run_selected_experts(
    model: Any, h: torch.Tensor, layer_idx: int, expert_ids: list[int],
) -> dict[int, torch.Tensor]:
    """Per-expert MoE branch. Matches HF's flat-residual normalization:
    pre_feedforward_layernorm_2 applied to the flattened [B*S, H] post-
    attn-residual before each expert. Returns dict of per-expert outputs
    reshaped back to [B, S, H]. Does NOT apply post_feedforward_layernorm_2
    (aggregate_experts does that on the weighted sum).

    Bypasses Gemma4TextExperts.forward so the distributed engine can fan
    out expert work across owners."""
    layer = _layer(model, layer_idx)
    experts = layer.experts
    original_shape = h.shape
    with torch.no_grad():
        flat = h.reshape(-1, h.shape[-1])
        normed = layer.pre_feedforward_layernorm_2(flat)
        out: dict[int, torch.Tensor] = {}
        for k in expert_ids:
            e_out_flat = _run_one_expert(
                normed, experts.gate_up_proj[k], experts.down_proj[k],
            )
            out[int(k)] = e_out_flat.reshape(original_shape)
    return out


def aggregate_experts(
    model: Any,
    layer_idx: int,
    expert_outputs: dict[int, torch.Tensor],
    top_k_ids: list[int] | torch.Tensor,
    top_k_weights: torch.Tensor,
    shared_out: torch.Tensor,
) -> torch.Tensor:
    """Combine dense-MLP branch + MoE branch with HF's per-branch post-norms.

    - shared_out goes through post_feedforward_layernorm_1.
    - MoE weighted-sum goes through post_feedforward_layernorm_2.
    - Their sum is the "block_out" that Task 4's orchestrator then runs
      through the OUTER post_feedforward_layernorm + residual + layer_scalar.
    """
    layer = _layer(model, layer_idx)
    if isinstance(top_k_ids, torch.Tensor):
        ids_list = top_k_ids.reshape(-1).tolist()
    else:
        ids_list = list(top_k_ids)
    stacked = torch.stack([expert_outputs[int(i)] for i in ids_list], dim=0)
    w = top_k_weights.reshape(-1).view(-1, 1, 1, 1).to(stacked.dtype)
    moe_branch = (stacked * w).sum(dim=0)
    with torch.no_grad():
        dense_normed = layer.post_feedforward_layernorm_1(shared_out)
        moe_normed = layer.post_feedforward_layernorm_2(moe_branch)
    return dense_normed + moe_normed  # type: ignore[no-any-return]

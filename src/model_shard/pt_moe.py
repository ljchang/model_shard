"""Phase 7-B: PyTorch MoE primitives for Gemma 4 split layers.

Mirror of moe.py. Bypasses ``MixtralExperts.forward``'s per-expert Python loop
so the distributed engine can route per-expert work across nodes; the shape
and semantics match the HF-native path.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

HeatObserver = Callable[[int, int, float], None] | None


# ---- helpers -----------------------------------------------------------

def _layer(model: Any, layer_idx: int) -> Any:
    return model.model.layers[layer_idx]


def _run_one_expert(h: torch.Tensor, gate_up_k: torch.Tensor, down_k: torch.Tensor) -> torch.Tensor:
    """Per-expert MLP: gate+up then SiLU*gate, then down.

    h:         [B, L, H]
    gate_up_k: [2*I, H]
    down_k:    [H, I]
    returns:   [B, L, H]
    """
    gu = F.linear(h, gate_up_k)
    g, u = gu.chunk(2, dim=-1)
    mid = F.silu(g) * u
    return F.linear(mid, down_k)


# ---- public API --------------------------------------------------------

def run_attention_and_route(
    model: Any,
    h: torch.Tensor,
    layer_idx: int,
    cache: Any,
    masks: tuple[Any, Any],
    heat_observer: HeatObserver = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run self-attention + post-attention layernorm + router.

    Returns (post_attn_hidden, top_k_ids [B,L,K], top_k_weights [B,L,K]).
    ``heat_observer`` is called once per (batch, position, expert) with
    (layer_idx, expert_id, weight).
    """
    layer = _layer(model, layer_idx)
    with torch.no_grad():
        residual = h
        x = layer.input_layernorm(h)
        x = layer.self_attn(x)
        x = x + residual
        post_attn = layer.post_attention_layernorm(x)
        router_in = layer.pre_feedforward_layernorm_2(post_attn)
        top_k_ids, top_k_weights = layer.router(router_in)
    if heat_observer is not None:
        ids_flat = top_k_ids.reshape(-1, top_k_ids.shape[-1]).tolist()
        w_flat = top_k_weights.reshape(-1, top_k_weights.shape[-1]).tolist()
        for ids_row, w_row in zip(ids_flat, w_flat, strict=True):
            for eid, w in zip(ids_row, w_row, strict=True):
                heat_observer(layer_idx, int(eid), float(w))
    return post_attn, top_k_ids, top_k_weights


def run_shared_expert(model: Any, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Dense MLP path (Gemma 4's "shared expert" runs on every token)."""
    layer = _layer(model, layer_idx)
    with torch.no_grad():
        out = layer.mlp(h)
    return out  # type: ignore[no-any-return]


def run_selected_experts(
    model: Any, h: torch.Tensor, layer_idx: int, expert_ids: list[int],
) -> dict[int, torch.Tensor]:
    """Run a subset of the 128 experts; each returns [B, L, H].

    Bypasses MixtralExperts.forward's per-expert dispatch loop so the
    distributed engine can route work across nodes. Output key is the
    expert id (matches MLX convention)."""
    layer = _layer(model, layer_idx)
    experts = layer.experts
    out: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for k in expert_ids:
            out[int(k)] = _run_one_expert(
                h, experts.gate_up_proj[k], experts.down_proj[k],
            )
    return out


def aggregate_experts(
    model: Any,
    layer_idx: int,
    expert_outputs: dict[int, torch.Tensor],
    top_k_ids: list[int] | torch.Tensor,
    top_k_weights: torch.Tensor,
    shared_out: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum of per-position expert outputs plus the shared branch.

    For the synthetic path (used by tests), we assume B=1, L=1, K=len(ids_list).
    The real orchestrator flattens per-position and calls repeatedly (same
    pattern as MLX ``moe.py:aggregate_experts``).
    """
    layer = _layer(model, layer_idx)
    if isinstance(top_k_ids, torch.Tensor):
        ids_list = top_k_ids.reshape(-1).tolist()
    else:
        ids_list = list(top_k_ids)
    stacked = torch.stack([expert_outputs[int(i)] for i in ids_list], dim=0)
    w = top_k_weights.reshape(-1).view(-1, 1, 1, 1).to(stacked.dtype)
    moe_branch = (stacked * w).sum(dim=0)
    return (  # type: ignore[no-any-return]
        layer.post_feedforward_layernorm_1(shared_out)
        + layer.post_feedforward_layernorm_2(moe_branch)
    )

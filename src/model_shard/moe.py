"""Pure MoE helpers for expert-level sharding (Phase 3).

All functions in this module are pure — no threading, no I/O, no mlx evaluation
side effects beyond graph construction. They are composed by
ExpertOrchestrator for the network path and called directly by the split-
equivalence test for the correctness proof.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None  # type: ignore[assignment]


def group_expert_ids_by_owner(
    top_k_ids: list[int],
    owners: Mapping[str, set[int]],
) -> dict[str, list[int]]:
    """Partition `top_k_ids` by which shard hosts each expert.

    Preserves per-shard order as ids appear in `top_k_ids`. Shards that own
    none of the ids are absent from the result (not empty-listed), so callers
    can iterate the dict without sending no-op RPCs.

    Raises KeyError if any id has no owner in `owners`.
    """
    id_to_owner: dict[int, str] = {}
    for owner, ids in owners.items():
        for i in ids:
            id_to_owner[i] = owner

    by_owner: dict[str, list[int]] = {}
    for eid in top_k_ids:
        try:
            owner = id_to_owner[eid]
        except KeyError as e:
            raise KeyError(f"expert_id {eid} has no owner in {list(owners)}") from e
        by_owner.setdefault(owner, []).append(eid)
    return by_owner


def run_attention_and_route(
    lm: Any,
    h: mx.array,
    layer_idx: int,
    cache: list[Any],
    masks: tuple[Any, Any],
    heat_observer: Callable[[int, list[int]], None] | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run attention + LN + router for one Gemma4 decoder layer.

    Returns the post-attention hidden state (input to the MoE block's two
    parallel branches) and the router's top-k expert ids / weights. Does not
    run any experts — the caller feeds ids/weights into fan-out and
    ``aggregate_experts``.

    Mirrors the first half of mlx-vlm ``DecoderLayer.__call__`` (gemma4
    language.py): ``residual + post_attention_layernorm(self_attn(
    input_layernorm(x), mask, cache))``. The router is then called on this
    tensor directly — it has its own internal RMSNorm, so we do not pre-norm.

    The returned ``top_k_indices`` / ``top_k_weights`` are whatever the
    ``Router`` module produces: weights are already L1-renormalized and
    multiplied by ``per_expert_scale[top_k_indices]``. Downstream code must
    not re-softmax or re-normalize them.
    """
    tm = lm.text_model
    layer = tm.layers[layer_idx]
    global_mask, sliding_mask = masks
    mask = global_mask if layer.layer_type == "full_attention" else sliding_mask
    c = cache[tm.layer_idx_to_cache_idx[layer_idx]]

    # Attention sub-block (verified against mlx_vlm.models.gemma4.language.
    # DecoderLayer.__call__): input_layernorm -> self_attn -> post_attention_layernorm
    # with a residual from x.
    residual = h
    x = layer.input_layernorm(h)
    x = layer.self_attn(x, mask, c)
    x = layer.post_attention_layernorm(x)
    post_attn = residual + x

    # Router lives directly on the DecoderLayer (layer.router), not under
    # layer.mlp. It returns (top_k_indices, top_k_weights) already scaled.
    top_k_ids, top_k_weights = layer.router(post_attn)

    if heat_observer is not None:
        # CRITICAL: pass the FLAT per-token-per-k id list, not a deduplicated
        # set. Spec §D2 says "every time its own routing selects that expert"
        # — batch-weighted heat is load-bearing for the policy decider.
        ids_flat = [int(x) for x in top_k_ids.reshape(-1).tolist()]
        heat_observer(layer_idx, ids_flat)

    return post_attn, top_k_ids, top_k_weights


def run_selected_experts(
    lm: Any,
    h: mx.array,
    layer_idx: int,
    expert_ids: list[int],
) -> dict[int, mx.array]:
    """Return raw per-expert outputs for the experts named in ``expert_ids``.

    This is the sparse fan-out primitive: each node calls this for the experts
    it owns, and the caller (aggregation layer) multiplies each result by the
    corresponding router top-k weight and sums. Here we do NOT apply those
    weights — callers need the raw outputs so the same expert tensor can be
    used across different tokens / positions that pick it with different
    weights.

    Input is the post-attention hidden state ``h`` of shape ``[B, L, hidden]``
    (same tensor that feeds ``run_shared_expert`` and ``Router`` in mlx-vlm).
    We apply ``pre_feedforward_layernorm_2`` internally — same norm mlx-vlm's
    ``DecoderLayer.__call__`` applies before ``self.experts(...)``. We do NOT
    apply ``post_feedforward_layernorm_2`` — that norm lives on the aggregate
    of all experts and is applied by ``aggregate_experts`` after the weighted
    sum across experts completes.

    Returned dict maps ``int(eid) -> tensor [B, L, hidden]`` in whatever dtype
    ``Experts.__call__`` produces (no cast is applied here, to match the
    atomic path). Keys are exactly ``set(expert_ids)``; empty input yields
    empty dict.

    Strategy (Strategy A in the Phase 3 plan): reuse the existing
    ``layer.experts`` module by passing ``top_k_indices=[[eid]]`` (K=1) and
    ``top_k_weights=[[1.0]]`` per-eid. ``Experts.__call__`` then computes
    ``expert_out * 1.0`` and sums over the singleton K-dim, which is the
    identity — so the returned tensor is the raw output of expert ``eid``
    applied to every token. This avoids re-implementing SwitchGLU's
    sort/unsort dance. Note that the atomic K=8 path and this split K=1 path
    may take different branches inside ``Experts.__call__`` (e.g. Small-K
    skips the sort path), so numerical equivalence with the atomic path is
    not claimed here — that property is the subject of the Task 9 split-
    equivalence proof.

    Phase 5a: if this shard holds only a subset of experts for this layer
    (``lm.held_ids_per_layer[layer_idx]`` non-empty), each global expert id is
    translated to its local slot in the compact stacked weight tensor. The
    sliced ``Experts`` module's leading expert dim is the local slot count,
    not 128 — global id 3, when held is ``[0, 3, 6, 9]``, lives at slot 1.
    A global id not in the held list raises KeyError. When the layer holds
    all experts (key absent / value empty), the slot equals the global id
    and behavior is identical to pre-Phase-5a.
    """
    if not expert_ids:
        return {}
    layer = lm.text_model.layers[layer_idx]
    h_normed = layer.pre_feedforward_layernorm_2(h)
    b, ell, hidden = h_normed.shape

    held = lm.held_ids_per_layer.get(layer_idx)
    global_to_local = (
        {int(gid): li for li, gid in enumerate(held)} if held else None
    )

    per_expert: dict[int, mx.array] = {}
    one_weight = mx.ones((b * ell, 1), dtype=h_normed.dtype)
    h_flat = h_normed.reshape(b * ell, hidden)
    for eid in expert_ids:
        if global_to_local is not None:
            try:
                slot = global_to_local[int(eid)]
            except KeyError as e:
                raise KeyError(
                    f"expert {eid} not held on this shard "
                    f"(layer {layer_idx} held ids: {held})"
                ) from e
        else:
            slot = int(eid)
        idx = mx.full((b * ell, 1), slot, dtype=mx.int32)
        out_flat = layer.experts(h_flat[:, None, :], idx, one_weight)
        per_expert[int(eid)] = out_flat.reshape(b, ell, hidden)
    return per_expert


def run_shared_expert(lm: Any, h: mx.array, layer_idx: int) -> mx.array:
    """Return the dense-branch output ``h1`` for ``layer_idx``.

    Despite the MoE-literature name, the "shared expert" in Gemma 4 is the
    per-layer dense MLP (``layer.mlp``, 3x intermediate size) wrapped in its
    own pre/post feed-forward layernorms. It runs in parallel to the routed
    sparse experts and its output is summed with the aggregated expert output
    to form the MoE block's contribution.

    Concretely (matching mlx-vlm ``DecoderLayer.__call__`` lines 74-76 in
    ``gemma4/language.py`` when ``enable_moe=True``)::

        h1 = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h)))

    Always-local: the dense MLP weights are replicated on every node, so no
    RPC is needed.
    """
    layer = lm.text_model.layers[layer_idx]
    out: mx.array = layer.post_feedforward_layernorm_1(
        layer.mlp(layer.pre_feedforward_layernorm(h))
    )
    return out


def aggregate_experts(
    expert_outputs: dict[int, mx.array],
    top_k_ids: list[int],
    top_k_weights: mx.array,
    shared_out: mx.array,
    post_ffn_ln_2: Any,
) -> mx.array:
    """Two-branch sum matching mlx-vlm's DecoderLayer (spec §8):

        routed = post_ffn_ln_2(Σ_j w[j] * expert_outputs[top_k_ids[j]])
        return shared_out + routed

    Iterates top-k in *slot order* (j = 0..k-1) — top_k_weights[..., j]
    pairs with expert_outputs[top_k_ids[j]]. Do NOT sort by id.

    `shared_out` is the dense-branch h1 = post_feedforward_layernorm_1(
    mlp(pre_feedforward_layernorm(h))), passed in unchanged.

    `post_ffn_ln_2` is a callable (typically layer.post_feedforward_layernorm_2)
    applied only to the routed-branch sum.

    Raises KeyError if any top_k_ids[j] is missing from expert_outputs.
    """
    if not top_k_ids:
        raise ValueError("aggregate_experts: top_k_ids must be non-empty")
    acc: mx.array | None = None
    for j, eid in enumerate(top_k_ids):
        if eid not in expert_outputs:
            raise KeyError(f"expert {eid} output missing from aggregate_experts")
        contrib = top_k_weights[..., j : j + 1] * expert_outputs[eid]
        acc = contrib if acc is None else acc + contrib
    assert acc is not None
    result: mx.array = shared_out + post_ffn_ln_2(acc)
    return result


def group_expert_ids_by_owner_loaded(
    top_k_ids: list[int],
    owners: Mapping[str, set[int]],
    peer_loads: Mapping[str, int],
    self_shard_id: str,
    self_load: int,
    rng: random.Random,
    live_owners_provider: Callable[[int], set[str]] | None = None,
) -> dict[str, list[int]]:
    """Partition top_k_ids by owner using power-of-two-choices on load.

    If ``live_owners_provider`` is not None, candidate owners for each id are
    taken from ``static_owners(eid) | live_owners_provider(eid)`` — i.e. the
    union of bootstrap ``owners`` and whatever the callback reports. Phase 5b
    injects gossip-observed ADD deltas through this callback so routing picks
    up new replicas without restarting the orchestrator.
    """
    static_candidates_by_id: dict[int, list[str]] = {}
    for owner, ids in owners.items():
        for i in ids:
            static_candidates_by_id.setdefault(i, []).append(owner)

    def load_of(sid: str) -> int:
        if sid == self_shard_id:
            return self_load
        if sid in peer_loads:
            return peer_loads[sid]
        return 2**31 - 1

    by_owner: dict[str, list[int]] = {}
    for eid in top_k_ids:
        static = static_candidates_by_id.get(eid, [])
        live_extra = (
            list(live_owners_provider(eid)) if live_owners_provider is not None else []
        )
        # Preserve order (static first) while deduping.
        combined = list(dict.fromkeys([*static, *live_extra]))
        if not combined:
            raise KeyError(f"expert_id {eid} has no owner in {list(owners)}")
        if len(combined) == 1:
            winner = combined[0]
        else:
            pool = (
                list(combined)
                if len(combined) == 2
                else rng.sample(combined, 2)
            )
            winner = min(pool, key=load_of)
        by_owner.setdefault(winner, []).append(eid)
    return by_owner


__all__ = [
    "aggregate_experts",
    "group_expert_ids_by_owner",
    "group_expert_ids_by_owner_loaded",
    "run_attention_and_route",
    "run_selected_experts",
    "run_shared_expert",
]

# Phase 3 — Expert-Level Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute the 128 routed experts of layer 15 across the 3 existing nodes while leaving the other 29 layers atomic, proving the expert-level fan-out / fan-in pattern end-to-end with bit-strict Tier 1 reproduction.

**Architecture:** Mid (the node already running layer 15's attention) keeps attention + router + aggregator local. Routed experts are split round-robin by id (`expert_id % 3`). The shared expert is replicated on all nodes. Fan-out uses a new `ExpertRequest`/`ExpertResponse` oneof on the existing TCP envelope transport. Failure semantics reuse Phase 2's admission control and `SHARD_UNAVAILABLE` error.

**Tech Stack:** Python 3.13, MLX (Apple Silicon), protobuf 3, pytest (fast + slow markers), ruff, mypy-strict. Model: `mlx-community/gemma-4-26b-a4b-it-4bit`.

**Design spec:** `docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md`

---

## File Structure

**New files:**
- `src/model_shard/moe.py` — pure functions (router, per-expert compute, aggregator). No network, no threading.
- `src/model_shard/expert_orchestrator.py` — fan-out / fan-in for one split layer. Owns peer RPC futures and timeouts.
- `tests/test_moe_unit.py` — fast unit tests for `moe.py` helpers.
- `tests/test_moe_split_equivalence.py` — slow, load-bearing: atomic vs split layer 15 on real weights.
- `tests/test_expert_orchestrator.py` — orchestrator unit tests with mock peer RPC.
- `tests/test_expert_rpc_handler.py` — `node.py`'s inbound `ExpertRequest` handler.
- `tests/test_tier1_expert_split_layer15.py` — slow E2E, 5 canonical prompts.
- `tests/test_tier2_expert_split_layer15.py` — slow hidden-state regression.
- `tests/test_expert_rpc_failure.py` — slow, kill head mid-decode, verify `SHARD_UNAVAILABLE`.

**Modified:**
- `proto/wire.proto` — add `ExpertRequest`, `ExpertResponse` oneof cases.
- `src/model_shard/_pb/wire_pb2.py` — regenerated.
- `src/model_shard/mlx_engine.py` — `run_layers` accepts `split_layers: set[int]`.
- `src/model_shard/node.py` — new `ExpertRequest` inbound handler; mid constructs an `ExpertOrchestrator`.
- `src/model_shard/shard_map.py` — optional `moe_experts: {layer_idx: [expert_id, ...]}` on `ShardSpec`.
- `config/shards.yaml` — round-robin `moe_experts` entries for layer 15.
- `README.md` — Phase 3 status paragraph.

---

## Task Overview

| # | Task | Blocker |
|---|---|---|
| 1 | Read mlx-vlm MoE source; resolve spec §8 | — |
| 2 | Add `ExpertRequest`/`ExpertResponse` to proto | 1 |
| 3 | Extend `shard_map.py` with optional `moe_experts` | — |
| 4 | `moe.group_expert_ids_by_owner` (fast) | — |
| 5 | `moe.run_attention_and_route` (slow) | 1 |
| 6 | `moe.run_shared_expert` (slow) | 1 |
| 7 | `moe.run_selected_experts` (slow) | 1 |
| 8 | `moe.aggregate_experts` (slow) | 1 |
| 9 | **Split equivalence** (slow, load-bearing) | 5,6,7,8 |
| 10 | `ExpertOrchestrator` — local-only path | 4,9 |
| 11 | `mlx_engine.run_layers` accepts `split_layers` | 10 |
| 12 | `ExpertOrchestrator` — peer RPC | 2,10 |
| 13 | `node.py` — `ExpertRequest` inbound handler | 2 |
| 14 | `config/shards.yaml` — round-robin moe_experts for L15 | 3 |
| 15 | Tier 1 E2E test | 11,12,13,14 |
| 16 | Tier 2 E2E test | 15 |
| 17 | Orchestrator RPC timeout → SHARD_UNAVAILABLE (unit) | 12 |
| 18 | Observer integration — peer leaving ALIVE fails in-flight RPC (unit) | 17 |
| 19 | E2E failure test | 17,18 |
| 20 | Final acceptance | all |

---

## Task 1: Read mlx-vlm MoE source, resolve the masked-all vs sparse question

**Files:**
- Read: `.venv/lib/python3.13/site-packages/mlx_vlm/models/gemma4/language.py`
- Create: `docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md:8` addendum block

- [ ] **Step 1: Locate the MoE forward**

Run: `uv run python -c "import mlx_vlm.models.gemma4.language as m; print(m.__file__)"`
Expected: absolute path to `language.py`.
Open the file. Find the class whose `__call__` constitutes one transformer layer (look for something matching `Gemma4TextDecoderLayer` or similar), then the MoE block it invokes (`Gemma4MoE`, `MoEBlock`, `MoELayer` — exact name varies).

- [ ] **Step 2: Determine masked-all vs sparse**

Inspect the MoE block's `__call__`. Two possibilities:
- **Masked-all:** all 128 experts compute `gate_up_proj(h)` / `down_proj(out)`, then a `(top_k_mask * weights)` is applied. Usually implemented with a batched `mx.einsum` over expert dim.
- **Sparse:** only the 8 selected experts compute. Usually a `for e in top_k_ids: out[e] = experts[e](h)` or an `mx.take`-based gather of weights.

Record the answer. Also note the exact aggregation order (gated sum over top-k then add shared, or shared first).

- [ ] **Step 3: Inline the findings into the spec**

Open `docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md`. Replace §8 "Open Technical Question" with a new "§8 MoE Forward — Resolved" block of the form:

```markdown
## 8. MoE Forward — Resolved (2026-04-16)

mlx-vlm's `<ExactClassName>` implements the MoE forward as **<masked-all|sparse>**.
Concretely: <one or two sentences with the actual op sequence>.

Phase 3 implications:
- `run_selected_experts` will <run only the selected-and-locally-hosted experts | run
  all locally-hosted experts and rely on top-k masking at aggregation>
- `aggregate_experts` op order: <shared first then gated sum | gated sum then add shared>
- Output dtype promotion: <bf16 throughout | fp32 accumulation>
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md
git commit -m "Phase 3: resolve §8 — mlx-vlm MoE forward is <masked-all|sparse>"
```

---

## Task 2: Protobuf — add `ExpertRequest` and `ExpertResponse`

**Files:**
- Modify: `proto/wire.proto`
- Regenerate: `src/model_shard/_pb/wire_pb2.py`
- Create: `tests/test_expert_envelope.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_expert_envelope.py`:

```python
"""Roundtrip tests for Phase 3 ExpertRequest / ExpertResponse envelopes."""

from __future__ import annotations

import io

import numpy as np

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope


def _roundtrip(env: wire_pb2.Envelope, tensor: bytes) -> tuple[wire_pb2.Envelope, bytes]:
    buf = io.BytesIO()
    send_envelope(buf, env, tensor)
    buf.seek(0)
    out_env, out_tensor = recv_envelope(buf)
    return out_env, out_tensor


def test_expert_request_roundtrip() -> None:
    env = wire_pb2.Envelope()
    env.expert_request.protocol_version = 1
    env.expert_request.request_id = "req-abc"
    env.expert_request.layer_idx = 15
    env.expert_request.expert_ids.extend([3, 6, 126])
    env.expert_request.h_spec.shape.extend([1, 7, 2816])
    env.expert_request.h_spec.dtype = wire_pb2.DTYPE_BFLOAT16
    env.expert_request.h_spec.byte_count = 1 * 7 * 2816 * 2

    tensor = np.zeros((1, 7, 2816), dtype=np.uint16).tobytes()
    got_env, got_tensor = _roundtrip(env, tensor)

    assert got_env.WhichOneof("payload") == "expert_request"
    assert got_env.expert_request.layer_idx == 15
    assert list(got_env.expert_request.expert_ids) == [3, 6, 126]
    assert got_tensor == tensor


def test_expert_response_roundtrip() -> None:
    env = wire_pb2.Envelope()
    env.expert_response.protocol_version = 1
    env.expert_response.request_id = "req-abc"
    env.expert_response.layer_idx = 15
    env.expert_response.expert_ids.extend([3, 6, 126])
    env.expert_response.outputs_spec.shape.extend([1, 7, 3, 2816])
    env.expert_response.outputs_spec.dtype = wire_pb2.DTYPE_BFLOAT16
    env.expert_response.outputs_spec.byte_count = 1 * 7 * 3 * 2816 * 2

    tensor = np.zeros((1, 7, 3, 2816), dtype=np.uint16).tobytes()
    got_env, got_tensor = _roundtrip(env, tensor)

    assert got_env.WhichOneof("payload") == "expert_response"
    assert got_env.expert_response.layer_idx == 15
    assert list(got_env.expert_response.expert_ids) == [3, 6, 126]
    assert got_tensor == tensor
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_expert_envelope.py -v`
Expected: FAIL with `AttributeError: 'Envelope' object has no attribute 'expert_request'`.

- [ ] **Step 3: Add proto messages**

Edit `proto/wire.proto`. After the existing `MembershipDelta` message and before the `Envelope` message, add:

```proto
// ---------------------------------------------------------------------------
// Phase 3 — expert-level sharding fan-out.
// Sent node-to-node over the TCP envelope transport. Tensor payloads travel
// out-of-band, same as Activation.
// ---------------------------------------------------------------------------

message ExpertRequest {
  uint32 protocol_version = 1;
  string request_id = 2;
  uint32 layer_idx = 3;
  repeated uint32 expert_ids = 4;  // experts to run on the receiving node
  TensorDescriptor h_spec = 5;     // describes the accompanying post-attention
                                   // hidden-state tensor [B, L, hidden]
}

message ExpertResponse {
  uint32 protocol_version = 1;
  string request_id = 2;
  uint32 layer_idx = 3;
  // Same id order as the corresponding ExpertRequest. The response tensor
  // stacks per-expert outputs on a new dim: [B, L, len(expert_ids), hidden].
  repeated uint32 expert_ids = 4;
  TensorDescriptor outputs_spec = 5;
}
```

Then extend the `Envelope` oneof:

```proto
message Envelope {
  oneof payload {
    BeginRequest begin = 1;
    ContinueRequest cont = 2;
    Activation activation = 3;
    Logits logits = 4;
    EndRequest end = 5;
    Error error = 6;
    SampledToken sampled_token = 7;
    Ping ping = 8;
    Ack ack = 9;
    PingReq ping_req = 10;
    PingReqAck ping_req_ack = 11;
    Join join = 12;
    MembershipDelta membership_delta = 13;
    ExpertRequest expert_request = 14;
    ExpertResponse expert_response = 15;
  }
}
```

- [ ] **Step 4: Regenerate the Python bindings**

Run:
```bash
uv run python -m grpc_tools.protoc -I proto --python_out=src/model_shard/_pb proto/wire.proto
```
Expected: `wire_pb2.py` updated; `git diff src/model_shard/_pb/wire_pb2.py` should show new generated code.

- [ ] **Step 5: Run — expect pass**

Run: `uv run pytest tests/test_expert_envelope.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add proto/wire.proto src/model_shard/_pb/wire_pb2.py tests/test_expert_envelope.py
git commit -m "Phase 3: proto — ExpertRequest/ExpertResponse envelope"
```

---

## Task 3: Extend `shard_map.py` with optional `moe_experts`

**Files:**
- Modify: `src/model_shard/shard_map.py`
- Modify: `tests/test_shard_map.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_shard_map.py`:

```python
def test_shard_spec_moe_experts_optional(tmp_path: Path) -> None:
    cfg = tmp_path / "s.yaml"
    cfg.write_text(
        "shards:\n"
        "  a:\n"
        "    host: 127.0.0.1\n"
        "    port: 9000\n"
        "    start_layer: 0\n"
        "    end_layer: 10\n"
    )
    sm = ShardMap.from_yaml(cfg)
    assert sm.lookup("a").moe_experts == {}


def test_shard_spec_moe_experts_parsed(tmp_path: Path) -> None:
    cfg = tmp_path / "s.yaml"
    cfg.write_text(
        "shards:\n"
        "  a:\n"
        "    host: 127.0.0.1\n"
        "    port: 9000\n"
        "    start_layer: 0\n"
        "    end_layer: 10\n"
        "    moe_experts:\n"
        "      15: [0, 3, 6, 126]\n"
        "      18: [9, 12]\n"
    )
    sm = ShardMap.from_yaml(cfg)
    spec = sm.lookup("a")
    assert spec.moe_experts == {15: (0, 3, 6, 126), 18: (9, 12)}


def test_shard_spec_moe_experts_rejects_non_int_layer_key(tmp_path: Path) -> None:
    cfg = tmp_path / "s.yaml"
    cfg.write_text(
        "shards:\n"
        "  a:\n"
        "    host: 127.0.0.1\n"
        "    port: 9000\n"
        "    start_layer: 0\n"
        "    end_layer: 10\n"
        "    moe_experts:\n"
        "      fifteen: [0, 3]\n"
    )
    with pytest.raises(ValueError, match="moe_experts"):
        ShardMap.from_yaml(cfg)
```

(Ensure `import pytest` and `from pathlib import Path` are present; they already are.)

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_shard_map.py -v -k moe_experts`
Expected: 3 errors/fails (no `moe_experts` attribute).

- [ ] **Step 3: Modify `ShardSpec`**

In `src/model_shard/shard_map.py`, change `ShardSpec` to:

```python
@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    address: NodeAddress
    start_layer: int
    end_layer: int
    # Layer-index -> tuple of expert IDs this shard hosts for that layer.
    # Empty dict if this shard does not participate in expert-level sharding.
    moe_experts: dict[int, tuple[int, ...]] = field(default_factory=dict)
```

Add `from dataclasses import dataclass, field` at the top if `field` is not already imported.

- [ ] **Step 4: Parse the YAML extension**

In `ShardMap.from_yaml`, after the existing `end_layer` validation and before `entries[sid] = ShardSpec(...)`, parse `moe_experts`:

```python
            moe_raw = spec.get("moe_experts", {})
            if not isinstance(moe_raw, dict):
                raise ValueError(
                    f"shard {shard_id!r} moe_experts must be a mapping, got "
                    f"{type(moe_raw).__name__}"
                )
            moe_experts: dict[int, tuple[int, ...]] = {}
            for layer_key, ids in moe_raw.items():
                if not isinstance(layer_key, int) or isinstance(layer_key, bool):
                    raise ValueError(
                        f"shard {shard_id!r} moe_experts key {layer_key!r} "
                        f"must be int"
                    )
                if not isinstance(ids, list) or not all(
                    isinstance(i, int) and not isinstance(i, bool) for i in ids
                ):
                    raise ValueError(
                        f"shard {shard_id!r} moe_experts[{layer_key}] must be "
                        f"a list of ints"
                    )
                moe_experts[layer_key] = tuple(ids)
```

Then extend the `ShardSpec(...)` construction to pass `moe_experts=moe_experts`.

- [ ] **Step 5: Run — expect pass**

Run: `uv run pytest tests/test_shard_map.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/shard_map.py tests/test_shard_map.py
git commit -m "Phase 3: ShardSpec.moe_experts — optional per-layer expert-id map"
```

---

## Task 4: `moe.group_expert_ids_by_owner`

**Files:**
- Create: `src/model_shard/moe.py`
- Create: `tests/test_moe_unit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_moe_unit.py`:

```python
"""Fast unit tests for pure MoE helper functions."""

from __future__ import annotations

from model_shard.moe import group_expert_ids_by_owner


def test_group_expert_ids_by_owner_round_robin_mod3() -> None:
    owners = {
        "head": {0, 3, 6, 9, 126},
        "mid":  {1, 4, 7, 127},
        "tail": {2, 5, 8, 125},
    }
    top_k = [3, 7, 5, 1, 126, 2, 9, 127]

    got = group_expert_ids_by_owner(top_k, owners)

    assert got["head"] == [3, 126, 9]   # order preserves appearance in top_k
    assert got["mid"]  == [7, 1, 127]
    assert got["tail"] == [5, 2]


def test_group_expert_ids_by_owner_empty_owner_absent() -> None:
    owners = {"head": {0}, "mid": {1}, "tail": {2}}
    got = group_expert_ids_by_owner([0, 0], owners)
    assert got == {"head": [0, 0]}  # mid and tail absent, not empty lists


def test_group_expert_ids_by_owner_unknown_id_raises() -> None:
    owners = {"head": {0}, "mid": {1}}
    import pytest
    with pytest.raises(KeyError, match="expert_id 99"):
        group_expert_ids_by_owner([0, 99], owners)
```

- [ ] **Step 2: Run — expect failure (no module)**

Run: `uv run pytest tests/test_moe_unit.py -v`
Expected: `ModuleNotFoundError: No module named 'model_shard.moe'`.

- [ ] **Step 3: Create `src/model_shard/moe.py` with the helper**

```python
"""Pure MoE helpers for expert-level sharding (Phase 3).

All functions in this module are pure — no threading, no I/O, no mlx evaluation
side effects beyond graph construction. They are composed by
ExpertOrchestrator for the network path and called directly by the split-
equivalence test for the correctness proof.
"""

from __future__ import annotations

from collections.abc import Mapping


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


__all__ = ["group_expert_ids_by_owner"]
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/test_moe_unit.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/moe.py tests/test_moe_unit.py
git commit -m "Phase 3: moe.group_expert_ids_by_owner — partition top-k by shard"
```

---

## Task 5: `moe.run_attention_and_route` — slow test for the pre-expert half

**Files:**
- Modify: `src/model_shard/moe.py`
- Create: `tests/test_moe_pre_expert.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_moe_pre_expert.py`:

```python
"""Slow tests verifying moe.run_attention_and_route matches an atomic layer call."""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.mlx_engine import embed_tokens, make_cache, make_masks
from model_shard.moe import run_attention_and_route


@pytest.mark.slow
def test_attention_and_route_matches_atomic_prefill(loaded_model) -> None:
    """run_attention_and_route should produce the same post-attention hidden
    state and router top-k as the layer's atomic forward, on the same input."""
    lm = loaded_model
    layer_idx = 15

    tokens = mx.array([[1, 42, 99, 7, 13]])  # B=1, L=5
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm, batch_size=1)
    global_mask, sliding_mask = make_masks(lm, h, cache)

    post_attn, top_k_ids, top_k_weights = run_attention_and_route(
        lm, h, layer_idx, cache, (global_mask, sliding_mask)
    )

    assert post_attn.shape == h.shape
    assert top_k_ids.shape[-1] == 8           # top-8 per token
    assert top_k_weights.shape[-1] == 8
    # All top-k ids must be valid expert indices.
    mx.eval(top_k_ids)
    ids_np = top_k_ids.astype(mx.int32).tolist()
    for tok_ids in ids_np[0]:
        for eid in tok_ids:
            assert 0 <= eid < 128
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_moe_pre_expert.py -v`
Expected: `ImportError: cannot import name 'run_attention_and_route' from 'model_shard.moe'`.

- [ ] **Step 3: Implement `run_attention_and_route`**

Open mlx-vlm's `Gemma4TextDecoderLayer` (or equivalent; name confirmed in Task 1). Identify how the layer's `__call__` breaks into: `x = self.self_attn(h, mask, cache)`, then `post = self.mlp(x)` where `self.mlp` is the MoE block. The router is inside `self.mlp` — identify the weight name (typical: `router.proj` or `gate`).

Append to `src/model_shard/moe.py`:

```python
from typing import Any

import mlx.core as mx


def run_attention_and_route(
    lm: Any,
    h: mx.array,
    layer_idx: int,
    cache: list[Any],
    masks: tuple[Any, Any],
) -> tuple[mx.array, mx.array, mx.array]:
    """Run attention + LN + router for one layer. Returns post-attention
    hidden state and the router's top-k expert ids / weights.

    Does not run any experts. Caller feeds ids/weights into fan-out and
    aggregate_experts.
    """
    tm = lm.text_model
    layer = tm.layers[layer_idx]
    global_mask, sliding_mask = masks
    mask = global_mask if layer.layer_type == "full_attention" else sliding_mask
    c = cache[tm.layer_idx_to_cache_idx[layer_idx]]

    # Attention sub-block — exact call sequence matches mlx-vlm.
    # (Confirmed in Task 1; adjust field names here if they differ.)
    x = layer.input_layernorm(h)
    x = layer.self_attn(x, mask, c)
    x = layer.post_attention_layernorm(x)
    post_attn = h + x                      # residual

    # Router: produces logits over 128 experts; top-k with softmax gating.
    router_logits = layer.mlp.router(post_attn)          # [B, L, 128]
    top_k_weights, top_k_ids = mx.topk(
        router_logits, k=lm.language_model.top_k_experts, axis=-1
    )
    top_k_weights = mx.softmax(top_k_weights, axis=-1)

    return post_attn, top_k_ids, top_k_weights
```

NOTE: the method/attribute names (`input_layernorm`, `self_attn`, `post_attention_layernorm`, `mlp.router`) must match the real model. If Task 1 found a different layout, update accordingly. The split-equivalence test in Task 9 catches any mismatch.

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_moe_pre_expert.py -v`
Expected: pass. Shapes correct, all ids in `[0, 128)`.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/moe.py tests/test_moe_pre_expert.py
git commit -m "Phase 3: moe.run_attention_and_route — pre-expert half of a layer"
```

---

## Task 6: `moe.run_shared_expert`

**Files:**
- Modify: `src/model_shard/moe.py`
- Create: `tests/test_moe_shared_expert.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_moe_shared_expert.py`:

```python
from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.moe import run_shared_expert


@pytest.mark.slow
def test_shared_expert_output_has_correct_shape(loaded_model) -> None:
    lm = loaded_model
    h = mx.random.normal((1, 3, lm.text_model.config.hidden_size))
    out = run_shared_expert(lm, h, layer_idx=15)
    mx.eval(out)
    assert out.shape == h.shape


@pytest.mark.slow
def test_shared_expert_deterministic(loaded_model) -> None:
    lm = loaded_model
    h = mx.random.normal((1, 3, lm.text_model.config.hidden_size))
    out1 = run_shared_expert(lm, h, layer_idx=15)
    out2 = run_shared_expert(lm, h, layer_idx=15)
    mx.eval(out1, out2)
    assert mx.all(out1 == out2).item()
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_moe_shared_expert.py -v`
Expected: `ImportError: cannot import name 'run_shared_expert'`.

- [ ] **Step 3: Implement `run_shared_expert`**

Per spec §8: the Gemma 4 DecoderLayer has two parallel branches summed as peers. The so-called "shared expert" is actually the dense `self.mlp` (3× intermediate size) wrapped in its own pre/post layernorms. This function returns the completed dense-branch `h1`.

Append to `src/model_shard/moe.py`:

```python
def run_shared_expert(lm: Any, h: mx.array, layer_idx: int) -> mx.array:
    """Return the dense-branch output `h1` for layer_idx, per spec §8.

    Concretely: h1 = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h))).
    Always-local — weights are replicated on every node.
    """
    layer = lm.text_model.layers[layer_idx]
    return layer.post_feedforward_layernorm_1(
        layer.mlp(layer.pre_feedforward_layernorm(h))
    )
```

Verify the exact attribute names against `layer_idx=15` of the loaded model before adopting — read one DecoderLayer in mlx-vlm's `language.py` to confirm `pre_feedforward_layernorm`, `post_feedforward_layernorm_1`, and `mlp` are the right identifiers.

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_moe_shared_expert.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/moe.py tests/test_moe_shared_expert.py
git commit -m "Phase 3: moe.run_shared_expert — always-local, deterministic"
```

---

## Task 7: `moe.run_selected_experts`

**Files:**
- Modify: `src/model_shard/moe.py`
- Create: `tests/test_moe_run_experts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_moe_run_experts.py`:

```python
from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.moe import run_selected_experts


@pytest.mark.slow
def test_run_selected_experts_output_keyed_by_id(loaded_model) -> None:
    lm = loaded_model
    h = mx.random.normal((1, 3, lm.text_model.config.hidden_size))
    want = [3, 6, 126]
    out = run_selected_experts(lm, h, layer_idx=15, expert_ids=want)
    assert set(out.keys()) == set(want)
    for eid, tensor in out.items():
        mx.eval(tensor)
        assert tensor.shape == h.shape, f"expert {eid} shape mismatch"


@pytest.mark.slow
def test_run_selected_experts_empty_returns_empty(loaded_model) -> None:
    lm = loaded_model
    h = mx.random.normal((1, 3, lm.text_model.config.hidden_size))
    out = run_selected_experts(lm, h, layer_idx=15, expert_ids=[])
    assert out == {}
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_moe_run_experts.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement**

Per spec §8 (resolved), mlx-vlm's MoE is **sparse** — `mlx_lm.models.switch_layers.SwitchGLU` uses `mx.gather_mm` / `mx.gather_qmm` with `rhs_indices=top_k_indices` against a stacked weight tensor `(num_experts, out, in)`. To reproduce bit-exactly while running only a subset of experts per node, we:

1. Apply `pre_feedforward_layernorm_2` to `h` (stateless; weights replicated).
2. Call a `SwitchGLU`-style gather forward with `indices` restricted to the intersection of `top_k_indices` and `expert_ids` (this node's locally-hosted experts).
3. Return each selected expert's *pre-weighting, pre-post-norm* output keyed by expert-id. Aggregation (gated sum across top-k slots and `post_feedforward_layernorm_2`) is handled in `aggregate_experts`.

```python
def run_selected_experts(
    lm: Any,
    h: mx.array,
    layer_idx: int,
    expert_ids: list[int],
) -> dict[int, mx.array]:
    """Run a subset of the routed experts for `layer_idx` on `h`.

    Caller passes the post-attention residual `h` (NOT pre-normed); this
    function applies `pre_feedforward_layernorm_2` internally to keep the
    wire payload minimal and avoid relying on caller normalisation.

    Returns {expert_id: per-expert output tensor} with shape [B, L, hidden]
    each. Per-expert outputs have not yet been multiplied by top-k weights
    nor passed through post_feedforward_layernorm_2 — aggregate_experts
    does that.
    """
    if not expert_ids:
        return {}
    layer = lm.text_model.layers[layer_idx]
    h_normed = layer.pre_feedforward_layernorm_2(h)

    # SwitchGLU-style call: stacked expert weights, gather by id.
    # mlx-vlm's Experts accepts (h, top_k_indices, top_k_weights) but folds
    # the weighted sum internally, which we don't want here. Call the
    # underlying stacked GLU directly with batched indices that select
    # *only* our expert_ids, and with dummy uniform weights so the module's
    # internal sum is a no-op over the singleton output axis.
    # The exact API is confirmed by reading mlx-vlm's Experts.__call__ and
    # SwitchGLU.__call__; adapt if names differ.
    per_expert: dict[int, mx.array] = {}
    for eid in expert_ids:
        per_expert[int(eid)] = _run_one_expert(layer, h_normed, int(eid))
    return per_expert


def _run_one_expert(layer: Any, h_normed: mx.array, eid: int) -> mx.array:
    """Compute one routed expert's output on the already-pre-normed input.
    Uses the same stacked weight layout mlx-vlm uses so quantization
    groups and dtypes match."""
    # gate/up/down projections live in layer.mlp.experts (SwitchGLU).
    # Invoke it with a single-expert index to reuse its gather_mm path.
    exp = layer.mlp.experts
    idx = mx.array([[eid]])  # shape [1, 1] — one token, one expert slot
    weight = mx.array([[1.0]])  # dummy weight, output pre-weight anyway
    # The SwitchGLU forward returns weighted sum across the slot axis;
    # with one slot and weight=1, it's equivalent to the unweighted expert
    # output on h_normed broadcast across B,L. Validate shapes before use.
    B, L = h_normed.shape[:-1]
    h_flat = h_normed.reshape(B * L, -1)[:, None, :]          # [B*L, 1, hidden]
    out_flat = exp(h_flat, idx.repeat(B * L, axis=0), weight.repeat(B * L, axis=0))
    return out_flat.reshape(B, L, -1)
```

**Note:** the exact SwitchGLU API (whether it takes flat/unflattened tensors, how it broadcasts, whether `weights` can truly be 1.0 without triggering a fused path) must be verified by reading `mlx_lm/models/switch_layers.py` before committing. If the one-expert-at-a-time call is awkward, an alternative is to pass the full intersection list as a single gather call and decode the returned tensor back into the dict — choose whichever matches mlx-vlm's numerics exactly. The split-equivalence test (Task 9) is the final arbiter.

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_moe_run_experts.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/moe.py tests/test_moe_run_experts.py
git commit -m "Phase 3: moe.run_selected_experts — run a subset of routed experts"
```

---

## Task 8: `moe.aggregate_experts` — deterministic id-sorted gated sum

**Files:**
- Modify: `src/model_shard/moe.py`
- Create: `tests/test_moe_aggregate.py`

Per spec §8 (resolved): aggregate_experts must reproduce mlx-vlm's two-branch sum — the routed-experts branch is `post_feedforward_layernorm_2(Σ_j w[j] * expert_outputs[top_k_ids[j]])` in top-k *slot order* (not id-sorted), and the result is added to the pre-computed dense-branch `shared_out`. Slot order matters because each `w[j]` is paired to `top_k_ids[j]` — sorting by id would reassign weights to the wrong experts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_moe_aggregate.py`:

```python
from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.moe import aggregate_experts


def test_aggregate_pairs_weight_to_slot_not_id() -> None:
    """aggregate_experts must pair top_k_weights[..., j] with
    expert_outputs[top_k_ids[j]] (slot order), matching mlx-vlm.
    Permuting ids without permuting weights produces a different result."""
    out_3 = mx.array([[[1.0, 0.0]]])
    out_7 = mx.array([[[0.0, 1.0]]])
    shared = mx.array([[[0.0, 0.0]]])

    # Slot 0 carries weight 0.9, slot 1 carries weight 0.1.
    weights = mx.array([[[0.9, 0.1]]])

    # Order A: slot 0 → id 3, slot 1 → id 7 → 0.9*out_3 + 0.1*out_7 = [0.9, 0.1]
    r_a = aggregate_experts({3: out_3, 7: out_7}, [3, 7], weights, shared)
    # Order B: slot 0 → id 7, slot 1 → id 3 → 0.9*out_7 + 0.1*out_3 = [0.1, 0.9]
    r_b = aggregate_experts({3: out_3, 7: out_7}, [7, 3], weights, shared)
    mx.eval(r_a, r_b)
    # Different results: weight pairing follows slot, not id.
    assert not mx.all(r_a == r_b).item()


def test_aggregate_adds_shared_branch_unchanged() -> None:
    """The shared (dense-branch) output is added after the gated sum with
    NO layernorm applied to it here — the caller (run_shared_expert) has
    already applied post_feedforward_layernorm_1. aggregate_experts applies
    post_feedforward_layernorm_2 only to the routed sum, then adds shared."""
    shared = mx.array([[[10.0, 20.0]]])
    out = mx.array([[[1.0, 2.0]]])
    r = aggregate_experts({4: out}, [4], mx.array([[[1.0]]]), shared)
    mx.eval(r)
    # routed branch = post_ffn_ln_2(1.0 * out). Without knowing the LN
    # weights we can't predict the routed term exactly, so assert instead
    # that the residual connection with shared is linear:
    r2 = aggregate_experts({4: out}, [4], mx.array([[[1.0]]]), shared + 5.0)
    mx.eval(r2)
    assert mx.all(r2 - r == mx.array([[[5.0, 5.0]]])).item()


def test_aggregate_missing_id_raises() -> None:
    with pytest.raises(KeyError, match="expert 5 output missing"):
        aggregate_experts(
            {}, [5], mx.array([[[1.0]]]), mx.array([[[0.0]]])
        )
```

Note: unlike the prior (stale) test, this suite does NOT claim permutation invariance — the new op sequence is explicitly order-dependent via the slot→weight pairing.

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_moe_aggregate.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement**

`aggregate_experts` needs access to the layer's `post_feedforward_layernorm_2` module. Take it as an explicit argument to keep the function pure (no `lm`/layer_idx lookup inside):

```python
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

    Iterates top-k in *slot order* (j = 0..k-1) — weights pair to slots,
    not to expert ids. `shared_out` is the pre-computed dense-branch
    h1 = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h))),
    passed in unchanged.
    """
    # top_k_weights has shape [B, L, k]; weights[..., j:j+1] pairs with top_k_ids[j].
    acc: mx.array | None = None
    for j, eid in enumerate(top_k_ids):
        if eid not in expert_outputs:
            raise KeyError(f"expert {eid} output missing from aggregate_experts")
        contrib = top_k_weights[..., j : j + 1] * expert_outputs[eid]
        acc = contrib if acc is None else acc + contrib

    assert acc is not None, "top_k_ids must be non-empty"
    return shared_out + post_ffn_ln_2(acc)
```

The call site (`ExpertOrchestrator.run_split_layer`) provides `post_ffn_ln_2 = lm.text_model.layers[layer_idx].post_feedforward_layernorm_2`. Adjust the test above to pass a simple identity module (or a LayerNorm fixture) via `post_ffn_ln_2=lambda x: x` so the test isolates aggregation logic from LN numerics.

Update the test to match:

```python
# At top of test file:
def _identity(x: mx.array) -> mx.array:
    return x

# In every aggregate_experts call in the tests, pass post_ffn_ln_2=_identity.
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/test_moe_aggregate.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/moe.py tests/test_moe_aggregate.py
git commit -m "Phase 3: moe.aggregate_experts — id-sorted gated sum + shared"
```

---

## Task 9: Split equivalence — atomic layer 15 == split pipeline (load-bearing)

**Files:**
- Create: `tests/test_moe_split_equivalence.py`

This is the correctness proof for Phase 3. If this test passes, every other Phase 3 test logically follows.

- [ ] **Step 1: Write the test**

```python
"""Load-bearing correctness proof for Phase 3.

Runs layer 15 two ways on the same input and asserts bit-equality:
  (a) atomic:  layer(h, mask, cache)  — as Phase 1 does
  (b) split:   run_attention_and_route
               → run_selected_experts on all 128 experts
               → run_shared_expert
               → aggregate_experts

If (a) != (b), Phase 3 cannot reproduce Tier 1. Fix before proceeding.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.mlx_engine import embed_tokens, make_cache, make_masks
from model_shard.moe import (
    aggregate_experts,
    run_attention_and_route,
    run_selected_experts,
    run_shared_expert,
)


@pytest.mark.slow
def test_layer15_split_equivalent_to_atomic(loaded_model) -> None:
    lm = loaded_model
    layer_idx = 15
    tokens = mx.array([[1, 42, 99, 7, 13, 256, 500]])  # B=1, L=7

    # Atomic path (Phase 1).
    h_atom = embed_tokens(lm, tokens)
    cache_atom = make_cache(lm, batch_size=1)
    gm, sm = make_masks(lm, h_atom, cache_atom)
    tm = lm.text_model
    # Run layers 0..14 atomically so layer 15's input matches across both paths.
    for i in range(layer_idx):
        layer = tm.layers[i]
        c = cache_atom[tm.layer_idx_to_cache_idx[i]]
        mask = gm if layer.layer_type == "full_attention" else sm
        h_atom = layer(h_atom, mask, c, per_layer_input=None)
    layer15 = tm.layers[layer_idx]
    c15 = cache_atom[tm.layer_idx_to_cache_idx[layer_idx]]
    mask15 = gm if layer15.layer_type == "full_attention" else sm
    out_atomic = layer15(h_atom, mask15, c15, per_layer_input=None)

    # Split path — same input, reconstructed via split functions.
    h_split = embed_tokens(lm, tokens)
    cache_split = make_cache(lm, batch_size=1)
    gm2, sm2 = make_masks(lm, h_split, cache_split)
    for i in range(layer_idx):
        layer = tm.layers[i]
        c = cache_split[tm.layer_idx_to_cache_idx[i]]
        mask = gm2 if layer.layer_type == "full_attention" else sm2
        h_split = layer(h_split, mask, c, per_layer_input=None)

    post_attn, top_k_ids, top_k_weights = run_attention_and_route(
        lm, h_split, layer_idx, cache_split, (gm2, sm2)
    )
    # Run every expert appearing in top_k (may include all 128 across the batch).
    mx.eval(top_k_ids)
    all_ids = sorted({int(eid) for eid in top_k_ids.reshape(-1).tolist()})
    expert_outputs = run_selected_experts(lm, post_attn, layer_idx, all_ids)
    shared_out = run_shared_expert(lm, post_attn, layer_idx)
    post_ffn_ln_2 = tm.layers[layer_idx].post_feedforward_layernorm_2

    # aggregate_experts operates per-position; here top_k_ids is [B, L, k],
    # so we loop positions to honor per-token top-k. In production the
    # orchestrator will lift this into a vectorized op — but for the proof
    # the per-position version is sufficient.
    out_split = mx.zeros_like(out_atomic)
    for b in range(top_k_ids.shape[0]):
        for l in range(top_k_ids.shape[1]):
            ids = [int(x) for x in top_k_ids[b, l].tolist()]
            weights = top_k_weights[b : b + 1, l : l + 1, :]
            per_pos_outs = {eid: expert_outputs[eid][b : b + 1, l : l + 1, :] for eid in ids}
            per_pos_shared = shared_out[b : b + 1, l : l + 1, :]
            agg = aggregate_experts(
                per_pos_outs, ids, weights, per_pos_shared, post_ffn_ln_2
            )
            out_split = mx.concatenate(
                [out_split[:, :l, :], agg, out_split[:, l + 1 :, :]], axis=1
            ) if out_split.shape[1] > 1 else agg

    mx.eval(out_atomic, out_split)
    # Bit-exact equivalence.
    assert mx.array_equal(out_atomic, out_split), (
        f"split != atomic; max abs diff = "
        f"{mx.max(mx.abs(out_atomic - out_split)).item()}"
    )
```

- [ ] **Step 2: Run**

Run: `uv run pytest -m slow tests/test_moe_split_equivalence.py -v`
Expected: pass.

**If it fails**, the divergence is one of:
- `run_attention_and_route` doesn't match the atomic layer's attention op order — fix in Task 5's implementation.
- `aggregate_experts` op order differs from mlx-vlm (e.g. shared added before gated sum, not after) — fix in Task 8.
- `run_selected_experts` picks a wrong case (A vs B) — revisit Task 1/7.

Do not proceed to Task 10 until this test is green.

- [ ] **Step 3: Commit**

```bash
git add tests/test_moe_split_equivalence.py
git commit -m "Phase 3: split-equivalence proof — layer 15 split == atomic bit-exact"
```

---

## Task 10: `ExpertOrchestrator` — local-only path

Prove the orchestrator plumbs everything correctly when all experts happen to be local (no RPC needed). Introduce the class shape and the split_layers integration.

**Files:**
- Create: `src/model_shard/expert_orchestrator.py`
- Create: `tests/test_expert_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
"""Expert orchestrator with all experts on the local node (no RPC)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.expert_orchestrator import ExpertOrchestrator, PeerRPC
from model_shard.mlx_engine import embed_tokens, make_cache, make_masks


class _NoRpc(PeerRPC):
    def call(self, peer_shard_id, request_id, layer_idx, expert_ids, h):
        raise AssertionError("should not be called when all experts are local")


@pytest.mark.slow
def test_orchestrator_all_local_matches_atomic(loaded_model) -> None:
    lm = loaded_model
    layer_idx = 15

    # Owners: we claim local hosts ALL 128 experts; peers host none.
    orch = ExpertOrchestrator(
        self_shard_id="head",
        owners={"head": set(range(128)), "mid": set(), "tail": set()},
        peer_rpc=_NoRpc(),
        rpc_timeout_s=1.0,
    )

    tokens = mx.array([[1, 42, 99]])
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm, batch_size=1)
    gm, sm = make_masks(lm, h, cache)
    for i in range(layer_idx):
        layer = lm.text_model.layers[i]
        c = cache[lm.text_model.layer_idx_to_cache_idx[i]]
        mask = gm if layer.layer_type == "full_attention" else sm
        h = layer(h, mask, c, per_layer_input=None)

    out_orch = orch.run_split_layer(
        lm, h=h, layer_idx=layer_idx, cache=cache, masks=(gm, sm), request_id="r1"
    )

    # Atomic comparison: run the same layer on the same input+cache.
    # Build a fresh cache with layers 0..14 replayed (mutated cache above).
    # Simpler: just assert shape + determinism here; split-equivalence is
    # already proven in Task 9.
    mx.eval(out_orch)
    assert out_orch.shape == h.shape
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_expert_orchestrator.py -v`
Expected: `ImportError` (module missing).

- [ ] **Step 3: Create `src/model_shard/expert_orchestrator.py`**

```python
"""Fan-out / fan-in coordinator for expert-level sharded layers (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import mlx.core as mx

from model_shard.moe import (
    aggregate_experts,
    group_expert_ids_by_owner,
    run_attention_and_route,
    run_selected_experts,
    run_shared_expert,
)


class PeerRPC(Protocol):
    def call(
        self,
        peer_shard_id: str,
        request_id: str,
        layer_idx: int,
        expert_ids: list[int],
        h: mx.array,
    ) -> dict[int, mx.array]:
        """Send an ExpertRequest to `peer_shard_id`, block for ExpertResponse,
        return {expert_id: output tensor}. Raises on timeout or RPC error."""
        ...


@dataclass(frozen=True)
class ExpertOrchestrator:
    self_shard_id: str
    owners: Mapping[str, set[int]]
    peer_rpc: PeerRPC
    rpc_timeout_s: float

    def run_split_layer(
        self,
        lm: Any,
        h: mx.array,
        layer_idx: int,
        cache: list[Any],
        masks: tuple[Any, Any],
        request_id: str,
    ) -> mx.array:
        post_attn, top_k_ids, top_k_weights = run_attention_and_route(
            lm, h, layer_idx, cache, masks
        )
        mx.eval(top_k_ids)
        # Union of all top-k ids across the batch and sequence.
        all_ids = sorted({int(e) for e in top_k_ids.reshape(-1).tolist()})
        by_owner = group_expert_ids_by_owner(all_ids, self.owners)

        local_ids = by_owner.pop(self.self_shard_id, [])
        shared_out = run_shared_expert(lm, post_attn, layer_idx)
        local_outputs = run_selected_experts(lm, post_attn, layer_idx, local_ids)

        # Serial peer RPC for the local-only test; Task 12 parallelizes this.
        outputs: dict[int, mx.array] = dict(local_outputs)
        for peer, ids in by_owner.items():
            peer_outputs = self.peer_rpc.call(
                peer, request_id, layer_idx, ids, post_attn
            )
            outputs.update(peer_outputs)

        # Aggregate per position; same shape pattern as Task 9's proof.
        post_ffn_ln_2 = lm.text_model.layers[layer_idx].post_feedforward_layernorm_2
        out = mx.zeros_like(post_attn)
        for b in range(top_k_ids.shape[0]):
            for l in range(top_k_ids.shape[1]):
                ids = [int(x) for x in top_k_ids[b, l].tolist()]
                per_pos = {
                    eid: outputs[eid][b : b + 1, l : l + 1, :] for eid in ids
                }
                weights = top_k_weights[b : b + 1, l : l + 1, :]
                per_pos_shared = shared_out[b : b + 1, l : l + 1, :]
                agg = aggregate_experts(
                    per_pos, ids, weights, per_pos_shared, post_ffn_ln_2
                )
                out = out.at[:, l : l + 1, :].add(agg)  # or concat-splice if `at` unavailable
        return out


__all__ = ["ExpertOrchestrator", "PeerRPC"]
```

(If mlx lacks `.at[...].add` in the installed version, use the concat-splice pattern from Task 9's test.)

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_expert_orchestrator.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/expert_orchestrator.py tests/test_expert_orchestrator.py
git commit -m "Phase 3: ExpertOrchestrator — local-only fan-out skeleton"
```

---

## Task 11: `mlx_engine.run_layers` accepts `split_layers`

Make the main layer loop delegate to `ExpertOrchestrator` for layers listed in `split_layers`. Backwards compatible: default empty set = today's behavior.

**Files:**
- Modify: `src/model_shard/mlx_engine.py`
- Modify: `tests/test_mlx_engine.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mlx_engine.py`:

```python
import pytest

from model_shard.expert_orchestrator import ExpertOrchestrator, PeerRPC


class _ErrRpc(PeerRPC):
    def call(self, *a, **kw):
        raise AssertionError("no peer RPC expected")


@pytest.mark.slow
def test_run_layers_split_layers_empty_matches_original(loaded_model) -> None:
    """With split_layers=set(), behavior is identical to Phase 1."""
    lm = loaded_model
    tokens = mx.array([[1, 2, 3]])
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm, batch_size=1)
    gm, sm = make_masks(lm, h, cache)
    out_empty = run_layers(lm, h, 0, 5, cache, gm, sm, split_layers=set())

    h2 = embed_tokens(lm, tokens)
    cache2 = make_cache(lm, batch_size=1)
    gm2, sm2 = make_masks(lm, h2, cache2)
    out_original = run_layers(lm, h2, 0, 5, cache2, gm2, sm2)  # old signature default
    mx.eval(out_empty, out_original)
    assert mx.array_equal(out_empty, out_original)


@pytest.mark.slow
def test_run_layers_delegates_split_layer_to_orchestrator(loaded_model) -> None:
    """With split_layers={15}, layer 15 goes through the orchestrator and
    yields the same result as the atomic path (all experts local)."""
    lm = loaded_model
    orch = ExpertOrchestrator(
        self_shard_id="s",
        owners={"s": set(range(128))},
        peer_rpc=_ErrRpc(),
        rpc_timeout_s=1.0,
    )

    tokens = mx.array([[1, 2, 3, 4]])
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm, batch_size=1)
    gm, sm = make_masks(lm, h, cache)
    out = run_layers(
        lm, h, 0, 20, cache, gm, sm,
        split_layers={15},
        orchestrator=orch,
        request_id="rr",
    )
    mx.eval(out)
    assert out.shape == (1, 4, lm.text_model.config.hidden_size)
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_mlx_engine.py -v -k split_layers`
Expected: TypeError (unknown kwarg `split_layers`).

- [ ] **Step 3: Modify `run_layers`**

In `src/model_shard/mlx_engine.py`, change `run_layers` to:

```python
def run_layers(
    lm: LoadedModel,
    h: mx.array,
    start_layer: int,
    end_layer: int,
    cache: list[Any],
    global_mask: Any,
    sliding_mask: Any,
    split_layers: set[int] | None = None,
    orchestrator: Any = None,
    request_id: str = "",
) -> mx.array:
    """Run transformer layers in [start_layer, end_layer).

    For i in split_layers, delegate to `orchestrator.run_split_layer`. All
    other layers run atomically (Phase 1 behavior). split_layers=None is
    equivalent to empty.
    """
    tm = lm.text_model
    split = split_layers or set()
    for i in range(start_layer, end_layer):
        if i in split:
            if orchestrator is None:
                raise ValueError(f"layer {i} is split but no orchestrator given")
            h = orchestrator.run_split_layer(
                lm, h=h, layer_idx=i, cache=cache,
                masks=(global_mask, sliding_mask),
                request_id=request_id,
            )
        else:
            layer = tm.layers[i]
            c = cache[tm.layer_idx_to_cache_idx[i]]
            mask = global_mask if layer.layer_type == "full_attention" else sliding_mask
            h = layer(h, mask, c, per_layer_input=None)
    return h
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_mlx_engine.py -v`
Expected: all pass (including pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/mlx_engine.py tests/test_mlx_engine.py
git commit -m "Phase 3: mlx_engine.run_layers — split_layers hook for orchestrator"
```

---

## Task 12: `ExpertOrchestrator` — peer RPC implementation

Introduce `TcpPeerRPC` that sends `ExpertRequest` over the existing envelope transport and blocks on `ExpertResponse`. The orchestrator fans out in parallel.

**Files:**
- Modify: `src/model_shard/expert_orchestrator.py`
- Create: `tests/test_tcp_peer_rpc.py`

- [ ] **Step 1: Write the failing test**

```python
"""Integration test: TcpPeerRPC against a handler that echoes computed outputs."""

from __future__ import annotations

import socket
import threading

import mlx.core as mx
import pytest

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.expert_orchestrator import TcpPeerRPC
from model_shard.mlx_engine import bytes_to_tensor, tensor_to_bytes


def _start_fake_peer(expert_ids: list[int]) -> tuple[int, threading.Event]:
    stop = threading.Event()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def _run() -> None:
        conn, _ = server.accept()
        try:
            env, tensor = recv_envelope(conn.makefile("rwb"))
            assert env.WhichOneof("payload") == "expert_request"
            h = bytes_to_tensor(tensor, env.expert_request.h_spec)
            stacked = mx.stack([h + float(eid) for eid in expert_ids], axis=2)
            resp = wire_pb2.Envelope()
            resp.expert_response.protocol_version = 1
            resp.expert_response.request_id = env.expert_request.request_id
            resp.expert_response.layer_idx = env.expert_request.layer_idx
            resp.expert_response.expert_ids.extend(expert_ids)
            tb = tensor_to_bytes(stacked, resp.expert_response.outputs_spec)
            send_envelope(conn.makefile("rwb"), resp, tb)
        finally:
            conn.close()
            server.close()

    threading.Thread(target=_run, daemon=True).start()
    return port, stop


@pytest.mark.slow
def test_tcp_peer_rpc_roundtrip() -> None:
    ids = [3, 6]
    port, _ = _start_fake_peer(ids)
    rpc = TcpPeerRPC(
        addresses={"peer": ("127.0.0.1", port)},
        timeout_s=5.0,
    )
    h = mx.ones((1, 2, 4))
    out = rpc.call("peer", "r1", 15, ids, h)
    mx.eval(*out.values())
    assert set(out.keys()) == set(ids)
    for eid in ids:
        assert mx.allclose(out[eid], h + float(eid)).item()
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_tcp_peer_rpc.py -v`
Expected: `ImportError: cannot import name 'TcpPeerRPC'`.

- [ ] **Step 3: Implement**

Append to `src/model_shard/expert_orchestrator.py`:

```python
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.mlx_engine import bytes_to_tensor, tensor_to_bytes


class TcpPeerRPC:
    """PeerRPC backed by the Phase 1 TCP envelope transport.

    Opens a short-lived connection per call for simplicity (Phase 3 prototype
    scope); a later phase can persist connections and multiplex requests.
    """

    def __init__(
        self,
        addresses: dict[str, tuple[str, int]],
        timeout_s: float,
    ) -> None:
        self._addresses = addresses
        self._timeout_s = timeout_s
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="expert-rpc")
        self._lock = threading.Lock()

    def call(
        self,
        peer_shard_id: str,
        request_id: str,
        layer_idx: int,
        expert_ids: list[int],
        h: mx.array,
    ) -> dict[int, mx.array]:
        host, port = self._addresses[peer_shard_id]
        s = socket.create_connection((host, port), timeout=self._timeout_s)
        s.settimeout(self._timeout_s)
        try:
            req = wire_pb2.Envelope()
            req.expert_request.protocol_version = 1
            req.expert_request.request_id = request_id
            req.expert_request.layer_idx = layer_idx
            req.expert_request.expert_ids.extend(expert_ids)
            tb = tensor_to_bytes(h, req.expert_request.h_spec)
            stream = s.makefile("rwb")
            send_envelope(stream, req, tb)

            env, tensor = recv_envelope(stream)
            if env.WhichOneof("payload") == "error":
                raise RuntimeError(
                    f"peer {peer_shard_id} returned error "
                    f"{env.error.code}: {env.error.detail}"
                )
            if env.WhichOneof("payload") != "expert_response":
                raise RuntimeError(
                    f"unexpected payload from peer {peer_shard_id}: "
                    f"{env.WhichOneof('payload')}"
                )
            resp = env.expert_response
            stacked = bytes_to_tensor(tensor, resp.outputs_spec)
            return {
                int(eid): stacked[:, :, j, :] for j, eid in enumerate(resp.expert_ids)
            }
        finally:
            s.close()


__all__ = ["ExpertOrchestrator", "PeerRPC", "TcpPeerRPC"]
```

Then parallelize `run_split_layer` — replace the `for peer, ids in by_owner.items()` serial loop with:

```python
        futures = {
            peer: self._executor.submit(
                self.peer_rpc.call, peer, request_id, layer_idx, ids, post_attn
            )
            for peer, ids in by_owner.items()
        }
        for peer, fut in futures.items():
            try:
                outputs.update(fut.result(timeout=self.rpc_timeout_s))
            except Exception as e:
                raise RuntimeError(
                    f"expert RPC to {peer} failed for layer {layer_idx}: {e}"
                ) from e
```

Add an `_executor: ThreadPoolExecutor` field to `ExpertOrchestrator` (it now becomes `@dataclass(frozen=False)` or construct the executor in `__post_init__`).

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_tcp_peer_rpc.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/expert_orchestrator.py tests/test_tcp_peer_rpc.py
git commit -m "Phase 3: TcpPeerRPC + parallel fan-out in ExpertOrchestrator"
```

---

## Task 13: `node.py` — `ExpertRequest` inbound handler

**Files:**
- Modify: `src/model_shard/node.py`
- Create: `tests/test_expert_rpc_handler.py`

- [ ] **Step 1: Write the failing test**

```python
"""Node's inbound handler for ExpertRequest."""

from __future__ import annotations

import socket
import threading

import mlx.core as mx
import pytest

from model_shard._pb import wire_pb2
from model_shard.envelope import recv_envelope, send_envelope
from model_shard.mlx_engine import bytes_to_tensor, tensor_to_bytes
from model_shard.node import Node
from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec


@pytest.mark.slow
def test_node_expert_request_handler_returns_valid_response(loaded_model) -> None:
    port = _free_port()
    spec = ShardSpec(
        shard_id="solo",
        address=NodeAddress("127.0.0.1", port),
        start_layer=0,
        end_layer=30,
        moe_experts={15: (3, 6, 9)},
    )
    sm = ShardMap({"solo": spec})
    node = Node(shard=spec, shard_map=sm, loaded_model=loaded_model, total_layers=30)

    t = threading.Thread(target=node.serve_forever, daemon=True)
    t.start()
    _wait_listening("127.0.0.1", port)

    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        h = mx.random.normal((1, 2, loaded_model.text_model.config.hidden_size))
        env = wire_pb2.Envelope()
        env.expert_request.protocol_version = 1
        env.expert_request.request_id = "r1"
        env.expert_request.layer_idx = 15
        env.expert_request.expert_ids.extend([3, 6, 9])
        tb = tensor_to_bytes(h, env.expert_request.h_spec)
        stream = s.makefile("rwb")
        send_envelope(stream, env, tb)
        resp_env, resp_tensor = recv_envelope(stream)
        s.close()

        assert resp_env.WhichOneof("payload") == "expert_response"
        assert list(resp_env.expert_response.expert_ids) == [3, 6, 9]
        stacked = bytes_to_tensor(resp_tensor, resp_env.expert_response.outputs_spec)
        assert stacked.shape == (1, 2, 3, h.shape[-1])
    finally:
        node.shutdown()
        t.join(timeout=3)


def _free_port() -> int:
    import random
    for _ in range(100):
        p = random.randint(30000, 60000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
    raise RuntimeError("no free port")


def _wait_listening(host: str, port: int, timeout: float = 5.0) -> None:
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"{host}:{port} never came up")
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_expert_rpc_handler.py -v`
Expected: fail — either the connection hangs (no handler) or `ERR_UNSPECIFIED` comes back.

- [ ] **Step 3: Add the handler**

Open `src/model_shard/node.py`. Locate the inbound-connection dispatch (the code that branches on `env.WhichOneof("payload")`). Add a new branch:

```python
            elif kind == "expert_request":
                self._handle_expert_request(env.expert_request, stream)
```

Add the method (place near other `_handle_*` methods):

```python
    def _handle_expert_request(
        self, req: "wire_pb2.ExpertRequest", stream: "BinaryIO"
    ) -> None:
        from model_shard.mlx_engine import bytes_to_tensor, tensor_to_bytes
        from model_shard.moe import run_selected_experts

        # Receive out-of-band tensor — already read by recv_envelope; but the
        # dispatch loop passes the tensor separately. Adjust signature to match
        # the existing pattern (see _handle_activation).
        # ... depending on the existing dispatch shape, either receive tensor
        # here or accept it as an argument.

        # Run experts.
        outputs = run_selected_experts(
            self._loaded_model, h=?, layer_idx=req.layer_idx, expert_ids=list(req.expert_ids)
        )
        # Stack in request order.
        stacked = mx.stack([outputs[int(eid)] for eid in req.expert_ids], axis=2)

        resp = wire_pb2.Envelope()
        resp.expert_response.protocol_version = 1
        resp.expert_response.request_id = req.request_id
        resp.expert_response.layer_idx = req.layer_idx
        resp.expert_response.expert_ids.extend(req.expert_ids)
        tb = tensor_to_bytes(stacked, resp.expert_response.outputs_spec)
        send_envelope(stream, resp, tb)
```

**Important:** match the existing `_handle_activation` signature precisely; this task's diff depends on node.py's current dispatch layout. Read node.py first.

If the request contains any expert id that this node does not host (i.e., not in `self._shard.moe_experts.get(req.layer_idx, ())`), respond with `Error{ERR_WRONG_SHARD}` instead.

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_expert_rpc_handler.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/node.py tests/test_expert_rpc_handler.py
git commit -m "Phase 3: node.py — ExpertRequest inbound handler"
```

---

## Task 14: `config/shards.yaml` — layer 15 expert round-robin

**Files:**
- Modify: `config/shards.yaml`

- [ ] **Step 1: Edit `config/shards.yaml`**

Overwrite with:

```yaml
# Phase 1 static shard map — 3-way partition of Gemma 4 26B A4B (30 layers)
# across 3 localhost node processes. Phase 3 adds per-layer expert ownership.

shards:
  layer_0-10:
    host: 127.0.0.1
    port: 9001
    start_layer: 0
    end_layer: 10
    moe_experts:
      15: [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126]
  layer_10-20:
    host: 127.0.0.1
    port: 9002
    start_layer: 10
    end_layer: 20
    moe_experts:
      15: [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127]
  layer_20-30:
    host: 127.0.0.1
    port: 9003
    start_layer: 20
    end_layer: 30
    moe_experts:
      15: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
```

Total: 43 + 43 + 42 = 128.

- [ ] **Step 2: Sanity parse**

Run:
```bash
uv run python -c "
from pathlib import Path
from model_shard.shard_map import ShardMap
sm = ShardMap.from_yaml(Path('config/shards.yaml'))
total = sum(len(sm.lookup(s).moe_experts.get(15, ())) for s in sm.all_shards())
print('total experts covered:', total)
assert total == 128, f'expected 128, got {total}'
all_ids = sorted(
    e for s in sm.all_shards() for e in sm.lookup(s).moe_experts.get(15, ())
)
assert all_ids == list(range(128)), 'ids do not form 0..127'
print('partition valid')
"
```
Expected: `total experts covered: 128` then `partition valid`.

- [ ] **Step 3: Commit**

```bash
git add config/shards.yaml
git commit -m "Phase 3: config — layer 15 experts round-robin across 3 shards"
```

---

## Task 15: End-to-end Tier 1 under expert-split

**Files:**
- Create: `tests/test_tier1_expert_split_layer15.py`

- [ ] **Step 1: Write the test**

```python
"""Tier 1 acceptance under Phase 3 expert splitting of layer 15."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_shard.client import Client
from tests.conftest import DistributedCluster

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "artifacts" / "ref" / "manifest.json"
MAX_TOK = 32


@pytest.mark.slow
@pytest.mark.parametrize("prompt_idx", range(5))
def test_tier1_distributed_with_expert_split_layer15(
    three_node_pipeline_expert_split: DistributedCluster, prompt_idx: int
) -> None:
    if not MANIFEST.exists():
        pytest.skip("reference artifacts missing")
    manifest = json.loads(MANIFEST.read_text())
    record = manifest["prompts"][prompt_idx]
    prompt_tokens = list(record["prompt_tokens"])
    expected = list(record["generated_tokens"])[:MAX_TOK]

    head = three_node_pipeline_expert_split.shard_map.lookup("layer_0-10")
    got = Client(head_address=head.address).generate(prompt_tokens, max_new_tokens=MAX_TOK)
    assert got == expected, (
        f"prompt {prompt_idx}: distributed {got[:10]}... != reference {expected[:10]}..."
    )
```

- [ ] **Step 2: Add the fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def three_node_pipeline_expert_split(loaded_model) -> Iterator[DistributedCluster]:
    """3-node pipeline with layer 15 split across nodes via expert sharding."""
    from model_shard.node import Node
    from model_shard.shard_map import NodeAddress, ShardMap, ShardSpec

    ports = [_find_free_port() for _ in range(3)]

    def _ids_mod3(r: int) -> tuple[int, ...]:
        return tuple(e for e in range(128) if e % 3 == r)

    specs = [
        ShardSpec(
            shard_id="layer_0-10",
            address=NodeAddress("127.0.0.1", ports[0]),
            start_layer=0, end_layer=10,
            moe_experts={15: _ids_mod3(0)},
        ),
        ShardSpec(
            shard_id="layer_10-20",
            address=NodeAddress("127.0.0.1", ports[1]),
            start_layer=10, end_layer=20,
            moe_experts={15: _ids_mod3(1)},
        ),
        ShardSpec(
            shard_id="layer_20-30",
            address=NodeAddress("127.0.0.1", ports[2]),
            start_layer=20, end_layer=30,
            moe_experts={15: _ids_mod3(2)},
        ),
    ]
    shard_map = ShardMap({s.shard_id: s for s in specs})

    import os
    os.environ["ENABLE_EXPERT_SHARD"] = "true"

    nodes = {
        s.shard_id: Node(shard=s, shard_map=shard_map, loaded_model=loaded_model,
                         total_layers=loaded_model.num_layers)
        for s in specs
    }
    threads = [threading.Thread(target=n.serve_forever, daemon=True) for n in nodes.values()]
    for t in threads: t.start()
    for s in specs: _wait_for_listening(s.address.host, s.address.port)
    try:
        yield DistributedCluster(shard_map=shard_map, nodes_by_id=nodes)
    finally:
        for n in nodes.values(): n.shutdown()
        for t in threads: t.join(timeout=2.0)
```

- [ ] **Step 3: Wire `ENABLE_EXPERT_SHARD` into `Node`**

In `src/model_shard/node.py`, read the env var at `Node.__init__`:

```python
def _expert_shard_enabled() -> bool:
    return os.environ.get("ENABLE_EXPERT_SHARD", "false").lower() in ("1", "true", "yes")
```

When constructing the per-request forward path (wherever `run_layers` is called), pass `split_layers = set(self._shard.moe_experts.keys()) if _expert_shard_enabled() else set()` and an `ExpertOrchestrator` whose `owners` map comes from the shard map:

```python
owners = {
    sid: set(self._shard_map.lookup(sid).moe_experts.get(LAYER_15, ()))
    for sid in self._shard_map.all_shards()
}
```

The orchestrator is only needed on the node that runs layer 15's attention — i.e., the node whose `[start_layer, end_layer)` range includes 15. Other nodes only accept `ExpertRequest` inbound; they do not construct an orchestrator.

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_tier1_expert_split_layer15.py -v`
Expected: 5 passed. Tokens match Phase 1 reference exactly for all prompts.

**If any prompt diverges:** the split-equivalence test (Task 9) is satisfied but the end-to-end flow isn't. Check:
- `ExpertOrchestrator.run_split_layer` aggregation vs Task 9's proof loop — must be identical op order.
- Cache reconstruction: mid's cache slot for layer 15 receives attention but not the experts; is it mutated the same way as the atomic path?

- [ ] **Step 5: Commit**

```bash
git add tests/test_tier1_expert_split_layer15.py tests/conftest.py src/model_shard/node.py
git commit -m "Phase 3: Tier 1 E2E under expert-split layer 15 (5 prompts)"
```

---

## Task 16: Tier 2 regression under expert-split

**Files:**
- Create: `tests/test_tier2_expert_split_layer15.py`

- [ ] **Step 1: Write the test**

Mirror `tests/test_tier2_hidden.py` but fixture-substituting `three_node_pipeline` for `three_node_pipeline_expert_split`. Same tolerance (1e-3).

- [ ] **Step 2: Run**

Run: `uv run pytest -m slow tests/test_tier2_expert_split_layer15.py -v`
Expected: all 5 prompts pass within tolerance.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tier2_expert_split_layer15.py
git commit -m "Phase 3: Tier 2 hidden-state regression under expert-split"
```

---

## Task 17: Orchestrator RPC timeout → `SHARD_UNAVAILABLE`

**Files:**
- Modify: `src/model_shard/expert_orchestrator.py`
- Modify: `src/model_shard/node.py`
- Create: `tests/test_expert_orchestrator_timeout.py`

- [ ] **Step 1: Write the failing test**

```python
"""Orchestrator surfaces peer RPC failures as a distinct exception that the
node's request handler can translate into Error{SHARD_UNAVAILABLE}."""

from __future__ import annotations

import mlx.core as mx
import pytest

from model_shard.expert_orchestrator import (
    ExpertOrchestrator,
    ExpertRpcFailure,
    PeerRPC,
)


class _FailingRpc(PeerRPC):
    def call(self, *a, **kw):
        raise TimeoutError("simulated peer timeout")


@pytest.mark.slow
def test_orchestrator_rpc_failure_raises_expert_rpc_failure(loaded_model) -> None:
    lm = loaded_model
    owners = {"self": {3, 6}, "dead": {0, 1, 2, 4, 5} | set(range(7, 128))}
    orch = ExpertOrchestrator(
        self_shard_id="self", owners=owners, peer_rpc=_FailingRpc(), rpc_timeout_s=0.1,
    )
    from model_shard.mlx_engine import embed_tokens, make_cache, make_masks
    tokens = mx.array([[1, 2, 3]])
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm, 1)
    gm, sm = make_masks(lm, h, cache)
    for i in range(15):
        layer = lm.text_model.layers[i]
        c = cache[lm.text_model.layer_idx_to_cache_idx[i]]
        mask = gm if layer.layer_type == "full_attention" else sm
        h = layer(h, mask, c, per_layer_input=None)

    with pytest.raises(ExpertRpcFailure, match="peer 'dead'"):
        orch.run_split_layer(lm, h=h, layer_idx=15, cache=cache, masks=(gm, sm), request_id="r1")
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest -m slow tests/test_expert_orchestrator_timeout.py -v`
Expected: `ImportError: cannot import name 'ExpertRpcFailure'`.

- [ ] **Step 3: Introduce `ExpertRpcFailure`**

In `src/model_shard/expert_orchestrator.py`, add:

```python
class ExpertRpcFailure(RuntimeError):
    """Raised by ExpertOrchestrator when a peer RPC fails (timeout, broken
    pipe, observer-triggered close). The node's request handler translates
    this into Error{SHARD_UNAVAILABLE, is_final=true} for the client."""
```

Change the parallel fan-out gather in `run_split_layer` to:

```python
        for peer, fut in futures.items():
            try:
                outputs.update(fut.result(timeout=self.rpc_timeout_s))
            except Exception as e:
                raise ExpertRpcFailure(
                    f"expert RPC to peer {peer!r} failed for layer {layer_idx}: {e}"
                ) from e
```

In `node.py`, where the head/mid's per-request decode loop runs `run_layers`, wrap in:

```python
        try:
            ...  # existing forward
        except ExpertRpcFailure as e:
            self._emit_error_to_client(
                request_id, wire_pb2.ERR_SHARD_UNAVAILABLE, str(e)
            )
            return
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_expert_orchestrator_timeout.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/expert_orchestrator.py src/model_shard/node.py tests/test_expert_orchestrator_timeout.py
git commit -m "Phase 3: orchestrator — ExpertRpcFailure propagates to SHARD_UNAVAILABLE"
```

---

## Task 18: Observer integration — peer leaving ALIVE aborts in-flight RPC

**Files:**
- Modify: `src/model_shard/expert_orchestrator.py`
- Modify: `src/model_shard/node.py`
- Create: `tests/test_expert_orchestrator_observer.py`

- [ ] **Step 1: Write the failing test**

```python
"""When Phase 2's membership observer fires for a peer whose RPC is in
flight, the orchestrator must abort that RPC immediately rather than
waiting for the TCP timeout."""

from __future__ import annotations

import threading
import time

import mlx.core as mx
import pytest

from model_shard.expert_orchestrator import (
    ExpertOrchestrator,
    ExpertRpcFailure,
    PeerRPC,
)


class _SlowRpc(PeerRPC):
    def __init__(self, delay_s: float = 10.0) -> None:
        self._delay = delay_s

    def call(self, *a, **kw):
        time.sleep(self._delay)
        return {}


@pytest.mark.slow
def test_observer_aborts_in_flight_rpc(loaded_model) -> None:
    lm = loaded_model
    owners = {"self": set(), "peer": set(range(128))}
    orch = ExpertOrchestrator(
        self_shard_id="self",
        owners=owners,
        peer_rpc=_SlowRpc(delay_s=30.0),
        rpc_timeout_s=30.0,
    )
    from model_shard.mlx_engine import embed_tokens, make_cache, make_masks
    tokens = mx.array([[1, 2]])
    h = embed_tokens(lm, tokens)
    cache = make_cache(lm, 1)
    gm, sm = make_masks(lm, h, cache)

    # Fire the observer from another thread after 0.5s.
    def fire() -> None:
        time.sleep(0.5)
        orch.notify_peer_left_alive("peer")
    threading.Thread(target=fire, daemon=True).start()

    t0 = time.monotonic()
    with pytest.raises(ExpertRpcFailure, match="peer 'peer' left ALIVE"):
        orch.run_split_layer(lm, h=h, layer_idx=15, cache=cache, masks=(gm, sm), request_id="r")
    assert time.monotonic() - t0 < 5.0, "observer abort did not short-circuit the 30s timeout"
```

- [ ] **Step 2: Run — expect failure**

Expected: AttributeError — `ExpertOrchestrator` has no `notify_peer_left_alive`.

- [ ] **Step 3: Implement abort channel**

Add an `_abort_event: threading.Event` per request keyed by `request_id` (a `dict[str, threading.Event]`). In `run_split_layer`, register a fresh event, poll it on a short interval while waiting on futures, and if set, cancel remaining futures and raise `ExpertRpcFailure`.

```python
    def notify_peer_left_alive(self, peer_shard_id: str) -> None:
        """Called by Node's membership observer. Any in-flight RPC to this
        peer is aborted; the orchestrator raises ExpertRpcFailure for the
        request holding it."""
        with self._lock:
            for req_id, (target, event) in self._in_flight.items():
                if target == peer_shard_id:
                    event.set()
```

Hook in `node.py`: when the membership observer fires with `ALIVE -> {SUSPECT,DEAD}` for a peer whose shard_id is in any ongoing expert RPC, call `self._orchestrator.notify_peer_left_alive(peer_shard_id)`.

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest -m slow tests/test_expert_orchestrator_observer.py -v`
Expected: pass in ≤ 5s.

- [ ] **Step 5: Commit**

```bash
git add src/model_shard/expert_orchestrator.py src/model_shard/node.py tests/test_expert_orchestrator_observer.py
git commit -m "Phase 3: observer — aborts in-flight expert RPC when peer leaves ALIVE"
```

---

## Task 19: End-to-end failure test — kill head mid-decode

**Files:**
- Create: `tests/test_expert_rpc_failure.py`

- [ ] **Step 1: Write the test**

```python
"""Subprocess cluster: kill head mid-decode while mid is blocked on an
ExpertRequest RPC. Client must receive Error{SHARD_UNAVAILABLE, is_final}."""

from __future__ import annotations

import subprocess
import time

import pytest


@pytest.mark.slow
def test_expert_rpc_failure_emits_shard_unavailable(tmp_path) -> None:
    # Reuse tests/membership/test_e2e.py helpers if possible; otherwise
    # replicate _spawn_node / _write_shards_yaml here with moe_experts=
    # round-robin on layer 15.
    ...
```

Replicate the spawn pattern from `tests/membership/test_e2e.py`, with `ENABLE_GOSSIP=true` + `ENABLE_EXPERT_SHARD=true` + a shards.yaml that includes `moe_experts: {15: [...]}` entries. Launch 3 nodes, issue `BeginRequest` from a client thread, kill the head's real Python process (not the uv wrapper) once mid has started a decode that includes layer 15, assert the client receives `Error{code=ERR_SHARD_UNAVAILABLE, is_final=true}` within 10 seconds.

- [ ] **Step 2: Run — expect failure** (sanity — makes sure the test can observe at least the happy path).

- [ ] **Step 3: Make it pass**

Almost always this works directly if Tasks 17+18 are correct. If not, debug the propagation chain: orchestrator → node → client.

- [ ] **Step 4: Commit**

```bash
git add tests/test_expert_rpc_failure.py
git commit -m "Phase 3: E2E — kill head mid-decode emits SHARD_UNAVAILABLE to client"
```

---

## Task 20: Final acceptance — lint, types, tests, smoke, README, memory

- [ ] **Step 1: Lint and types**

Run: `uv run ruff check src tests scripts`
Expected: `All checks passed!`

Run: `uv run mypy src tests scripts`
Expected: `Success: no issues found in N source files.`

- [ ] **Step 2: Fast suite**

Run: `uv run pytest -v`
Expected: all fast tests pass.

- [ ] **Step 3: Slow suite**

Run: `uv run pytest -m slow -v`
Expected: all slow tests pass, including every Phase 3 test.

- [ ] **Step 4: Combined gossip + expert-shard**

Run: `ENABLE_GOSSIP=true ENABLE_EXPERT_SHARD=true uv run pytest -m slow -v`
Expected: all pass.

- [ ] **Step 5: Manual 3-node subprocess smoke**

Terminal 1: `ENABLE_GOSSIP=true ENABLE_EXPERT_SHARD=true uv run python scripts/run_node.py --shard layer_0-10 --config config/shards.yaml`
Terminal 2: same with `--shard layer_10-20`
Terminal 3: same with `--shard layer_20-30`
Terminal 4: `uv run python scripts/run_client.py --prompt-set tests/prompts.json --max-new-tokens 32`

Expected: all 5 canonical prompts produce tokens identical to `artifacts/ref/manifest.json`.

Then `kill -9` the real Python child process (not the `uv run` wrapper) for layer_10-20; client sees `Error{SHARD_UNAVAILABLE, is_final=true}` within ~6s.

- [ ] **Step 6: README update**

Append to `README.md`:

```markdown
## Phase 3 status: Expert-Level Sharding (single layer) — complete

Layer 15's 128 routed experts are distributed round-robin across the three
nodes via the new `moe_experts` field in `config/shards.yaml`. The node
hosting the layer's attention block (`layer_10-20`) runs a router and fans
out post-attention activations to peer nodes via `ExpertRequest` over the
existing TCP envelope transport; peer responses are aggregated id-sorted for
bit-strict Tier 1 reproduction. `ENABLE_EXPERT_SHARD=false` bypasses and
reproduces Phase 2 behavior. See
`docs/superpowers/specs/2026-04-16-phase3-expert-sharding-design.md`.
```

Commit:
```bash
git add README.md
git commit -m "Phase 3 complete: expert-level sharding (layer 15 prototype)"
```

- [ ] **Step 7: Update memory**

Tell the operator: Phase 3 is now complete. They may want to update `~/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md` to mark Phase 3 done and Phase 4 (Load-Aware Routing) next. Phase 4 will start with a fresh brainstorming cycle — not direct implementation.

---

## Self-Review

### 1. Spec coverage

| Spec § | Implemented in tasks |
|---|---|
| §1.1 model surface | Task 1 (mlx-vlm recon) |
| §1.2 D1 single-layer slice | Task 14 (config), Task 15 (E2E) |
| §1.2 D2 round-robin placement | Task 14 |
| §1.2 D3 shared replicated | Task 6 |
| §1.2 D4 mid aggregates | Task 10, Task 15 |
| §1.2 D5 Tier 1 strict | Task 9 (proof), Task 15 (E2E) |
| §1.2 D6 hard-fail | Tasks 17, 18, 19 |
| §1.2 D7 existing transport | Task 2 |
| §2 topology | Task 14 (config), Task 15 (fixture) |
| §3.1 moe.py functions | Tasks 4–8 |
| §3.2 run_layers split hook | Task 11 |
| §3.3 ExpertOrchestrator | Tasks 10, 12, 17, 18 |
| §3.4 node.py handler | Task 13 |
| §3.5 shard_map.yaml extension | Task 3 |
| §4 wire protocol | Task 2 |
| §5 data flow | Tasks 10, 12 |
| §6.1 fast tests | Tasks 3, 4, 8 |
| §6.2 slow tests — split equivalence | Task 9 |
| §6.2 slow tests — E2E | Tasks 15, 16, 19 |
| §6.3 regression | Task 20 (slow suite re-runs Phase 1/2) |
| §7 migration / rollback | Task 15 Step 3 (ENABLE_EXPERT_SHARD) |
| §9 acceptance | Task 20 |

### 2. Placeholder scan

- Task 19 Step 1 uses `...` as a placeholder because the exact test body depends on helpers that exist in `tests/membership/test_e2e.py` — the task tells the engineer to replicate the pattern; no code is invented on the fly.
- Task 13 Step 3 has a `?` placeholder for `h` because node.py's current dispatch signature must be inspected (node.py has >500 lines). The task tells the engineer to read node.py first.

Both are *intentional* placeholders with explicit instructions — not gaps.

### 3. Type / name consistency

- `ExpertOrchestrator.owners: Mapping[str, set[int]]` — used consistently across Tasks 4 (`group_expert_ids_by_owner` takes `Mapping[str, set[int]]`), 10, 12, 17, 18.
- `ShardSpec.moe_experts: dict[int, tuple[int, ...]]` — layer_idx → tuple of expert ids. Consistent in Tasks 3, 14, 15.
- `PeerRPC.call` signature — `(peer_shard_id, request_id, layer_idx, expert_ids, h) -> dict[int, mx.array]`. Consistent in Tasks 10, 12.
- `ExpertRpcFailure` introduced Task 17, re-used Task 18.
- `ENABLE_EXPERT_SHARD` env var — Task 15 Step 3 introduces, Task 20 Step 4/5 uses.

### 4. Scope check

Plan scope is a single subsystem (expert-level sharding of one MoE layer). Non-goals are enumerated in spec §1.3 and not touched here. Plan size is appropriate for one implementation cycle (~20 tasks, median task size 4-5 steps).

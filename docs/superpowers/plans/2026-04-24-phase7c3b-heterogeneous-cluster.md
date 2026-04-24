# Phase 7-C-3b: Heterogeneous Gossip Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a single inference cluster where shards execute on different backends — MLX on Apple Silicon and PyTorch CUDA elsewhere — all serving the same source weights (`google/gemma-4-26B-A4B-it`). Verified by Tier 1 token-exact agreement against the Phase 1 oracle on both an automated 2-subprocess pytest (Mac MLX + Mac PyTorch CPU) and a manual 3-machine deployment (Mac MLX + DGX Spark PyTorch + Ubuntu 3090 PyTorch with partial load).

**Architecture:** Both engines already serialize bf16 as raw IEEE 754 bytes — the wire format works cross-backend without bridging; we verify with a unit test rather than redesigning. SWIM gossip carries a new `model_id` field in `MemberRecordPb`; receivers reject peers with mismatched id from membership view, catching misconfigurations before they corrupt cluster output. Mac switches from gossiping a local conversion path to gossiping the canonical HF id; a small `_resolve_local_for_mlx` helper transparently maps the HF id to a local cache directory.

**Tech Stack:** Python 3.13, MLX (Apple Silicon), PyTorch (CUDA on Spark + 3090, CPU fallback on Mac for the 2-subprocess test), Protobuf for SWIM, Tailscale for cross-machine networking.

**Spec:** `docs/superpowers/specs/2026-04-24-phase7c3b-heterogeneous-cluster-design.md` (commit `070dfa8`).

---

## File Structure

**Create:**
- `tests/test_cross_backend_wire_roundtrip.py` — fast TDD test for MLX↔PyTorch tensor wire equivalence.
- `tests/test_membership_model_id_admission.py` — fast tests for cluster admission contract.
- `tests/test_resolve_local_for_mlx.py` — fast tests for the HF id → local cache resolution helper.
- `tests/test_heterogeneous_2subprocess.py` — slow heterogeneous Tier 1 pytest (2 subprocesses on Mac).
- `docs/runbooks/heterogeneous_3node.md` — manual deployment runbook for the Mac+Spark+3090 demo.
- `config/shards.heterogeneous.example.yaml` — example 3-machine config with Tailscale hostnames as placeholders.

**Modify:**
- `proto/wire.proto` — add `string model_id = 6` to `MemberRecordPb`.
- `src/model_shard/_pb/wire_pb2.py` — regenerated (do NOT hand-edit).
- `src/model_shard/membership/records.py` — `MemberRecord` dataclass gets `model_id: str` field.
- `src/model_shard/membership/messages.py` — `_record_to_pb` and `_record_from_pb` thread `model_id`.
- `src/model_shard/membership/state.py` — `MembershipState.__init__` takes `local_model_id`; `try_apply_record` validates incoming peer's `model_id`.
- `src/model_shard/membership/runner.py` — `MembershipRunner.__init__` takes `local_model_id`, passes through to state.
- `src/model_shard/node.py` — at the membership construction site (~line 1312), pass `local_model_id=self._shard_map.model_id`.
- `src/model_shard/mlx_engine.py` — add `_resolve_local_for_mlx(hf_id: str) -> str` helper; `load_model` uses it transparently.
- `config/shards.yaml` — change `model_id` from local conversion path to canonical HF id (`google/gemma-4-26B-A4B-it`).
- `tests/conftest.py` — `shards_model_id` fixture continues to read from `config/shards.yaml`; the value just changes from a path to an HF id, but the fixture body is unchanged.
- Existing membership tests where MemberRecord is constructed directly — add `model_id=""` default kwarg as needed.
- `README.md` — Phase 7-C-3b status paragraph.
- `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md` — Phase 7-C-3b COMPLETE entry.

---

## Task ordering rationale

1. **Task 1** (wire-format roundtrip) is independent and proves the most foundational claim — the bf16 wire format is already cross-backend compatible. Lands first as standalone evidence.
2. **Tasks 2-5** are the gossip extension (proto → dataclass → serialization → state validation → runner threading). Strict order because each depends on the previous.
3. **Task 6** (MLX HF-id resolver) can land before or after Tasks 2-5; placed after to cluster all the membership work together.
4. **Task 7** (2-subprocess test) requires Tasks 1, 5, 6 to be done. This is the first end-to-end heterogeneous test on local hardware.
5. **Task 8** (runbook + example yaml) is documentation — can be drafted in parallel with Task 7 but committed after.
6. **Task 9** (manual smoke verification) is a USER-ACTION gate. Pause and ask the user.
7. **Task 10** (README + memory + sweep) wraps once the manual demo is confirmed.

---

### Task 1: Cross-backend wire-format roundtrip unit test

**Files:**
- Create: `tests/test_cross_backend_wire_roundtrip.py`

This test proves that an MLX bf16 tensor serialized via `mlx_engine.tensor_to_bytes` is bit-equivalent to the same logical tensor created in PyTorch and serialized via `pytorch_engine.tensor_to_bytes`. And that each backend's `bytes_to_tensor` correctly deserializes the other backend's bytes. Catches any future regression in the wire format alignment.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cross_backend_wire_roundtrip.py`:

```python
"""Phase 7-C-3b Task 1: cross-backend wire-format roundtrip.

Both ``mlx_engine.tensor_to_bytes`` and ``pytorch_engine.tensor_to_bytes``
serialize bf16 as raw IEEE 754 bytes. This test pins that contract:
  * Same logical tensor → same bytes from both backends.
  * MLX bytes deserialize correctly via PyTorch ``bytes_to_tensor`` and
    vice versa.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from model_shard import mlx_engine, pytorch_engine  # noqa: E402
from model_shard._pb import wire_pb2  # noqa: E402


def _mlx_tensor_from_values(values: list[float]) -> mx.array:
    return mx.array(values, dtype=mx.bfloat16).reshape(1, -1)


def _torch_tensor_from_values(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bfloat16).reshape(1, -1)


def test_mlx_and_pytorch_serialize_bf16_to_same_bytes():
    """Same logical bf16 tensor → byte-identical from both backends."""
    values = [0.0, 1.0, -1.0, 0.5, -0.5, 1e-3, -1e-3, 12.34, -56.78, 100.0]
    mlx_t = _mlx_tensor_from_values(values)
    pt_t = _torch_tensor_from_values(values)
    mlx_bytes = mlx_engine.tensor_to_bytes(mlx_t)
    pt_bytes = pytorch_engine.tensor_to_bytes(pt_t)
    assert mlx_bytes == pt_bytes, (
        f"MLX bytes differ from PyTorch bytes for the same bf16 tensor; "
        f"mlx={mlx_bytes.hex()} pt={pt_bytes.hex()}"
    )


def test_mlx_bytes_deserialize_via_pytorch():
    """MLX-serialized bf16 bytes deserialize correctly with PyTorch."""
    values = [0.0, 1.0, -1.0, 12.34, -56.78]
    mlx_t = _mlx_tensor_from_values(values)
    mlx_bytes = mlx_engine.tensor_to_bytes(mlx_t)
    shape = list(mlx_t.shape)
    pt_recovered = pytorch_engine.bytes_to_tensor(
        mlx_bytes, shape, wire_pb2.DTYPE_BFLOAT16,
    )
    assert pt_recovered.dtype == torch.bfloat16
    assert list(pt_recovered.shape) == shape
    pt_expected = _torch_tensor_from_values(values)
    assert torch.equal(pt_recovered, pt_expected), (
        f"MLX bytes deserialized via PyTorch don't match expected; "
        f"got={pt_recovered} expected={pt_expected}"
    )


def test_pytorch_bytes_deserialize_via_mlx():
    """PyTorch-serialized bf16 bytes deserialize correctly with MLX."""
    values = [0.0, 1.0, -1.0, 12.34, -56.78]
    pt_t = _torch_tensor_from_values(values)
    pt_bytes = pytorch_engine.tensor_to_bytes(pt_t)
    shape = list(pt_t.shape)
    mlx_recovered = mlx_engine.bytes_to_tensor(
        pt_bytes, shape, wire_pb2.DTYPE_BFLOAT16,
    )
    assert mlx_recovered.dtype == mx.bfloat16
    assert list(mlx_recovered.shape) == shape
    mlx_expected = _mlx_tensor_from_values(values)
    # Compare via raw bytes since mx.array_equal on bf16 is straightforward.
    assert mx.array_equal(mlx_recovered, mlx_expected).item(), (
        f"PyTorch bytes deserialized via MLX don't match expected; "
        f"max abs diff = {mx.max(mx.abs(mlx_recovered - mlx_expected)).item()}"
    )


def test_full_roundtrip_mlx_to_pytorch_to_mlx():
    """MLX → bytes → PyTorch tensor → bytes → MLX tensor preserves values."""
    values = [0.0, 1.0, -1.0, 12.34, -56.78, 1e-3, -1e-3]
    mlx_orig = _mlx_tensor_from_values(values)
    shape = list(mlx_orig.shape)
    bytes_a = mlx_engine.tensor_to_bytes(mlx_orig)
    pt_intermediate = pytorch_engine.bytes_to_tensor(
        bytes_a, shape, wire_pb2.DTYPE_BFLOAT16,
    )
    bytes_b = pytorch_engine.tensor_to_bytes(pt_intermediate)
    mlx_final = mlx_engine.bytes_to_tensor(
        bytes_b, shape, wire_pb2.DTYPE_BFLOAT16,
    )
    assert mx.array_equal(mlx_orig, mlx_final).item(), (
        f"Full roundtrip MLX→PT→MLX lost values; "
        f"max abs diff = {mx.max(mx.abs(mlx_orig - mlx_final)).item()}"
    )
```

- [ ] **Step 2: Run the test — expect pass (the wire format is already aligned)**

```bash
cd /Users/lukechang/Github/model_shard
uv run pytest tests/test_cross_backend_wire_roundtrip.py -v
```

Expected: 4 passed. If any test fails, the wire format ISN'T actually aligned — that's a real bug in either `mlx_engine.tensor_to_bytes` or `pytorch_engine.tensor_to_bytes`. Stop and investigate before continuing.

- [ ] **Step 3: Ruff + mypy**

```bash
uv run ruff check tests/test_cross_backend_wire_roundtrip.py
uv run mypy tests/test_cross_backend_wire_roundtrip.py
```

Both zero errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cross_backend_wire_roundtrip.py
git commit -m "Phase 7-C-3b Task 1: cross-backend wire-format roundtrip unit tests"
```

## Context

- **Working directory:** `/Users/lukechang/Github/model_shard`
- **Branch:** `main` (project authorizes direct main commits per prior phase pattern)
- **Predecessor commit:** `070dfa8` (design spec)
- **Spec:** §3.1 (wire format)
- **Why TDD-but-expect-pass?** This is a regression-pinning test, not a feature test. The wire format is already cross-backend compatible (we believe); this test makes that belief explicit. If it fails on first run, the design assumption is wrong and we need to know immediately.

## Your Job

1. Follow Steps 1-4.
2. 4 tests pass.
3. Ruff + mypy clean.
4. Commit with the exact message.
5. If any test fails, STOP and report — don't try to "fix" the production code blindly.
6. Self-review.
7. Report back.

---

### Task 2: Add `model_id` field to `MemberRecordPb`

**Files:**
- Modify: `proto/wire.proto`
- Modify: `src/model_shard/_pb/wire_pb2.py` (regenerated, not hand-edited)

- [ ] **Step 1: Update `proto/wire.proto`**

Open `proto/wire.proto`. Find the `MemberRecordPb` definition (around line 131) and add the `model_id` field as field number 6:

```protobuf
message MemberRecordPb {
  string shard_id = 1;
  string host = 2;
  uint32 udp_port = 3;
  // 0 = alive, 1 = suspect, 2 = dead. Must match records.MemberState ordering.
  uint32 state = 4;
  uint64 incarnation = 5;
  // Phase 7-C-3b: cluster admission contract. Each node gossips its
  // model_id; receivers refuse peers with mismatched id. Empty string
  // means "not set" (legacy nodes).
  string model_id = 6;
}
```

- [ ] **Step 2: Regenerate the protobuf Python module**

```bash
cd /Users/lukechang/Github/model_shard
uv run python -m grpc_tools.protoc -I proto --python_out=src/model_shard/_pb proto/wire.proto
```

Expected: no output, exit 0. The file `src/model_shard/_pb/wire_pb2.py` is regenerated.

- [ ] **Step 3: Verify the new field exists**

```bash
uv run python -c "
from model_shard._pb import wire_pb2
r = wire_pb2.MemberRecordPb(shard_id='a', host='h', udp_port=1, state=0, incarnation=0, model_id='m')
print('model_id:', r.model_id)
print('serialized:', r.SerializeToString().hex())
"
```

Expected: `model_id: m` and a non-empty hex string.

- [ ] **Step 4: Fast suite still passes (no behavior change yet — Tasks 3-5 wire it in)**

```bash
uv run pytest -q
```

Expected: same pass count as before. If anything fails, the protoc regen broke something — investigate.

- [ ] **Step 5: Commit**

```bash
git add proto/wire.proto src/model_shard/_pb/wire_pb2.py
git commit -m "Phase 7-C-3b Task 2: add model_id to MemberRecordPb (proto + regen)"
```

## Context

- **Predecessor commit:** Task 1
- **Spec:** §3.2
- The `_pb/wire_pb2.py` file is in `[tool.ruff].extend-exclude` so ruff won't lint it. Mypy is configured with `follow_imports = "skip"` for the `_pb` module so it won't strict-check it either.
- Field number `6` is the next available after `incarnation = 5`. Confirm by grepping for "= 5;" in `MemberRecordPb` before assigning 6.

## Your Job

1. Follow Steps 1-5.
2. Field number is 6 (verify by reading the proto first).
3. Regenerated `wire_pb2.py` exposes the new field.
4. Fast suite stays green.
5. Commit.
6. Report back with the commit SHA.

---

### Task 3: Add `model_id` to `MemberRecord` dataclass + update serialization

**Files:**
- Modify: `src/model_shard/membership/records.py`
- Modify: `src/model_shard/membership/messages.py`

- [ ] **Step 1: Add `model_id` to `MemberRecord` dataclass**

Open `src/model_shard/membership/records.py`. Find the `MemberRecord` dataclass (around line 24). Add `model_id: str = ""` as a new field. Place it AFTER `incarnation` and BEFORE `last_state_change` to keep the wire-mirroring fields together:

```python
@dataclass(frozen=True)
class MemberRecord:
    shard_id: str
    host: str
    udp_port: int
    state: MemberState
    incarnation: int
    model_id: str  # Phase 7-C-3b: cluster admission contract; "" = legacy/unset
    last_state_change: float = field(compare=False)
    suspect_deadline: float | None = field(compare=False)  # set iff state == SUSPECT
```

NOTE: `model_id` does NOT have a default value here because the existing fields don't either, and a default would make it positional-only. Existing call sites must be updated to pass `model_id=""` explicitly. We update them in Step 3 (and existing tests in Step 5).

- [ ] **Step 2: Update wire serialization**

Open `src/model_shard/membership/messages.py`. Update `_record_to_pb` (around line 28) and `_record_from_pb` (around line 38):

```python
def _record_to_pb(r: MemberRecord) -> wire_pb2.MemberRecordPb:
    return wire_pb2.MemberRecordPb(
        shard_id=r.shard_id,
        host=r.host,
        udp_port=r.udp_port,
        state=int(r.state),
        incarnation=r.incarnation,
        model_id=r.model_id,
    )


def _record_from_pb(pb: wire_pb2.MemberRecordPb) -> MemberRecord:
    return MemberRecord(
        shard_id=pb.shard_id,
        host=pb.host,
        udp_port=int(pb.udp_port),
        state=MemberState(int(pb.state)),
        incarnation=int(pb.incarnation),
        model_id=str(pb.model_id),
        last_state_change=0.0,  # wire does not transport this; receiver re-stamps
        suspect_deadline=None,  # similarly, deadlines are recomputed locally
    )
```

- [ ] **Step 3: Find every direct `MemberRecord(...)` constructor call in src/**

```bash
cd /Users/lukechang/Github/model_shard
grep -rn "MemberRecord(" src/ --include="*.py" | grep -v "MemberRecordPb\|MemberRecordRecord"
```

Each match is a constructor call. Open each file and add `model_id=""` to the kwargs (or the appropriate model_id source). Specifically expect to see hits in:
- `src/model_shard/membership/state.py` — when constructing the self-record at startup. Use `self._local_model_id` (will be added in Task 4) here, but for now use `""` as a placeholder; Task 4 will plumb the real value.
- Other internal sites where MemberRecord is created when applying a transition.

For each, add `model_id=""` to the kwargs.

- [ ] **Step 4: Find every direct `MemberRecord(...)` constructor call in tests/**

```bash
grep -rn "MemberRecord(" tests/ --include="*.py" | grep -v "MemberRecordPb"
```

For each, add `model_id=""` to the kwargs. There may be many; this is mechanical.

- [ ] **Step 5: Update the dataclass roundtrip tests**

If there's an existing membership test that exercises `_record_to_pb` and `_record_from_pb` (likely in `tests/test_membership_*.py`), make sure it now also tests that `model_id` survives the roundtrip. Add a new fast test:

```python
# Add to whichever existing test file already covers _record_to_pb roundtrip
def test_record_roundtrip_preserves_model_id():
    from model_shard.membership.messages import _record_to_pb, _record_from_pb
    from model_shard.membership.records import MemberRecord, MemberState

    r = MemberRecord(
        shard_id="x", host="127.0.0.1", udp_port=9001,
        state=MemberState.ALIVE, incarnation=42,
        model_id="google/gemma-4-26B-A4B-it",
        last_state_change=0.0, suspect_deadline=None,
    )
    rt = _record_from_pb(_record_to_pb(r))
    assert rt.model_id == "google/gemma-4-26B-A4B-it"
```

If no such existing file exists, create `tests/test_member_record_roundtrip.py` with just this one test.

- [ ] **Step 6: Run fast suite — fix any leftover MemberRecord call sites**

```bash
uv run pytest -q
```

Expected: all pass. If any test errors with `TypeError: __init__() missing 1 required positional argument: 'model_id'`, you missed a call site in Step 3 or 4 — go back and add `model_id=""`.

- [ ] **Step 7: Ruff + mypy**

```bash
uv run ruff check src tests
uv run mypy src
```

Both zero errors.

- [ ] **Step 8: Commit**

```bash
git add src/model_shard/membership/records.py src/model_shard/membership/messages.py src/ tests/
git commit -m "Phase 7-C-3b Task 3: MemberRecord.model_id field + serialization"
```

## Context

- **Predecessor commit:** Task 2
- **Spec:** §3.2, §3.3
- `MemberRecord` is `frozen=True`, so adding a field doesn't break existing instance immutability.
- The field must be in a position where downstream tests can pass it as a kwarg without reordering; placing after `incarnation` keeps it grouped with the wire-mirroring fields.
- Don't add a default value for `model_id` on the dataclass — it would let downstream code accidentally drop it.

## Your Job

1. Follow Steps 1-8.
2. All MemberRecord call sites updated.
3. Fast suite all green.
4. Ruff + mypy clean.
5. Commit.

---

### Task 4: `MembershipState.try_apply_record` admission logic + fast tests

**Files:**
- Modify: `src/model_shard/membership/state.py`
- Create: `tests/test_membership_model_id_admission.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_membership_model_id_admission.py`:

```python
"""Phase 7-C-3b Task 4: cluster admission contract — model_id validation
in MembershipState.try_apply_record.

Reject peers with mismatched model_id; accept matching; reject empty
peer model_id when local has set one.
"""
from __future__ import annotations

import random

import pytest

from model_shard.membership.config import SwimConfig
from model_shard.membership.records import MemberRecord, MemberState
from model_shard.membership.state import MembershipState, PeerSpec


def _make_state(local_model_id: str = "") -> MembershipState:
    self_spec = PeerSpec(shard_id="self", host="127.0.0.1", udp_port=10001)
    peer_specs = [PeerSpec(shard_id="peer", host="127.0.0.1", udp_port=10002)]
    return MembershipState(
        self_spec=self_spec,
        peer_specs=peer_specs,
        rng=random.Random(0),
        config=SwimConfig(),
        local_model_id=local_model_id,
    )


def _peer_record(model_id: str = "", incarnation: int = 1) -> MemberRecord:
    return MemberRecord(
        shard_id="peer", host="127.0.0.1", udp_port=10002,
        state=MemberState.ALIVE, incarnation=incarnation,
        model_id=model_id,
        last_state_change=0.0, suspect_deadline=None,
    )


def test_admission_accepts_matching_model_id():
    state = _make_state(local_model_id="google/gemma-4-26B-A4B-it")
    accepted = state.try_apply_record(
        _peer_record(model_id="google/gemma-4-26B-A4B-it")
    )
    assert accepted is True


def test_admission_rejects_mismatched_model_id():
    state = _make_state(local_model_id="google/gemma-4-26B-A4B-it")
    accepted = state.try_apply_record(
        _peer_record(model_id="mlx-community/gemma-4-26b-a4b-it-4bit")
    )
    assert accepted is False


def test_admission_rejects_empty_peer_when_local_set():
    """A new node with model_id='X' rejects a peer reporting model_id=''.
    This is intentional: once the cluster is on the new contract, legacy
    peers can't silently join."""
    state = _make_state(local_model_id="google/gemma-4-26B-A4B-it")
    accepted = state.try_apply_record(_peer_record(model_id=""))
    assert accepted is False


def test_admission_accepts_when_both_empty():
    """Backwards compat: if both sides have no model_id set, admission
    is permissive (legacy behavior)."""
    state = _make_state(local_model_id="")
    accepted = state.try_apply_record(_peer_record(model_id=""))
    assert accepted is True


def test_admission_accepts_peer_when_local_empty():
    """Local hasn't set model_id → don't gate on peer's value. Useful
    during a rolling upgrade where the local node is still being
    configured."""
    state = _make_state(local_model_id="")
    accepted = state.try_apply_record(
        _peer_record(model_id="google/gemma-4-26B-A4B-it")
    )
    assert accepted is True
```

- [ ] **Step 2: Run tests — expect import errors / signature mismatches**

```bash
cd /Users/lukechang/Github/model_shard
uv run pytest tests/test_membership_model_id_admission.py -v
```

Expected: errors because `MembershipState.__init__` doesn't take `local_model_id` yet, and `try_apply_record` doesn't return a bool yet (or doesn't exist with that name). That's fine.

- [ ] **Step 3: Inspect current `MembershipState.__init__` and `try_apply_record`**

```bash
grep -nE "def __init__|def try_apply_record|def _apply_record" src/model_shard/membership/state.py | head
```

Note the current signatures. The admission logic may already exist under a different name. If `try_apply_record` doesn't exist as a public method, find the equivalent path (e.g., `_try_apply` or inline within `recv`). Adapt the tests to use the actual public API. **If you adapt the tests, document the public API name in your report so the controller knows what to expect.**

- [ ] **Step 4: Update `MembershipState.__init__` to accept `local_model_id`**

Open `src/model_shard/membership/state.py`. Modify the `__init__` signature (around line 70):

```python
class MembershipState:
    def __init__(
        self,
        self_spec: PeerSpec,
        peer_specs: list[PeerSpec],
        rng: random.Random,
        config: SwimConfig,
        local_model_id: str = "",
    ) -> None:
        self._self_id = self_spec.shard_id
        self._self_incarnation = 0
        self._cfg = config
        self._rng = rng
        self._local_model_id = local_model_id  # Phase 7-C-3b
        # ... rest unchanged ...
```

When constructing the initial self-record (around line 86-94), pass `model_id=self._local_model_id`:

```python
self._members[p.shard_id] = MemberRecord(
    shard_id=p.shard_id,
    host=p.host,
    udp_port=p.udp_port,
    state=MemberState.ALIVE,
    incarnation=0,
    model_id=self._local_model_id if p.shard_id == self._self_id else "",
    last_state_change=0.0,
    suspect_deadline=None,
)
```

- [ ] **Step 5: Add admission validation in `try_apply_record`**

Find the `try_apply_record` method (or whatever the equivalent is — see Step 3). At the very top of the function, BEFORE incarnation comparison, add:

```python
def try_apply_record(self, record: MemberRecord) -> bool:
    # Phase 7-C-3b: cluster admission contract.
    # If the local node has set model_id and the incoming record either
    # disagrees or is empty, refuse to admit.
    if self._local_model_id and record.model_id != self._local_model_id:
        log.warning(
            "rejecting peer %s with model_id mismatch: "
            "local=%r peer=%r",
            record.shard_id, self._local_model_id, record.model_id,
        )
        return False

    # ... existing incarnation comparison logic continues ...
```

The `log` import already exists at the top of the file (`logging`). Use it.

- [ ] **Step 6: Run admission tests — expect pass**

```bash
uv run pytest tests/test_membership_model_id_admission.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Run all existing membership tests — expect pass**

```bash
uv run pytest tests/test_membership_*.py tests/test_node_membership.py -q
```

Expected: all pass. If anything fails, the validation is rejecting legitimate test scenarios. Most likely cause: existing test constructs MemberState with `local_model_id=""` (default) but the test peer has a non-empty model_id from somewhere. The "local empty → accept any peer" logic in Step 4 should cover that case.

- [ ] **Step 8: Ruff + mypy**

```bash
uv run ruff check src tests
uv run mypy src
```

Both zero errors.

- [ ] **Step 9: Commit**

```bash
git add src/model_shard/membership/state.py tests/test_membership_model_id_admission.py
git commit -m "Phase 7-C-3b Task 4: MembershipState model_id admission contract"
```

## Context

- **Predecessor commit:** Task 3
- **Spec:** §3.2
- The `try_apply_record` method may have a different actual name in the codebase. Confirm via Step 3's grep; adapt tests accordingly.
- The "if local is empty, accept any peer" branch is intentional permissiveness during rolling upgrade. Once production is fully on Phase 7-C-3b, every node sets model_id and there's no permissive case.

## Your Job

1. Follow Steps 1-9.
2. 5 admission tests pass + all existing membership tests pass.
3. Ruff + mypy clean.
4. Commit.
5. Report back with the actual public method name (if it wasn't `try_apply_record`).

---

### Task 5: `MembershipRunner` + `Node.__init__` thread `model_id`

**Files:**
- Modify: `src/model_shard/membership/runner.py`
- Modify: `src/model_shard/node.py`

- [ ] **Step 1: Update `MembershipRunner.__init__`**

Open `src/model_shard/membership/runner.py`. Find the `__init__` (around line 58-65). Add `local_model_id: str = ""` as a kwarg, pass it to `MembershipState`:

```python
class MembershipRunner:
    def __init__(
        self,
        self_spec: PeerSpec,
        peers: list[PeerSpec],
        config: SwimConfig,
        rng_seed: int | None = None,
        local_model_id: str = "",
    ) -> None:
        self._cfg = config
        self._self_spec = self_spec
        self._peers = peers
        self._rng = random.Random(rng_seed)
        self._state = MembershipState(
            self_spec=self_spec,
            peer_specs=peers,
            rng=self._rng,
            config=config,
            local_model_id=local_model_id,
        )
        # ... rest unchanged ...
```

- [ ] **Step 2: Update `Node.__init__` membership construction**

Open `src/model_shard/node.py`. Find the `MembershipRunner(...)` call (around line 1312). Add `local_model_id`:

```python
runner = MembershipRunner(
    self_spec=self_spec,
    peers=peer_specs,
    config=SwimConfig(),
    local_model_id=self._shard_map.model_id,
)
```

`self._shard_map` is the `ShardMap` parameter passed into `Node.__init__` and stored on `self`. The `model_id` field was added in Phase 7-C-3a Task 2.

- [ ] **Step 3: Add an end-to-end test that the runner exposes the local model_id in its self-record**

Append to `tests/test_membership_model_id_admission.py`:

```python
def test_runner_includes_local_model_id_in_self_record():
    """MembershipRunner constructed with local_model_id surfaces it on
    the self-record visible via state.view()."""
    self_spec = PeerSpec(shard_id="self", host="127.0.0.1", udp_port=20001)
    runner = MembershipRunner(
        self_spec=self_spec,
        peers=[],
        config=SwimConfig(),
        local_model_id="google/gemma-4-26B-A4B-it",
    )
    try:
        view = runner.state.view()
        self_rec = view["self"]
        assert self_rec.model_id == "google/gemma-4-26B-A4B-it"
    finally:
        runner._transport.stop() if hasattr(runner, "_transport") else None
```

NOTE: `MembershipRunner` may not have a public `state` accessor — check via `grep -n "@property\|state" src/model_shard/membership/runner.py | head`. If not, the test can use `runner._state.view()` (private but acceptable for a unit test) or you add a public `view()` method on the runner that delegates to `self._state.view()`. Prefer the latter for cleaner test boundaries.

If you add a public `view()` method, update the test to use it.

- [ ] **Step 4: Run the new test + admission tests**

```bash
cd /Users/lukechang/Github/model_shard
uv run pytest tests/test_membership_model_id_admission.py -v
```

Expected: 6 passed (5 admission + 1 runner integration).

- [ ] **Step 5: Run all membership tests + Node tests**

```bash
uv run pytest tests/test_membership_*.py tests/test_node_*.py -q
```

Expected: all pass.

- [ ] **Step 6: Ruff + mypy**

```bash
uv run ruff check src tests
uv run mypy src
```

Both zero errors.

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/membership/runner.py src/model_shard/node.py tests/test_membership_model_id_admission.py
git commit -m "Phase 7-C-3b Task 5: thread model_id through MembershipRunner + Node"
```

## Context

- **Predecessor commit:** Task 4
- **Spec:** §3.3
- After this task, every Node started from `scripts/run_node.py` automatically gossips its `shards.yaml::model_id` and validates incoming peer ids. No CLI flag needed.

## Your Job

1. Follow Steps 1-7.
2. New test passes + all existing membership/node tests pass.
3. Ruff + mypy clean.
4. Commit.

---

### Task 6: `_resolve_local_for_mlx` helper + canonical HF id in `config/shards.yaml`

**Files:**
- Modify: `src/model_shard/mlx_engine.py`
- Modify: `config/shards.yaml`
- Create: `tests/test_resolve_local_for_mlx.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_resolve_local_for_mlx.py`:

```python
"""Phase 7-C-3b Task 6: HF id → local MLX cache resolution.

When the model_id passed to load_model is an HF repo id (e.g.
"google/gemma-4-26B-A4B-it"), the MLX backend should transparently
load from a local MLX bf16 conversion if one exists at the conventional
cache path. This lets all cluster nodes gossip the same canonical
HF id while letting MLX read locally without an HF download."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from model_shard.mlx_engine import _resolve_local_for_mlx


def test_local_path_passes_through(tmp_path: Path) -> None:
    """If the input is already a local directory path, return it unchanged."""
    (tmp_path / "config.json").write_text("{}")
    result = _resolve_local_for_mlx(str(tmp_path))
    assert result == str(tmp_path)


def test_hf_id_resolves_to_cache_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the input is an HF id and the conventional cache directory
    exists, return the cache path instead of the HF id."""
    cache_root = tmp_path / "mlx-models"
    cache_dir = cache_root / "gemma-4-26b-a4b-it-bf16"
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_text("{}")
    monkeypatch.setattr(
        "model_shard.mlx_engine._MLX_MODEL_CACHE_ROOT", cache_root,
    )
    result = _resolve_local_for_mlx("google/gemma-4-26B-A4B-it")
    assert result == str(cache_dir)


def test_hf_id_passes_through_when_cache_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the input is an HF id but no local cache exists, return the HF
    id unchanged so the caller (mlx_vlm.load) downloads from HF."""
    cache_root = tmp_path / "mlx-models"
    cache_root.mkdir(parents=True)  # exists but is empty
    monkeypatch.setattr(
        "model_shard.mlx_engine._MLX_MODEL_CACHE_ROOT", cache_root,
    )
    result = _resolve_local_for_mlx("google/gemma-4-26B-A4B-it")
    assert result == "google/gemma-4-26B-A4B-it"


def test_env_var_override_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLX_MODEL_BF16_LOCAL_PATH env var overrides cache lookup."""
    explicit = tmp_path / "explicit-path"
    explicit.mkdir()
    (explicit / "config.json").write_text("{}")
    monkeypatch.setenv("MLX_MODEL_BF16_LOCAL_PATH", str(explicit))
    result = _resolve_local_for_mlx("google/gemma-4-26B-A4B-it")
    assert result == str(explicit)
```

- [ ] **Step 2: Run test — expect failure**

```bash
cd /Users/lukechang/Github/model_shard
uv run pytest tests/test_resolve_local_for_mlx.py -v
```

Expected: ImportError because `_resolve_local_for_mlx` doesn't exist yet.

- [ ] **Step 3: Implement the helper in `mlx_engine.py`**

Open `src/model_shard/mlx_engine.py`. Add near the top (after the existing imports, before `LoadedModel`):

```python
import os as _os
from pathlib import Path as _Path

# Phase 7-C-3b: conventional cache root for MLX bf16 conversions of HF
# models. The convention is "<root>/<basename-of-hf-id>-bf16/" — e.g.,
# "~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16/" for HF id
# "google/gemma-4-26B-A4B-it". Override via MLX_MODEL_BF16_LOCAL_PATH
# env var.
_MLX_MODEL_CACHE_ROOT: _Path = _Path(
    _os.path.expanduser("~/.cache/mlx-models")
)


def _resolve_local_for_mlx(model_id: str) -> str:
    """If model_id is an HF id and a local MLX bf16 conversion exists at
    the conventional cache path, return the cache path; else return
    model_id unchanged.

    MLX_MODEL_BF16_LOCAL_PATH env var overrides the conventional path.
    Used by the cluster admission contract: all nodes gossip the same
    canonical HF id, and the MLX backend transparently loads from
    local cache when present."""
    override = _os.environ.get("MLX_MODEL_BF16_LOCAL_PATH")
    if override:
        return override
    # If the input looks like an existing local path, pass through.
    p = _Path(model_id)
    if p.exists() and p.is_dir():
        return model_id
    # If it's an HF id and we have a cache hit, return the cache path.
    basename = model_id.rsplit("/", 1)[-1].lower()
    cache_dir = _MLX_MODEL_CACHE_ROOT / f"{basename}-bf16"
    if cache_dir.exists() and cache_dir.is_dir():
        return str(cache_dir)
    return model_id
```

Update `load_model` to call the resolver:

```python
def load_model(hf_id: str) -> LoadedModel:
    from mlx_vlm import load

    resolved = _resolve_local_for_mlx(hf_id)
    model, processor = load(resolved)
    # ... rest unchanged ...
```

- [ ] **Step 4: Run resolver tests — expect pass**

```bash
uv run pytest tests/test_resolve_local_for_mlx.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Update `config/shards.yaml` to use the canonical HF id**

Open `config/shards.yaml`. Change the `model_id` line from the local conversion path to the canonical HF id:

```yaml
# Phase 7-C-3b: model_id is the canonical HF id. The MLX backend's
# _resolve_local_for_mlx() transparently maps this to a local cache
# directory (~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16/ by default,
# overridable via MLX_MODEL_BF16_LOCAL_PATH env var).
# Cluster admission requires every node gossip the SAME string here.

model_id: "google/gemma-4-26B-A4B-it"

# (rest of file unchanged — shards section remains)
```

- [ ] **Step 6: Verify the resolver works with the actual local cache path**

```bash
ls ~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16/ | head -3
uv run python -c "
from model_shard.mlx_engine import _resolve_local_for_mlx
print(_resolve_local_for_mlx('google/gemma-4-26B-A4B-it'))
"
```

Expected: prints the local cache path (`/Users/lukechang/.cache/mlx-models/gemma-4-26b-a4b-it-bf16` or wherever the user's conversion landed). If it prints the HF id instead, the cache directory naming convention doesn't match what the resolver expects — verify the directory name matches `<basename>-bf16` (case-insensitive).

- [ ] **Step 7: Run a Tier 1 slow test to confirm bf16 still loads correctly**

```bash
uv run pytest -m slow tests/test_tier1_tokens.py -v
```

Expected: 5 passed (same as Phase 7-C-3a verification). The model now loads via the resolver instead of via the literal local path, but the underlying weights are the same.

- [ ] **Step 8: Run fast suite + ruff + mypy**

```bash
uv run pytest -q
uv run ruff check src tests
uv run mypy src
```

All green.

- [ ] **Step 9: Commit**

```bash
git add src/model_shard/mlx_engine.py tests/test_resolve_local_for_mlx.py config/shards.yaml
git commit -m "Phase 7-C-3b Task 6: HF-id-canonical model_id + _resolve_local_for_mlx helper"
```

## Context

- **Predecessor commit:** Task 5
- **Spec:** §3.4
- After this task, the Mac's `config/shards.yaml::model_id` is the SAME string Spark and 3090 will use. Cluster admission compares this string directly.
- The resolver basename matching is case-insensitive (`.lower()`) because the HF id `google/gemma-4-26B-A4B-it` has mixed case but the local cache directory `gemma-4-26b-a4b-it-bf16` is conventionally lowercase.

## Your Job

1. Follow Steps 1-9.
2. Resolver test passes (4 cases).
3. Tier 1 slow test still passes after switching `config/shards.yaml`.
4. Fast + lint + types green.
5. Commit.
6. Report the actual resolved path from Step 6.

---

### Task 7: 2-subprocess heterogeneous pytest

**Files:**
- Create: `tests/test_heterogeneous_2subprocess.py`

This is the first end-to-end heterogeneous test. It spawns 2 subprocesses on localhost (one MLX, one PyTorch CPU on Mac) and runs Tier 1 prompts through the heterogeneous pipeline. Memory requirement: ≥80 GB unified (loads bf16 model twice in parallel).

- [ ] **Step 1: Write the test scaffold**

Create `tests/test_heterogeneous_2subprocess.py`:

```python
"""Phase 7-C-3b Task 7: heterogeneous Mac MLX + Mac PyTorch CPU pipeline.

Spawns two subprocesses on localhost using the existing
``scripts/run_node.py``, one with ``MODEL_SHARD_BACKEND=mlx`` and one
with ``MODEL_SHARD_BACKEND=pytorch``. Then drives a Tier 1 prompt
through the pipeline using the existing ``Client``. Asserts token-exact
match against the Phase 1 oracle.

Memory requirement: ≥80 GB unified (Mac M5 default config). PyTorch
on Mac CPU is slow on Gemma 4 26B (~minutes per token), so this test
limits to 1 prompt and a short ``max_new_tokens``. The point is
protocol correctness, not throughput.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = _REPO_ROOT / "artifacts" / "ref" / "manifest.json"
_SHARDS_YAML_TMPL = """
model_id: "google/gemma-4-26B-A4B-it"
shards:
  head:
    host: 127.0.0.1
    port: {port_head}
    start_layer: 0
    end_layer: 15
  tail:
    host: 127.0.0.1
    port: {port_tail}
    start_layer: 15
    end_layer: 30
"""

# How many tokens to generate. Lower than the standard Tier 1 (64) because
# PyTorch on Mac CPU is glacial on Gemma 4 26B.
_MAX_NEW_TOKENS = 4

# Which prompt index from artifacts/ref/manifest.json to test.
_PROMPT_IDX = 0


def _free_port() -> int:
    # Membership UDP is tcp+1000; cap below 64535.
    import random
    for _ in range(100):
        port = random.randint(30000, 60000)
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise RuntimeError("could not obtain free port")


def _wait_listening(host: str, port: int, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with closing(socket.create_connection((host, port), timeout=1.0)):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{host}:{port} never came up")


@pytest.mark.slow
def test_heterogeneous_mlx_head_pytorch_tail_tier1():
    """Mac MLX head (layers 0-14) + Mac PyTorch CPU tail (layers 15-29)
    produce the same token sequence as the bf16 single-backend oracle."""
    if not _MANIFEST.exists():
        pytest.skip("reference manifest missing; run scripts/run_reference.py first")

    manifest = json.loads(_MANIFEST.read_text())
    record = manifest["prompts"][_PROMPT_IDX]
    prompt_tokens = list(record["prompt_tokens"])
    expected = list(record["generated_tokens"])[:_MAX_NEW_TOKENS]

    port_head = _free_port()
    port_tail = _free_port()
    cfg_text = _SHARDS_YAML_TMPL.format(port_head=port_head, port_tail=port_tail)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "shards.yaml"
        cfg_path.write_text(cfg_text)

        env_head = {**os.environ, "MODEL_SHARD_BACKEND": "mlx"}
        env_tail = {**os.environ, "MODEL_SHARD_BACKEND": "pytorch"}

        cmd_head = [
            sys.executable, "-m", "model_shard.scripts.run_node",
            "--config", str(cfg_path), "--shard", "head",
        ]
        # The above import path may not exist; the actual entry point is
        # scripts/run_node.py executed via uv run python scripts/run_node.py.
        # Use the same form here for both:
        cmd_head = [
            "uv", "run", "python", "scripts/run_node.py",
            "--config", str(cfg_path), "--shard", "head",
        ]
        cmd_tail = [
            "uv", "run", "python", "scripts/run_node.py",
            "--config", str(cfg_path), "--shard", "tail",
        ]

        proc_head = subprocess.Popen(
            cmd_head, env=env_head, cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        proc_tail = subprocess.Popen(
            cmd_tail, env=env_tail, cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        try:
            _wait_listening("127.0.0.1", port_head, timeout_s=180.0)
            _wait_listening("127.0.0.1", port_tail, timeout_s=300.0)
            # Allow SWIM membership to stabilize.
            time.sleep(5.0)

            from model_shard.client import Client
            from model_shard.shard_map import NodeAddress
            client = Client(head_address=NodeAddress(host="127.0.0.1", port=port_head))
            got = client.generate(prompt_tokens, max_new_tokens=_MAX_NEW_TOKENS)
            assert got == expected, (
                f"heterogeneous pipeline output {got!r} != "
                f"reference {expected!r} (prompt {_PROMPT_IDX})"
            )
        finally:
            for proc in (proc_head, proc_tail):
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
```

- [ ] **Step 2: Sanity-check the test compiles + collects**

```bash
cd /Users/lukechang/Github/model_shard
uv run pytest --collect-only tests/test_heterogeneous_2subprocess.py -v
```

Expected: 1 test collected (slow-marked, deselected from the default `pytest -q` run).

- [ ] **Step 3: Run the test**

```bash
uv run pytest -m slow tests/test_heterogeneous_2subprocess.py -v
```

Expected: 1 passed. Runtime: 5-15 minutes (PyTorch on Mac CPU is slow). If the test hangs at "_wait_listening tail":
- The PyTorch subprocess is taking >5 minutes to load the bf16 model on Mac CPU. Bump `_wait_listening` timeout for the tail to 600s.
- Or the PyTorch subprocess crashed. Look at its captured stdout/stderr (the test currently buffers those — for debugging, change `stderr=subprocess.STDOUT` to write to files in `tmpdir`).

If the test fails with token mismatch:
- Print both `got` and `expected` and compare.
- Re-run a single-backend Tier 1 test to confirm the oracle and the single-backend path still agree.
- The likely root cause is a wire-format mismatch — re-run Task 1's test to confirm it still passes.

- [ ] **Step 4: Ruff + mypy**

```bash
uv run ruff check tests/test_heterogeneous_2subprocess.py
uv run mypy tests/test_heterogeneous_2subprocess.py
```

Both zero errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_heterogeneous_2subprocess.py
git commit -m "Phase 7-C-3b Task 7: 2-subprocess heterogeneous Tier 1 pytest"
```

## Context

- **Predecessor commit:** Task 6
- **Spec:** §3.6 Surface 1
- The `scripts/run_node.py` entry point already supports the `--config` and `--shard` args used here.
- PyTorch on Mac CPU does NOT use MPS for Gemma 4 26B in mlx-vlm — it falls back to CPU bf16. This is intentional for the test (consistent precision across both subprocesses) and tolerable because the test only runs 4 tokens.
- Memory peak: bf16 model loaded twice in parallel. With mmap sharing on the same machine, total resident is much less than 2×48 GB but Python overhead is real. Document the ≥80 GB requirement in the test docstring.

## Your Job

1. Follow Steps 1-5.
2. Test passes on M5 (≥80 GB unified).
3. Ruff + mypy clean.
4. Commit.
5. Report the actual runtime of the test.

---

### Task 8: 3-machine deployment runbook + example shards.yaml

**Files:**
- Create: `docs/runbooks/heterogeneous_3node.md`
- Create: `config/shards.heterogeneous.example.yaml`

- [ ] **Step 1: Create the example shards.yaml template**

Create `config/shards.heterogeneous.example.yaml`:

```yaml
# Phase 7-C-3b heterogeneous cluster — 3-machine demo.
#
# Copy this file to each machine, customize the host fields with the
# Tailscale hostnames or IPs of YOUR three machines, then commit per-
# machine to /etc/model_shard/shards.yaml or pass via --config.
#
# All three machines MUST have the same model_id string for cluster
# admission to accept the peers. The MLX backend on Mac transparently
# resolves this HF id to a local conversion at
# ~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16/ (override with
# MLX_MODEL_BF16_LOCAL_PATH).

model_id: "google/gemma-4-26B-A4B-it"

shards:
  # Mac M5: MLX bf16 head, layers 0-9, full load.
  head:
    host: <mac-tailscale-hostname>  # e.g., m5.tail-net.ts.net
    port: 9001
    start_layer: 0
    end_layer: 10

  # DGX Spark: PyTorch bf16 mid, layers 10-19, full load.
  mid:
    host: <spark-tailscale-hostname>  # e.g., spark-8c43.tail-net.ts.net
    port: 9002
    start_layer: 10
    end_layer: 20

  # Ubuntu 3090: PyTorch bf16 tail, layers 20-29, partial load (24 GB VRAM).
  # The moe_experts assignment below holds 1/3 of routed experts at each
  # MoE layer (layers 20-29). Tune to fit available VRAM.
  tail:
    host: <3090-tailscale-hostname>
    port: 9003
    start_layer: 20
    end_layer: 30
    moe_experts:
      20: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
      21: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
      22: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
      23: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
      24: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
      25: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
      26: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
      27: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
      28: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
      29: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
```

- [ ] **Step 2: Create the runbook**

Create `docs/runbooks/heterogeneous_3node.md`:

```markdown
# Heterogeneous 3-node deployment runbook (Phase 7-C-3b)

This runbook walks through deploying a 3-machine inference cluster with
mixed MLX and PyTorch backends:

- **Mac M5** — MLX bf16 head, layers 0-9, full load
- **DGX Spark** — PyTorch bf16 mid, layers 10-19, full load
- **Ubuntu 3090 (24 GB VRAM)** — PyTorch bf16 tail, layers 20-29, partial load

All three machines connect over Tailscale and serve the same source weights
(`google/gemma-4-26B-A4B-it`).

## Prerequisites

- All 3 machines on the same Tailscale tailnet. Verify with:
  ```bash
  tailscale status
  tailscale ping <each-other-host>
  ```

- HuggingFace authentication for `google/gemma-4-26B-A4B-it` on Spark and
  3090 (Mac uses the local conversion):
  ```bash
  huggingface-cli login
  huggingface-cli whoami  # confirm
  ```

- On Mac: MLX bf16 conversion already produced (Phase 7-C-3a Task 7).
  Verify the cache exists:
  ```bash
  ls ~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16/
  ```

- On Spark and 3090: clone or rsync the model_shard repo and `uv sync
  --extra dev --extra pytorch`.

- On 3090: confirm at least 22 GB free VRAM (`nvidia-smi`).

## Configuration

1. Copy `config/shards.heterogeneous.example.yaml` on each machine.
2. Replace `<mac-tailscale-hostname>`, `<spark-tailscale-hostname>`, and
   `<3090-tailscale-hostname>` with the actual hostnames or IPs.
3. The `model_id` string MUST be identical on all three machines —
   admission control will reject mismatched peers.

Save as `~/model-shard-shards.yaml` (or anywhere; pass via `--config`).

## Pre-flight smoke checks

On each machine:
```bash
# Confirm config parses and model_id is the canonical HF id.
uv run python -c "
from pathlib import Path
from model_shard.shard_map import ShardMap
sm = ShardMap.from_yaml(Path('~/model-shard-shards.yaml').expanduser())
print('model_id:', sm.model_id)
print('shards:', sm.all_shards())
"
```

Expected on all 3 machines: `model_id: google/gemma-4-26B-A4B-it`.

## Start the cluster

Order of startup doesn't matter — SWIM gossip is order-independent. But
starting tail first lets you watch its memory usage as the others
connect.

### On the 3090 (tail):

```bash
cd ~/Github/model_shard
ENABLE_PARTIAL_LOAD=true \
MODEL_SHARD_BACKEND=pytorch \
uv run python scripts/run_node.py \
    --config ~/model-shard-shards.yaml \
    --shard tail
```

In another terminal on the 3090, watch VRAM:
```bash
watch -n 1 nvidia-smi
```

The tail should load partial weights and stabilize at <22 GB VRAM. If
it OOMs, reduce `moe_experts` per layer in the shards.yaml.

### On Spark (mid):

```bash
cd ~/Github/model_shard
MODEL_SHARD_BACKEND=pytorch \
uv run python scripts/run_node.py \
    --config ~/model-shard-shards.yaml \
    --shard mid
```

### On Mac (head):

```bash
cd ~/Github/model_shard
MODEL_SHARD_BACKEND=mlx \
uv run python scripts/run_node.py \
    --config ~/model-shard-shards.yaml \
    --shard head
```

## Smoke verification

In a 4th terminal on Mac, run a single-prompt client against the head:

```bash
cd ~/Github/model_shard
MODEL_SHARD_BACKEND=mlx \
uv run python scripts/run_client.py \
    --config ~/model-shard-shards.yaml \
    --prompt-set tests/prompts.json \
    --out-dir /tmp/heterogeneous-out \
    --max-new-tokens 16
```

Compare the generated tokens for prompt 0 against the bf16 oracle:

```bash
uv run python -c "
import json
ref = json.load(open('artifacts/ref/manifest.json'))
got = json.load(open('/tmp/heterogeneous-out/results.json'))
ref_ids = ref['prompts'][0]['generated_tokens'][:16]
got_ids = got['prompts'][0]['generated_tokens'][:16]
print('reference:', ref_ids)
print('cluster:  ', got_ids)
print('match:', ref_ids == got_ids)
"
```

Expected: `match: True`. If False, see "Common failure modes" below.

## Common failure modes

### `RuntimeError: rejecting peer ... with model_id mismatch`

A node has a different `model_id` in its `shards.yaml`. Verify all 3
configs have `model_id: "google/gemma-4-26B-A4B-it"` exactly.

### Tail OOMs on the 3090

Reduce `moe_experts` per layer in the tail's `shards.yaml`. The example
config holds ~42 experts per layer × 10 MoE layers; if that doesn't fit,
try ~28 experts per layer.

### Cluster never stabilizes (peers stuck SUSPECT)

Tailscale connectivity issue. Run `tailscale ping <peer>` from each
machine to confirm bidirectional reachability. If only one direction
works, check Tailscale firewall rules.

### Token sequence mismatches the oracle

If position-0 differs, the wire format isn't aligned across backends —
re-run `tests/test_cross_backend_wire_roundtrip.py` on Mac to verify.

If positions 1+ drift while position-0 matches, that's accumulating
floating-point divergence between backends, which is expected to a
small degree. Tier 1 tolerance is exact-match, so any drift fails the
test. The 7-C-2 cross-backend agreement bar already measures this; if
it's >3.07 top-5 overlap (the post-7-C-3a baseline), something
regressed.

### Head can't reach mid or tail

Verify the `host` fields in `shards.yaml` are reachable Tailscale
hostnames. Try `nslookup <host>` and `ping <host>` from the Mac.
```

- [ ] **Step 3: Verify both files exist and are well-formed**

```bash
cd /Users/lukechang/Github/model_shard
ls -la config/shards.heterogeneous.example.yaml docs/runbooks/heterogeneous_3node.md
uv run python -c "
import yaml
y = yaml.safe_load(open('config/shards.heterogeneous.example.yaml'))
print('model_id:', y['model_id'])
print('shards:', list(y['shards'].keys()))
"
```

Expected: shards = `['head', 'mid', 'tail']`, model_id is the HF id.

- [ ] **Step 4: Commit**

```bash
git add config/shards.heterogeneous.example.yaml docs/runbooks/heterogeneous_3node.md
git commit -m "Phase 7-C-3b Task 8: deployment runbook + example heterogeneous shards.yaml"
```

## Context

- **Predecessor commit:** Task 7
- **Spec:** §3.6 Surface 2
- The runbook is verified by Task 9 (the actual user-action smoke test). If the runbook is wrong, Task 9 fails.

## Your Job

1. Follow Steps 1-4.
2. Both files committed.
3. Verify YAML parses cleanly.

---

### Task 9: USER ACTION — manual smoke verification on Mac+Spark+3090

**STOP. This task requires the user to physically run the runbook on real hardware. The agentic worker should pause here and ask the user to perform the steps below, then report back.**

This is the load-bearing verification of the entire phase: a real heterogeneous cluster running real inference. Without this, 7-C-3b is just code that hasn't been deployed.

The user will:

1. Sync the latest code (`git pull` on Spark + 3090 if remote, or rsync from Mac).
2. Confirm prerequisites per the runbook (Tailscale connectivity, HF auth, MLX cache, 3090 VRAM headroom).
3. Start the 3 nodes in order (tail → mid → head).
4. Run the smoke verification client.
5. Compare cluster output to the bf16 oracle.
6. Report: pass / fail / what failed.

If the smoke verification PASSES, Task 9 is done — the heterogeneous cluster works end-to-end.

If it FAILS, the failure mode determines next steps:
- Token mismatch → cross-backend wire format regression (Task 1 should have caught this; investigate)
- Cluster never stabilizes → Tailscale or admission control issue (debug per the runbook)
- 3090 OOM → tune `moe_experts` (this is configuration, not a code bug)

The agentic worker should NOT attempt to debug remote machine issues without user input — they're physical hardware deployment failures, not code bugs in most cases.

## Your Job (as the agentic worker)

1. **STOP.** Pause execution.
2. Print this message verbatim to the user:

> Task 9 needs you to physically run the heterogeneous cluster on Mac+Spark+3090.
>
> Open `docs/runbooks/heterogeneous_3node.md` and follow it end-to-end.
> Report back when done with one of:
>   - "smoke pass" if the cluster output matches the oracle
>   - "smoke fail: <description>" with what went wrong
>
> Estimated user time: 30-60 minutes (mostly waiting on model loads).

3. Wait for the user's response.
4. If "smoke pass", commit a note documenting the verification and proceed to Task 10:

```bash
git commit --allow-empty -m "Phase 7-C-3b Task 9: heterogeneous 3-machine smoke verification PASS"
```

5. If "smoke fail", help debug per the runbook's "Common failure modes" section. Don't proceed to Task 10 until the smoke verification passes.

## Context

- **Predecessor commit:** Task 8
- **Spec:** §3.6 Surface 2, §4 verification #6
- This is the only USER-ACTION gate in the plan. The rest is automatable.

---

### Task 10: README + memory + final sweep

**Files:**
- Modify: `README.md`
- Modify: `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`

- [ ] **Step 1: Add README status paragraph**

Open `README.md`. Find the Phase 7-C-3a section (begins `## Phase 7-C-3a status: Bf16 Canonical Rebaseline — complete`). Insert a new Phase 7-C-3b paragraph IMMEDIATELY AFTER the Phase 7-C-3a section (before the next `## Phase ...` heading). Match existing style: prose, no emojis, ~280 words.

```markdown
## Phase 7-C-3b status: Heterogeneous Gossip Cluster — complete

Phase 7-C-3b enables a single inference cluster with shards running on
different backends — MLX on Apple Silicon, PyTorch CUDA elsewhere — all
serving the same source weights (`google/gemma-4-26B-A4B-it`). The bf16
rebaseline in Phase 7-C-3a closed the precision gap between MLX and
PyTorch; 7-C-3b proves that the existing wire format already speaks
the same byte language across backends, then adds a cluster admission
contract via gossiped `model_id` to catch misconfigured peers before
they corrupt cluster output.

The wire format change is "no change required, just verified": both
`mlx_engine.tensor_to_bytes` and `pytorch_engine.tensor_to_bytes`
serialize bf16 as raw IEEE 754 bytes. A new fast unit test
(`tests/test_cross_backend_wire_roundtrip.py`) pins this contract by
running MLX → bytes → PyTorch tensor → bytes → MLX tensor and asserting
exact equality.

Cluster admission adds a `model_id` field to `MemberRecordPb`. SWIM
Ping/Ack carry it; receivers refuse to add peers with mismatched id to
their membership view. Mac switched from gossiping a local conversion
path to gossiping the canonical HF id; a new `_resolve_local_for_mlx`
helper in `mlx_engine.py` transparently maps the HF id to the local
cache directory (`~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16/` by
default; `MLX_MODEL_BF16_LOCAL_PATH` overrides). All cluster nodes now
gossip the SAME canonical string.

Two verification surfaces: an automated 2-subprocess pytest
(`tests/test_heterogeneous_2subprocess.py`) on Mac that spawns one MLX
process and one PyTorch CPU process and runs Tier 1 prompts through
the heterogeneous pipeline; and a manual 3-machine deployment runbook
(`docs/runbooks/heterogeneous_3node.md`) for the Mac MLX head + DGX
Spark PyTorch mid + Ubuntu 3090 PyTorch tail (with Phase 5a partial
loading on the 3090). The latter exercises the project's "CDN for
experts on heterogeneous hardware" thesis end-to-end.

Non-goals (deferred): cross-backend expert migration (separate phase
if/when needed), Phase 6-B provenance on the PyTorch path (deferred to
7-C-3c), boundary `allclose` instrumentation (Tier 1 catches divergence;
root-causing has low marginal cost). See
`docs/superpowers/specs/2026-04-24-phase7c3b-heterogeneous-cluster-design.md`.
```

- [ ] **Step 2: Memory entry**

Edit `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`. Find the existing `**Phase 7-C-3a STATUS: COMPLETE...**` entry. Add a Phase 7-C-3b COMPLETE entry IMMEDIATELY AFTER it (before whatever comes next).

```markdown
**Phase 7-C-3b STATUS: COMPLETE (2026-04-24).** Heterogeneous gossip cluster (MLX + PyTorch). All 10 plan tasks done.
- **Plan:** `docs/superpowers/plans/2026-04-24-phase7c3b-heterogeneous-cluster.md`
- **Design spec:** `docs/superpowers/specs/2026-04-24-phase7c3b-heterogeneous-cluster-design.md`
- **Phase 7-C-3b commit list:** see `git log --grep "Phase 7-C-3b" --oneline`
- **What it enables:** A single inference cluster with mixed MLX (Apple Silicon) and PyTorch (CUDA) backends serving the same source weights. The "CDN for experts on heterogeneous hardware" thesis demonstrated end-to-end on Mac M5 + DGX Spark + Ubuntu 3090 (with partial load).
- **Wire format:** unchanged. Both `mlx_engine.tensor_to_bytes` and `pytorch_engine.tensor_to_bytes` already serialize bf16 as raw IEEE 754 bytes. New `tests/test_cross_backend_wire_roundtrip.py` pins this contract.
- **Cluster admission:** `MemberRecordPb` gains a `string model_id = 6` field. SWIM Ping/Ack carry it; `MembershipState.try_apply_record` rejects peers with mismatched id. Local-empty branch is permissive (rolling-upgrade tolerance). 5 fast tests in `tests/test_membership_model_id_admission.py`.
- **Canonical HF id everywhere:** `config/shards.yaml::model_id` switched from local conversion path to `google/gemma-4-26B-A4B-it`. New `_resolve_local_for_mlx` helper in `mlx_engine.py` transparently maps HF id → `~/.cache/mlx-models/<basename>-bf16/` (overridable via `MLX_MODEL_BF16_LOCAL_PATH`).
- **Test surfaces:**
  - Automated 2-subprocess pytest on Mac (`tests/test_heterogeneous_2subprocess.py`): MLX head + PyTorch CPU tail, 1 Tier 1 prompt at low max_new_tokens. Memory ≥80 GB unified required.
  - Manual 3-machine runbook (`docs/runbooks/heterogeneous_3node.md`): Mac+Spark+3090 over Tailscale. Smoke verification compared cluster tokens to bf16 oracle.
- **What didn't change:** Backend protocol signatures, gossip beyond `model_id`, wire framing, provenance code paths, retry/eviction/migration logic, partial-load logic.
- **Phase 7-C-3c/4 carry-forwards (unchanged from prior memory plus new):**
  - 7-C-3c: Phase 6-B provenance on the PyTorch `_run_my_layers` path (orchestrator already provenance-aware; engine path doesn't append entries yet).
  - 7-C-4: tech-debt cleanup — `lm` param threading, `_MLX_COMPUTE_LOCK` alias, gate `mlx.core` import in `node.py`, gate `pytorch_backend` import in `backends/__init__.py`, retire the two heavy bf16 E2E tests once 3-Node-in-process is fundamentally fixed (likely via subprocess isolation).
  - Cross-backend expert migration (slice/attach format bridge between MLX 9-tensor quantized and PyTorch 2-tensor bf16): deferred indefinitely; routing-only is sufficient for the cluster thesis.
- **Next:** Phase 7-C-3c brainstorm (provenance on PyTorch) or Phase 7-C-4 brainstorm (tech-debt cleanup), user's choice.
```

- [ ] **Step 3: Final verification sweep**

```bash
cd /Users/lukechang/Github/model_shard

# Fast suite
uv run pytest -q -m "not slow"

# All Phase 7-C-3b new tests
uv run pytest -m slow tests/test_cross_backend_wire_roundtrip.py tests/test_membership_model_id_admission.py tests/test_resolve_local_for_mlx.py

# Slow Tier 1 + cross-backend regression
uv run pytest -m slow tests/test_tier1_tokens.py tests/test_tier2_hidden.py tests/test_cross_backend_correctness.py tests/test_bf16_memory_smoke.py

# Lint + types
uv run ruff check src tests scripts
uv run mypy src
```

All green.

Final grep — `model_id` literals should ONLY appear as documentation references or in `config/shards.tests.yaml` (intentional 4-bit test config):

```bash
grep -rE 'mlx-community/gemma-4-26b-a4b-it-4bit|google/gemma-4-26B-A4B-it' src/ scripts/ config/ --include="*.py" --include="*.yaml"
```

Expected output similar to before Phase 7-C-3b: matches in `config/shards.tests.yaml` (test config), `config/shards.yaml` (now uses HF id, that's the expected match), docstrings/comments in `shard_map.py`, `pytorch_engine.py`, `convert_mlx_bf16.py`. No new executable defaults.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Phase 7-C-3b Task 10: README + memory (7-C-3b COMPLETE)"
```

(The memory file lives outside the git repo and is updated in place — no `git add` needed for it.)

## Context

- **Predecessor commit:** Task 9 (smoke verification PASS)
- **Spec:** §6 task 10
- The heterogeneous routing claim is verified end-to-end across two surfaces (automated + manual). The phase is genuinely shippable.

## Your Job

1. Follow Steps 1-4.
2. README paragraph inserted in correct location.
3. Memory entry inserted in correct location.
4. Final verification sweep all green.
5. Final grep returns expected results.
6. Commit.
7. Report final commit list: `git log --grep "Phase 7-C-3b" --oneline`.

---

## Self-Review Notes

**Spec coverage:**
- §3.1 (wire format verified, not redesigned) → Task 1
- §3.2 (`model_id` in `MemberRecordPb`, admission validation) → Tasks 2, 3, 4
- §3.3 (runner threads model_id) → Task 5
- §3.4 (HF-id-canonical, `_resolve_local_for_mlx`) → Task 6
- §3.5 (3-shard topology) → Task 8 (example yaml)
- §3.6 Surface 1 (2-subprocess pytest) → Task 7
- §3.6 Surface 2 (3-machine runbook) → Task 8
- §4 verification table → all 7 entries mapped (Tasks 1, 4, 3, all-prior, 7, 9, 10)
- §5 risks → addressed in task notes
- §6 task breakdown → 10 tasks 1:1 (Spec listed 10, plan has 10)

**Placeholder scan:**
- "Fill in observed runtime in Task 7 report" — deliberate lookup-on-completion (we don't know what runtime PyTorch CPU on Gemma 4 26B will produce until it runs). Not a placeholder; a report field.
- Task 9 has no automatable code — by design, it's a USER-ACTION gate. The spec acknowledges this in §6 task 9.
- No "TBD" / "add error handling" / "similar to Task N" / "implement later" patterns.

**Type consistency:**
- `MemberRecord.model_id: str` consistent across Task 3 (introduce), Task 4 (consume in admission), Task 5 (thread through runner).
- `MembershipState.__init__(local_model_id: str = "")` consistent across Tasks 4 (define) and 5 (consume in runner).
- `MembershipRunner.__init__(local_model_id: str = "")` consistent across Tasks 5 (define) and Task 8 example (already passed via `Node.__init__`).
- `_resolve_local_for_mlx(model_id: str) -> str` consistent across Task 6 (define) and `load_model` (consume).
- `model_id` field tag = 6 in proto (Task 2), referenced via the regenerated module.

No type or signature drift. All referenced functions, fields, and fixtures exist in the tasks where they're first introduced.

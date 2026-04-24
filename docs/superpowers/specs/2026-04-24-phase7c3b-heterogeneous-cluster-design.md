# Phase 7-C-3b: Heterogeneous Gossip Cluster — Design

**Status:** Draft, awaiting user review.
**Date:** 2026-04-24
**Phase predecessor:** 7-C-3a (Bf16 Canonical Rebaseline, commits `6d3cc60` through `421b6d5`).
**Phase successors:** 7-C-3c (Phase 6-B provenance on PyTorch path) and/or 7-C-4 (tech-debt cleanup).

## 1. Goal

Run a single inference cluster where shards execute on different backends — MLX on Apple Silicon and PyTorch CUDA elsewhere — all serving the same source weights (`google/gemma-4-26B-A4B-it`, locally bf16 on each backend's native format). Verified by Tier 1 token-exact agreement against the Phase 1 oracle on both an automated 2-subprocess test and a manual 3-machine deployment.

This is the original "CDN for experts on heterogeneous hardware" thesis made concrete: a Mac talking to a CUDA box talking to a memory-constrained CUDA box, all serving coherent inference of the same model.

## 2. Scope and non-goals

**In scope:**
- Cross-backend activation transport at the wire level. Both `mlx_engine.tensor_to_bytes` and `pytorch_engine.tensor_to_bytes` already serialize bf16 as raw IEEE 754 bytes (matching layouts). 7-C-3b verifies this with a unit test rather than redesigning a bridging layer.
- Cluster admission contract: each SWIM Ping/Ack carries the node's `model_id`. A receiving node refuses to add a peer with a mismatched id to its membership view. Catches misconfigurations like "rogue 4-bit MLX node tries to join the bf16 cluster" before the cluster silently produces garbage.
- Automated heterogeneous test: 2 subprocesses on localhost, one MLX and one PyTorch CPU, pipelined through the standard activation transport. Runs the Phase 1 prompt set (1-2 prompts, slow-marked) and asserts token-exact match against the bf16 oracle.
- Manual 3-machine deployment runbook: Mac MLX head + DGX Spark PyTorch mid + Ubuntu 3090 PyTorch tail (with Phase 5a partial loading), connected via Tailscale. Runs Tier 1 against the bf16 oracle as smoke verification.

**Explicitly out of scope (deferred):**
- Cross-backend expert migration (separate phase if/when needed; the slice/attach/detach format mismatch between MLX 9-tensor and PyTorch 2-tensor representations is a real research project that doesn't gate routing).
- Phase 6-B provenance on the PyTorch path (the orchestrator is provenance-aware but the PyTorch `_run_my_layers` doesn't currently append entries; deferred to 7-C-3c or folded into 7-C-4).
- Boundary `allclose` instrumentation. Tier 1 token-exact catches divergence; root-causing whether the divergence is at the wire boundary or deeper has low marginal cost once you're already in the failing test.
- Backend auto-detection improvements beyond what already exists (`MODEL_SHARD_BACKEND` env-var with platform-based default).
- Production cluster orchestration / lifecycle management. Deployment is manual via SSH + Tailscale per the runbook.

## 3. Architecture

### 3.1 Wire format: nothing to redesign

Verified state (2026-04-24):

- `src/model_shard/mlx_engine.py::tensor_to_bytes` (line 210): `staged = np.array(arr.view(mx.uint16)) if arr.dtype == mx.bfloat16 else np.array(arr)` — bf16 viewed as uint16, then numpy `tobytes()`. 2 bytes per element, IEEE 754 layout.
- `src/model_shard/pytorch_engine.py::tensor_to_bytes` (line 275): `t.contiguous().cpu().view(torch.uint8).numpy().tobytes()` — bf16 viewed as uint8 stream. 2 bytes per element, IEEE 754 layout.

Both produce identical byte streams for the same bf16 tensor. The receiving side calls its own `bytes_to_tensor` to deserialize into its native tensor type. **No protocol bridging layer needed.**

The wire-format invariant is pinned by `tests/test_cross_backend_wire_roundtrip.py` (new): take an MLX bf16 tensor, serialize via MLX, deserialize via PyTorch, verify equality (and the reverse direction). This catches any future regression where one backend changes its serialization layout.

### 3.2 Gossip extension: `model_id` in `MemberRecordPb`

Add one field to the existing protobuf:

```protobuf
message MemberRecordPb {
  // ... existing fields ...
  string model_id = N;  // NEW: cluster-wide canonical model identifier
}
```

`N` is the next available field tag (currently 6 or 7 depending on existing fields — confirm at implementation time). The new field is optional in protobuf semantics (default `""`).

Validation logic in `MembershipState.try_apply_record`:

```python
def try_apply_record(self, record: MemberRecord) -> bool:
    # Existing incarnation comparison logic ...

    # Phase 7-C-3b: cluster admission contract.
    # If both sides have set model_id and they don't match, refuse to admit.
    if self._local_model_id and record.model_id:
        if record.model_id != self._local_model_id:
            log.warning(
                "rejecting peer %s with mismatched model_id: "
                "local=%r peer=%r",
                record.shard_id, self._local_model_id, record.model_id,
            )
            return False

    # Continue with normal apply ...
```

Backwards compat: nodes that don't set `model_id` (legacy or test paths) get treated as `""`. A new node that DOES set its own `model_id` will reject any peer reporting `""` — this is intentional. Once a cluster is on the new contract, mixed-version peers can't silently join.

### 3.3 Membership runner threads model_id from ShardMap

`MembershipRunner.__init__` gains a `model_id: str` parameter. The local `MemberRecord` it constructs and gossips includes this field. `Node.__init__` reads `self._shard_map.model_id` and passes it to the runner constructor.

### 3.4 Backend selection: unchanged

`Node.__init__` already reads `_default_backend()`, which honors `MODEL_SHARD_BACKEND=mlx|pytorch` with platform auto-detect (MLX on Apple Silicon, PyTorch elsewhere). The 3 demo machines:

| Machine | env var | Loads from |
|---|---|---|
| Mac M5 (head) | `MODEL_SHARD_BACKEND=mlx` (or auto) | `~/.cache/mlx-models/gemma-4-26b-a4b-it-bf16` (local) |
| DGX Spark (mid) | `MODEL_SHARD_BACKEND=pytorch` | `google/gemma-4-26B-A4B-it` (HF) |
| Ubuntu 3090 (tail) | `MODEL_SHARD_BACKEND=pytorch` | `google/gemma-4-26B-A4B-it` (HF), partial-load |

Each machine has its own `config/shards.yaml` with `model_id` set to whatever path/id its backend can load. Cluster admission contract requires the *same string* across all nodes — but Mac uses a local path while Spark uses an HF id. Resolution: declare the **canonical HF id** as the shared `model_id`, and let the MLX backend resolve "if `model_id` is an HF id and a local conversion exists at the conventional cache path, load from cache; else load from HF." The conventional cache path mapping is documented in the runbook.

This means `Node`-level admission is on the canonical HF id, while the backend's actual `load()` call resolves to a local path on Mac. Implementation: a small `_resolve_local_for_mlx(hf_id: str) -> str` helper in `mlx_engine.py` checks `~/.cache/mlx-models/<basename>-bf16/` and returns that path if present, else returns the HF id unchanged.

### 3.5 Topology

Standard 3-shard layer split (10/10/10):

```
Client → Mac (MLX, head, layers 0-9, full bf16)
       → Spark (PyTorch, mid, layers 10-19, full bf16)
       → 3090 (PyTorch, tail, layers 20-29, partial bf16)
       → SampledToken back to Mac (head)
       → Mac → Client
```

The 3090 (24 GB VRAM) cannot fit a full ~48 GB bf16 load. Phase 5a partial loading (production code, Spark-tested in 7-B) holds chassis weights + assigned experts per layer. For the demo, the tail's expert assignment can mirror existing `config/shards.yaml` (e.g., the `e % 3 == 2` split for layers 20-29), or use a simpler "tail holds all routed experts for layers 20-29 minus a few" assignment that fits in 24 GB.

### 3.6 Test surfaces

**Surface 1 — automated pytest (slow-marked, Mac only):**

`tests/test_heterogeneous_2subprocess.py` spawns 2 subprocesses on localhost via `subprocess.Popen`:

- Process A: `MODEL_SHARD_BACKEND=mlx` (Apple Metal), serves layers 0-14, listens on a free TCP port.
- Process B: `MODEL_SHARD_BACKEND=pytorch` (Mac CPU bf16), serves layers 15-29, listens on a different free port.

The test runs 1-2 prompts from the Phase 1 prompt set through the heterogeneous pipeline (Mac client → A → B → token back to A → A → client) and asserts token-exact match against `artifacts/ref/manifest.json`. Tear-down kills both subprocesses.

PyTorch on Mac CPU is slow (~minutes per prompt), so the test:
- Limits to 1-2 prompts (not all 5).
- Uses `max_new_tokens` lower than the standard Tier 1 (e.g., 16 instead of 64).
- Marked `@pytest.mark.slow` and skipped under `addopts = -m 'not slow'`.

This test proves the protocol correctness on local hardware reproducibly. Memory requirement: ≥80 GB unified (loads bf16 model twice in parallel even with mmap sharing). Documented in test docstring.

**Surface 2 — 3-machine manual runbook:**

`docs/runbooks/heterogeneous_3node.md` covers:
- Tailscale connectivity smoke check (`tailscale ping <each-pair>`)
- Per-machine `shards.yaml` (with example template at `config/shards.heterogeneous.example.yaml`)
- Pre-flight: `huggingface-cli login` on Spark + 3090; `scripts/convert_mlx_bf16.py` already done on Mac
- Per-machine startup commands (use the existing `scripts/run_node.py`)
- Smoke verification: `scripts/run_client.py` against the head, run 1 prompt, compare output token sequence to the bf16 oracle in `artifacts/ref/manifest.json`
- Common failure modes: model_id mismatch (admission rejection), Tailscale firewall, partial-load OOM on 3090

Not automated. The user runs it manually after each significant code change that affects multi-machine deployment.

## 4. Verification

| # | Test | Where | Notes |
|---|---|---|---|
| 1 | `tests/test_cross_backend_wire_roundtrip.py` (fast) | Mac | Bit-exact MLX↔PyTorch tensor roundtrip |
| 2 | `tests/test_membership_model_id_admission.py` (fast) | Mac | Mismatch → reject, match → admit, missing → reject from new node |
| 3 | All existing membership tests still pass | Mac | Backwards compat sweep |
| 4 | All Phase 1-7-C-3a slow buckets remain green | Mac | Regression confirmation; no behavior change for single-backend single-machine clusters |
| 5 | `tests/test_heterogeneous_2subprocess.py` (slow) | Mac | Automated heterogeneous Tier 1 token-exact, 1-2 prompts |
| 6 | Manual runbook smoke verification | Mac+Spark+3090 (real) | 1 prompt end-to-end, output matches bf16 oracle |
| 7 | Fast suite + lint + types | Mac | `uv run pytest -q && uv run ruff check src tests scripts && uv run mypy src` |

## 5. Risks & mitigations

| ID | Risk | Mitigation |
|---|---|---|
| R1 | bf16 byte-order subtly differs between MLX and PyTorch on certain platforms. | Wire roundtrip unit test pins the contract on each platform it runs on. Both x86-64 and ARM-64 are little-endian, so this is theoretical, but the test catches future regressions. |
| R2 | 2-subprocess pytest takes minutes per prompt because PyTorch on Mac CPU is slow on Gemma 4 26B. | Limit to 1-2 prompts, lower `max_new_tokens`, mark `slow`. Protocol correctness is what's being tested, not throughput. |
| R3 | 3090 partial-load on PyTorch wasn't exercised by Spark Phase 7-B (Spark has full memory). 3090 will be the first real partial-load PyTorch deployment. | Manual runbook includes a "verify only assigned experts are resident" check. If partial load doesn't actually free memory on a real 24 GB GPU, that's a Phase 5a bug to fix on the PyTorch path — surface it as an issue, don't paper over. |
| R4 | Cluster admission via `model_id` rejects legacy nodes silently sending `model_id=""`. | Acceptable: this is intentional safety. The error message names both the local and peer ids so misconfigurations are debuggable. Document in the runbook. |
| R5 | Tailscale hostname resolution flakiness across 3 machines. | Runbook includes a `tailscale ping` smoke check from each pair before starting the cluster. Fix Tailscale first if broken. |
| R6 | The 2-subprocess test loads bf16 model twice in parallel. ~80 GB peak even with mmap sharing. Smaller Macs (e.g., 64 GB) can't run it. | Test docstring documents the ≥80 GB requirement. Smaller-Mac dev uses the 4-bit test config (same pattern as 7-C-3a's two-config split). |
| R7 | The HF-id-to-local-MLX-path resolution helper in §3.4 is an implicit convention. If a user puts the conversion at a non-standard path, the MLX backend tries to download from HF and fails. | The `_resolve_local_for_mlx` helper checks an env-var override first (`MLX_MODEL_BF16_LOCAL_PATH`) before falling back to the conventional cache path. Documented in the runbook. |
| R8 | Existing tests that programmatically build ShardMap and feed it into MembershipRunner might break when the runner gains a required `model_id` parameter. | Sweep is partial (already done in 7-C-3a Task 9 follow-up). Any remaining hits get fixed in the membership runner update task; small surface. |

## 6. Rough task breakdown

The full plan is produced by the writing-plans skill. This is a preview to validate scope:

1. **Wire-format roundtrip unit test.** Fast TDD test: MLX bf16 → bytes → PyTorch tensor → bytes → MLX tensor, verify equality. Both directions.
2. **`proto/wire.proto` extension.** Add `string model_id` to `MemberRecordPb`. Regenerate `_pb/wire_pb2.py`. No other proto changes.
3. **`MemberRecord` dataclass + serialization.** Add `model_id` field; update `to_proto` / `from_proto`. Existing membership unit tests get `model_id=""` defaults where not relevant.
4. **`MembershipState.try_apply_record` admission logic + fast tests.** Reject mismatched, admit matching, reject when local has model_id but peer doesn't.
5. **`MembershipRunner` + `Node.__init__` thread model_id.** Runner takes `model_id: str`; Node passes `self._shard_map.model_id`.
6. **`_resolve_local_for_mlx` helper in mlx_engine.py.** Resolve HF id → local cache path; env-var override; fast unit test. Update `config/shards.yaml` to use the canonical HF id (`google/gemma-4-26B-A4B-it`) instead of the local conversion path so all cluster nodes gossip the same admission string. The resolver makes the local cache path transparent to the MLX backend's load call.
7. **2-subprocess heterogeneous pytest.** Spawn 2 processes on localhost with different `MODEL_SHARD_BACKEND`, run 1-2 Tier 1 prompts, verify against oracle.
8. **3-machine deployment runbook + example shards.yaml.** Document Tailscale setup, per-machine config, startup commands, smoke verification, failure modes.
9. **Manual smoke verification on real hardware.** User runs the runbook on Mac+Spark+3090, reports back. Plan includes the procedure for doing this and what success looks like; the actual run is user-paced.
10. **README + memory + final sweep.** Document Phase 7-C-3b status; commit memory entry; full verification sweep.

Estimated 9-10 tasks. Similar shape to prior phases but with one user-action gate (Task 9, the manual demo).

## 7. Open questions

None at design time. All scope decisions resolved during brainstorming:

- Routing-only scope (carried from original 7-C-3 brainstorm).
- Bf16 everywhere (achieved by 7-C-3a).
- 3-machine demo: Mac MLX + Spark PyTorch + 3090 PyTorch partial-load.
- Verification: bf16 Tier 1 reference + cluster admission contract + wire-roundtrip unit test.
- Provenance on PyTorch deferred to 7-C-3c.
- Boundary allclose deferred (Tier 1 catches divergence; root-causing has low marginal cost).
- Two test surfaces: automated 2-subprocess pytest + manual 3-machine runbook.

# Phase 7-C-2 Cross-Backend Correctness Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish an empirical agreement bar between `MLXBackend` (Mac, 4-bit Gemma 4) and `PyTorchBackend` (DGX Spark, bf16 Gemma 4) via a fixture-based top-K overlap test — necessary groundwork before 7-C-3 mixes the two backends in one gossip pipeline.

**Architecture:** Each engine gets a small `top_k_ids_and_weights(logits, k=5)` helper. A unified fixture generator dispatches on `MODEL_SHARD_BACKEND=mlx|pytorch`, producing `mlx_tier1_tokens.json` or `pytorch_tier1_tokens.json` with top-K per decode position. A device-independent pytest (`test_cross_backend_correctness.py`) loads both JSONs and asserts graded agreement: minimum first-token top-1 matches + minimum average top-K overlap. A markdown side-by-side report is regenerated every run for human inspection.

**Tech Stack:** Python 3.13, `torch ≥ 2.6`, `mlx ≥ 0.19` (Apple Silicon) + `transformers ≥ 5.5.0`. Pure JSON comparison — no additional deps.

**Spec:** `docs/superpowers/specs/2026-04-20-phase7c2-cross-backend-correctness-design.md` — decisions D1–D8.

---

## File Structure

**Create:**
- `tests/test_mlx_engine_topk.py` — fast unit tests for MLX `top_k_ids_and_weights`.
- `tests/test_pytorch_engine_topk.py` — fast unit tests for PyTorch `top_k_ids_and_weights`.
- `scripts/generate_tier1_comparison_fixture.py` — unified generator with `MODEL_SHARD_BACKEND` dispatch.
- `tests/fixtures/mlx_tier1_tokens.json` — generated on Mac, committed.
- `tests/test_cross_backend_correctness.py` — the comparison test.

**Modify:**
- `src/model_shard/pytorch_engine.py` — add `top_k_ids_and_weights`.
- `src/model_shard/mlx_engine.py` — add `top_k_ids_and_weights`.
- `tests/fixtures/pytorch_tier1_tokens.json` — regenerated on Spark in new top-K format.
- `tests/test_pytorch_tier1.py` — consume new fixture format (`top_k_per_position[i].ids[0]` is top-1).
- `scripts/generate_pytorch_tier1_fixture.py` — deprecate/remove (replaced by unified generator).
- `README.md` — Phase 7-C-2 status paragraph.
- `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md` — Phase 7-C-2 COMPLETE entry.

---

## Task ordering

1. `top_k_ids_and_weights` helpers in both engines + fast unit tests (Mac-runnable — MLX needs Apple Silicon; PyTorch unit test is CPU-fine).
2. Unified fixture generator with `MODEL_SHARD_BACKEND` dispatch; delete old per-sided script.
3. Fixture generation — **Mac side**: run `MODEL_SHARD_BACKEND=mlx` and commit `mlx_tier1_tokens.json`. **Spark side**: run `MODEL_SHARD_BACKEND=pytorch` and regenerate `pytorch_tier1_tokens.json` in new format.
4. Update `tests/test_pytorch_tier1.py` to consume the new fixture shape; verify green on Spark.
5. `tests/test_cross_backend_correctness.py` + side-by-side markdown report.
6. README + memory Phase 7-C-2 COMPLETE.

---

### Task 1: `top_k_ids_and_weights` helpers + unit tests

**Files:**
- Modify: `src/model_shard/pytorch_engine.py`
- Modify: `src/model_shard/mlx_engine.py`
- Create: `tests/test_pytorch_engine_topk.py`
- Create: `tests/test_mlx_engine_topk.py`

- [ ] **Step 1: Write the PyTorch failing test**

Create `tests/test_pytorch_engine_topk.py`:

```python
"""Phase 7-C-2 Task 1: PyTorch top_k_ids_and_weights helper."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model_shard import pytorch_engine  # noqa: E402


def test_top_k_ids_and_weights_returns_python_lists():
    """Helper output must be JSON-serializable (list[int], list[float])."""
    logits = torch.zeros((1, 1, 10), dtype=torch.float32)
    logits[0, -1, 3] = 5.0
    logits[0, -1, 7] = 3.0
    logits[0, -1, 1] = 1.0
    ids, weights = pytorch_engine.top_k_ids_and_weights(logits, k=3)
    assert isinstance(ids, list)
    assert isinstance(weights, list)
    assert all(isinstance(i, int) for i in ids)
    assert all(isinstance(w, float) for w in weights)


def test_top_k_ids_and_weights_correct_order():
    """Highest-probability token first."""
    logits = torch.zeros((1, 1, 10), dtype=torch.float32)
    logits[0, -1, 3] = 5.0
    logits[0, -1, 7] = 3.0
    logits[0, -1, 1] = 1.0
    ids, _ = pytorch_engine.top_k_ids_and_weights(logits, k=3)
    assert ids == [3, 7, 1]


def test_top_k_ids_and_weights_returns_softmax_probs():
    """Weights are softmax probabilities — sum to 1 over full vocab,
    top-k slice sums to <=1 but top-1 dominates."""
    logits = torch.zeros((1, 1, 4), dtype=torch.float32)
    logits[0, -1, 0] = 10.0  # winner
    _, weights = pytorch_engine.top_k_ids_and_weights(logits, k=4)
    assert weights[0] > 0.99
    assert sum(weights) == pytest.approx(1.0, abs=1e-5)


def test_top_k_ids_and_weights_uses_last_position():
    """For a [B, L, V] tensor with L>1, take the LAST position only."""
    logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    logits[0, 0, 4] = 100.0  # would win if we looked at position 0
    logits[0, -1, 2] = 10.0  # actual winner at last position
    ids, _ = pytorch_engine.top_k_ids_and_weights(logits, k=1)
    assert ids == [2]


def test_top_k_ids_and_weights_k_larger_than_vocab():
    """Gracefully handle k > vocab (truncate to vocab size)."""
    logits = torch.zeros((1, 1, 3), dtype=torch.float32)
    ids, weights = pytorch_engine.top_k_ids_and_weights(logits, k=5)
    assert len(ids) == 3
    assert len(weights) == 3
```

- [ ] **Step 2: Run the PyTorch test — expect failure**

```bash
cd /Users/lukechang/Github/model_shard
uv run pytest tests/test_pytorch_engine_topk.py -v
```

Expected: AttributeError — `top_k_ids_and_weights` doesn't exist yet.

- [ ] **Step 3: Implement PyTorch helper**

Open `src/model_shard/pytorch_engine.py`. Find the wire-serialization section (near `tensor_to_bytes`). Add this function just after `bytes_to_tensor` (so it sits with the other pure-utility helpers):

```python
def top_k_ids_and_weights(
    logits: torch.Tensor, k: int = 5,
) -> tuple[list[int], list[float]]:
    """Return the top-K token IDs and softmax probabilities from the last
    position of a [B, L, V] logits tensor. Returns Python lists for
    fixture serialization. ``k`` is clamped to the vocab size."""
    last = logits[0, -1, :]
    weights = torch.softmax(last.float(), dim=-1)
    effective_k = min(k, last.shape[-1])
    top_w, top_i = torch.topk(weights, k=effective_k)
    return (
        [int(x) for x in top_i.cpu().tolist()],
        [float(w) for w in top_w.cpu().tolist()],
    )
```

Note: `last.float()` promotes before softmax — bf16 softmax can saturate on large logits. Match HF's own softmax dtype behavior.

- [ ] **Step 4: Run the PyTorch test — expect pass**

```bash
uv run pytest tests/test_pytorch_engine_topk.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Write the MLX failing test**

Create `tests/test_mlx_engine_topk.py`:

```python
"""Phase 7-C-2 Task 1: MLX top_k_ids_and_weights helper."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from model_shard import mlx_engine  # noqa: E402


def test_top_k_ids_and_weights_returns_python_lists():
    logits = mx.zeros((1, 1, 10), dtype=mx.float32)
    logits[0, -1, 3] = 5.0
    logits[0, -1, 7] = 3.0
    logits[0, -1, 1] = 1.0
    ids, weights = mlx_engine.top_k_ids_and_weights(logits, k=3)
    assert isinstance(ids, list)
    assert isinstance(weights, list)
    assert all(isinstance(i, int) for i in ids)
    assert all(isinstance(w, float) for w in weights)


def test_top_k_ids_and_weights_correct_order():
    logits = mx.zeros((1, 1, 10), dtype=mx.float32)
    logits[0, -1, 3] = 5.0
    logits[0, -1, 7] = 3.0
    logits[0, -1, 1] = 1.0
    ids, _ = mlx_engine.top_k_ids_and_weights(logits, k=3)
    assert ids == [3, 7, 1]


def test_top_k_ids_and_weights_returns_softmax_probs():
    logits = mx.zeros((1, 1, 4), dtype=mx.float32)
    logits[0, -1, 0] = 10.0
    _, weights = mlx_engine.top_k_ids_and_weights(logits, k=4)
    assert weights[0] > 0.99
    assert sum(weights) == pytest.approx(1.0, abs=1e-5)


def test_top_k_ids_and_weights_uses_last_position():
    logits = mx.zeros((1, 3, 5), dtype=mx.float32)
    logits[0, 0, 4] = 100.0
    logits[0, -1, 2] = 10.0
    ids, _ = mlx_engine.top_k_ids_and_weights(logits, k=1)
    assert ids == [2]


def test_top_k_ids_and_weights_k_larger_than_vocab():
    logits = mx.zeros((1, 1, 3), dtype=mx.float32)
    ids, weights = mlx_engine.top_k_ids_and_weights(logits, k=5)
    assert len(ids) == 3
    assert len(weights) == 3
```

- [ ] **Step 6: Run the MLX test — expect failure**

```bash
uv run pytest tests/test_mlx_engine_topk.py -v
```

Expected: AttributeError — `top_k_ids_and_weights` doesn't exist in `mlx_engine`.

- [ ] **Step 7: Verify MLX topk + softmax APIs**

Before implementing, confirm MLX's topk signature and whether it descends-by-default:

```bash
uv run python -c "
import mlx.core as mx
x = mx.array([3.0, 1.0, 4.0, 1.0, 5.0])
# Try topk — may be mx.topk or mx.argpartition; verify which exists and descending.
print('has topk:', hasattr(mx, 'topk'))
print('has argpartition:', hasattr(mx, 'argpartition'))
print('has argsort:', hasattr(mx, 'argsort'))
if hasattr(mx, 'topk'):
    print('topk signature try:')
    try:
        result = mx.topk(x, k=3)
        print('topk returned:', result)
        print('type:', type(result))
    except Exception as e:
        print('topk raised:', type(e).__name__, e)
"
```

Record which API is available. Adapt Step 8 implementation based on result. The implementation below assumes `mx.topk(x, k)` returns values-only (common case). If it's different (e.g., tuple return, or ascending order), adapt.

- [ ] **Step 8: Implement MLX helper**

Open `src/model_shard/mlx_engine.py`. Find the dtype-mapping / wire-serialization section and add this function alongside other utilities:

```python
def top_k_ids_and_weights(
    logits: mx.array, k: int = 5,
) -> tuple[list[int], list[float]]:
    """Return the top-K token IDs and softmax probabilities from the last
    position of a [B, L, V] logits tensor. Mirror of
    ``pytorch_engine.top_k_ids_and_weights``. Returns Python lists for
    fixture serialization. ``k`` is clamped to the vocab size."""
    last = logits[0, -1, :]
    weights = mx.softmax(last.astype(mx.float32), axis=-1)
    effective_k = min(k, int(last.shape[-1]))
    # mx.topk returns values only (descending). Pair with argsort for indices.
    # If mx.topk returns (values, indices) tuple in this version, unpack
    # directly — see Step 7 verification.
    top_w = mx.topk(weights, k=effective_k)
    # Recover indices: argsort descending on weights, take first k.
    top_i = mx.argsort(-weights)[:effective_k]
    return (
        [int(x) for x in top_i.tolist()],
        [float(w) for w in top_w.tolist()],
    )
```

**If Step 7 verification found `mx.topk` returns a (values, indices) tuple:** simplify to:
```python
top_w, top_i = mx.topk(weights, k=effective_k)
```
and skip the `argsort` line.

**If Step 7 found `mx.topk` does NOT exist:** fall back to argsort-only:
```python
sorted_desc = mx.argsort(-weights)
top_i = sorted_desc[:effective_k]
top_w = weights[top_i]
```

Pick whichever matches the installed MLX version.

- [ ] **Step 9: Run the MLX test — expect pass**

```bash
uv run pytest tests/test_mlx_engine_topk.py -v
```

Expected: 5 passed.

- [ ] **Step 10: Ruff + mypy**

```bash
uv run ruff check src/model_shard/pytorch_engine.py src/model_shard/mlx_engine.py tests/test_pytorch_engine_topk.py tests/test_mlx_engine_topk.py
uv run mypy src/model_shard/pytorch_engine.py src/model_shard/mlx_engine.py
```

Both zero errors. Apply narrow `# noqa` / `# type: ignore` only as needed.

- [ ] **Step 11: Commit**

```bash
git add src/model_shard/pytorch_engine.py src/model_shard/mlx_engine.py tests/test_pytorch_engine_topk.py tests/test_mlx_engine_topk.py
git commit -m "Phase 7-C-2 Task 1: top_k_ids_and_weights helpers (mlx + pytorch engines)"
```

## Context

- **Working directory:** `/Users/lukechang/Github/model_shard`
- **Branch:** `main` (user authorized main commits for this phase series)
- **Predecessor commit:** `432d680` (Phase 7-C-2 design spec)
- **Plan file:** this file.
- **Spec:** §3.2.

## Your Job

1. Follow Steps 1-11 exactly. TDD.
2. 5 + 5 = 10 tests pass.
3. Verify MLX topk API before blindly coding Step 8.
4. Ruff + mypy clean.
5. Commit with exact message.
6. Report back which MLX topk variant worked.

---

### Task 2: Unified fixture generator

**Files:**
- Create: `scripts/generate_tier1_comparison_fixture.py`
- Delete: `scripts/generate_pytorch_tier1_fixture.py`

- [ ] **Step 1: Create the unified generator**

Create `scripts/generate_tier1_comparison_fixture.py`:

```python
#!/usr/bin/env python
"""Phase 7-C-2: unified Tier-1 fixture generator for cross-backend comparison.

Dispatches on ``MODEL_SHARD_BACKEND=mlx|pytorch`` (defaults to pytorch) and
produces ``tests/fixtures/{mlx,pytorch}_tier1_tokens.json`` with top-K per
decode position. Consumed by:

  * ``tests/test_pytorch_tier1.py`` — internal regression (top-1 = ids[0]).
  * ``tests/test_cross_backend_correctness.py`` — cross-backend top-K
    overlap between the two fixtures.

Usage:
    # On Mac (Apple Silicon):
    MODEL_SHARD_BACKEND=mlx uv run python scripts/generate_tier1_comparison_fixture.py

    # On DGX Spark (CUDA):
    MODEL_SHARD_BACKEND=pytorch uv run python scripts/generate_tier1_comparison_fixture.py
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

PROMPTS = [
    "The quick brown fox",
    "In a galaxy far far away",
    "Once upon a time",
]
N_POSITIONS = 10
TOP_K_RECORDED = 5


def _load_backend() -> tuple[Any, str, str, str, Any]:
    """Return (backend, hf_id, device, dtype_str, topk_helper).

    topk_helper is the engine-specific ``top_k_ids_and_weights`` function
    so the per-prompt loop below can call either uniformly."""
    name = os.environ.get("MODEL_SHARD_BACKEND", "").lower() or "pytorch"
    if name == "mlx":
        from model_shard import mlx_engine
        from model_shard.backends import MLXBackend
        backend = MLXBackend(mlx_lock=threading.Lock())
        hf_id = "mlx-community/gemma-4-26b-a4b-it-4bit"
        device = "mps"
        dtype_str = "mlx-4bit"
        topk = mlx_engine.top_k_ids_and_weights
    elif name == "pytorch":
        from model_shard import pytorch_engine
        from model_shard.backends import PyTorchBackend
        backend = PyTorchBackend()  # auto-detect cuda/mps/cpu
        hf_id = "google/gemma-4-26B-A4B-it"
        device = backend._device
        # Convert torch dtype to a stable string (e.g. "bfloat16")
        dtype_str = str(backend._dtype).removeprefix("torch.")
        topk = pytorch_engine.top_k_ids_and_weights
    else:
        raise ValueError(
            f"MODEL_SHARD_BACKEND={name!r} not recognized "
            "(expected 'mlx' or 'pytorch')"
        )
    backend.load(hf_id)
    return backend, hf_id, device, dtype_str, topk


def _greedy_decode_with_topk(
    backend: Any, topk: Any, prompt_ids: list[int], n_positions: int, k: int,
) -> list[dict]:
    """Prefill + greedy decode through the backend. At each position record
    top-K (ids, weights). Returns a list of length n_positions, each entry
    ``{"ids": [...K], "weights": [...K]}``."""
    cache = backend.make_cache()
    h = backend.embed(prompt_ids)
    masks = backend.make_masks(h, cache)
    num_layers = backend.num_layers()
    # Prefill
    for i in range(num_layers):
        h = backend.run_layer_atomic(i, h, cache, masks)
    logits = backend.finalize(h)
    out: list[dict] = []
    ids, weights = topk(logits, k=k)
    out.append({"ids": ids, "weights": weights})
    # Decode remaining n_positions - 1
    for _ in range(n_positions - 1):
        tok_id = ids[0]  # greedy: follow top-1
        h = backend.embed([tok_id])
        masks = backend.make_masks(h, cache)
        for i in range(num_layers):
            h = backend.run_layer_atomic(i, h, cache, masks)
        logits = backend.finalize(h)
        ids, weights = topk(logits, k=k)
        out.append({"ids": ids, "weights": weights})
    return out


def main() -> None:
    backend, hf_id, device, dtype_str, topk = _load_backend()
    backend_name: str = backend.name

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)

    fixture: dict = {
        "model_id": hf_id,
        "backend": backend_name,
        "device": device,
        "dtype": dtype_str,
        "n_positions": N_POSITIONS,
        "top_k_recorded": TOP_K_RECORDED,
        "generator": f"{backend_name} greedy decode + top-{TOP_K_RECORDED} record",
        "prompts": [],
    }

    for prompt in PROMPTS:
        prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        top_k_per_position = _greedy_decode_with_topk(
            backend, topk, prompt_ids, N_POSITIONS, TOP_K_RECORDED,
        )
        fixture["prompts"].append({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "top_k_per_position": top_k_per_position,
        })

    out_path = (
        Path(__file__).resolve().parent.parent
        / "tests" / "fixtures" / f"{backend_name}_tier1_tokens.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Delete the old per-sided script**

```bash
cd /Users/lukechang/Github/model_shard
rm scripts/generate_pytorch_tier1_fixture.py
```

The unified generator supersedes it. Any reference to the old path in docs or memory is updated in Task 6.

- [ ] **Step 3: Ruff check on the new script**

```bash
uv run ruff check scripts/generate_tier1_comparison_fixture.py
```

Zero errors. Note: scripts intentionally skip strict mypy; PyTorch + MLX conditional imports make typing unwieldy.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_tier1_comparison_fixture.py
git rm scripts/generate_pytorch_tier1_fixture.py
git commit -m "Phase 7-C-2 Task 2: unified tier1 fixture generator (MODEL_SHARD_BACKEND dispatch)"
```

## Context

- **Predecessor commit:** Task 1.
- **Spec:** §3.1, §3.3.
- **Script is standalone** — not imported by any test, only run manually (Mac, Spark).
- **Why dispatch via env var?** Matches the existing Phase 7-B `_default_backend()` pattern in `node.py`. Avoids duplicating the same prefill+decode loop twice for MLX vs PyTorch.

## Your Job

1. Follow Steps 1-4.
2. Ruff clean.
3. Commit.
4. Report back.

---

### Task 3: Generate MLX fixture on Mac + regenerate PyTorch fixture on Spark

**Files:**
- Create: `tests/fixtures/mlx_tier1_tokens.json`
- Modify: `tests/fixtures/pytorch_tier1_tokens.json` (format bump)

This task has two halves on different hosts. The Mac half can run immediately; the Spark half requires reaching the Spark box via Tailscale / SSH as `ljchang@spark-8c43`.

- [ ] **Step 1: Mac-side — generate MLX fixture**

Prereq: Apple Silicon Mac with working MLX (verified by the fast MLX tests passing).

```bash
cd /Users/lukechang/Github/model_shard
MODEL_SHARD_BACKEND=mlx uv run python scripts/generate_tier1_comparison_fixture.py
```

Expected: takes a minute or two on M5 128GB (MLX 4-bit load is fast). Prints `Wrote /Users/.../tests/fixtures/mlx_tier1_tokens.json`.

- [ ] **Step 2: Verify the MLX fixture content**

```bash
cat tests/fixtures/mlx_tier1_tokens.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('backend:', d['backend'])
print('device:', d['device'])
print('dtype:', d['dtype'])
print('n_positions:', d['n_positions'])
print('top_k_recorded:', d['top_k_recorded'])
for p in d['prompts']:
    t0 = p['top_k_per_position'][0]
    print(f'  prompt={p[\"prompt\"]!r} top-1 pos0 id={t0[\"ids\"][0]} weight={t0[\"weights\"][0]:.3f}')
"
```

Verify:
- `backend == "mlx"`
- `dtype == "mlx-4bit"`
- `top_k_recorded == 5`
- Each prompt has 10 positions, each with 5 ids/weights.

- [ ] **Step 3: Spark-side — regenerate PyTorch fixture in new format**

SSH to Spark (via Tailscale) and rsync the current repo state. This may need user assistance if SSH isn't set up:

```bash
# From Mac:
rsync -az --delete \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='*.pyc' --exclude='.DS_Store' \
  /Users/lukechang/Github/model_shard/ ljchang@spark-8c43:~/Github/model_shard/

ssh ljchang@spark-8c43 "cd ~/Github/model_shard && \
  MODEL_SHARD_BACKEND=pytorch nohup ~/.local/bin/uv run python scripts/generate_tier1_comparison_fixture.py \
  > /tmp/fixture_gen.log 2>&1 & echo DISPATCHED"
```

The model is cached on Spark from Phase 7-C-1 (~54 GB already on disk); no re-download. Generation takes roughly 6 minutes (5-6 min weight load + ~20 s for 3 prompts × 10 tokens).

Poll completion:

```bash
ssh ljchang@spark-8c43 "tail -5 /tmp/fixture_gen.log ; pgrep -af generate_tier1_comparison_fixture | head"
```

When log ends with `Wrote /home/ljchang/Github/model_shard/tests/fixtures/pytorch_tier1_tokens.json`, scp it back:

```bash
scp ljchang@spark-8c43:~/Github/model_shard/tests/fixtures/pytorch_tier1_tokens.json \
    /Users/lukechang/Github/model_shard/tests/fixtures/pytorch_tier1_tokens.json
```

- [ ] **Step 4: Verify both fixtures have same prompts / prompt_ids**

```bash
cd /Users/lukechang/Github/model_shard
uv run python -c "
import json
mlx = json.load(open('tests/fixtures/mlx_tier1_tokens.json'))
pt = json.load(open('tests/fixtures/pytorch_tier1_tokens.json'))
for mp, pp in zip(mlx['prompts'], pt['prompts'], strict=True):
    assert mp['prompt'] == pp['prompt'], (mp['prompt'], pp['prompt'])
    assert mp['prompt_ids'] == pp['prompt_ids'], (mp['prompt_ids'], pp['prompt_ids'])
    print(f'{mp[\"prompt\"]!r}: prompt_ids match ({len(mp[\"prompt_ids\"])} tokens)')
"
```

If this fails with different `prompt_ids`, the tokenizer differs between `mlx-community/gemma-4-26b-a4b-it-4bit` and `google/gemma-4-26B-A4B-it` — the whole comparison is invalid. Stop and investigate (compare `tokenizer.json` files across the two models).

- [ ] **Step 5: Commit both fixtures**

```bash
git add tests/fixtures/mlx_tier1_tokens.json tests/fixtures/pytorch_tier1_tokens.json
git commit -m "Phase 7-C-2 Task 3: MLX (Mac) + PyTorch (Spark) tier1 fixtures in new top-K format"
```

## Context

- **Predecessor commit:** Task 2.
- **Spec:** §3.3 (fixture format), §4 (tokenizer equivalence assumption).
- **Split responsibility:** Mac generates MLX fixture; Spark regenerates PyTorch fixture; Mac commits both.
- **If Spark isn't reachable:** land the Mac half (Step 1-2) as a standalone commit; Spark half is a follow-up once Spark is reachable again. Don't block subsequent tasks — the next tasks depend on fixture format, not actual values.

## Your Job

1. Follow Steps 1-5.
2. Both fixtures present locally + committed.
3. prompt_ids match across fixtures (Step 4 guard).
4. Report back with observed position-0 top-1 tokens per prompt on both sides — early signal for whether the cross-backend test will pass.

---

### Task 4: Update `tests/test_pytorch_tier1.py` to consume new fixture format

**Files:**
- Modify: `tests/test_pytorch_tier1.py`

The Phase 7-C-1 test reads `case["generated_ids"]` — a flat list of top-1 tokens per position. The new fixture format has `case["top_k_per_position"][i]["ids"][0]` as the equivalent top-1. Update the test.

- [ ] **Step 1: Read current test**

```bash
grep -n "generated_ids\|top_k" tests/test_pytorch_tier1.py | head
```

Record the line numbers of `generated_ids` references.

- [ ] **Step 2: Update to new format**

Open `tests/test_pytorch_tier1.py`. Find the per-prompt loop that unpacks `case["generated_ids"]`. Replace the list-of-ints extraction with `[p["ids"][0] for p in case["top_k_per_position"]]`:

```python
@pytest.mark.slow
@pytest.mark.cuda
def test_tier1_tokens_match_fixture_top1(backend, fixture):
    """For each prompt in the fixture, greedy-decode N tokens through the
    backend's forward pass and compare top-1 IDs against
    fixture['prompts'][i]['top_k_per_position'][j]['ids'][0].

    Phase 7-C-2 fixture format: top-K per position. Top-1 is ``ids[0]`` of
    each position's entry; the extra K-1 entries are used by
    ``test_cross_backend_correctness.py`` but not asserted here."""
    _ = AutoTokenizer.from_pretrained(fixture["model_id"])
    for case in fixture["prompts"]:
        prompt_ids = case["prompt_ids"]
        expected_ids = [
            p["ids"][0] for p in case["top_k_per_position"]
        ]
        cache = backend.make_cache()
        h = backend.embed(prompt_ids)
        masks = backend.make_masks(h, cache)
        num_layers = backend.num_layers()
        for i in range(num_layers):
            h = backend.run_layer_atomic(i, h, cache, masks)
        logits = backend.finalize(h)
        token_id = backend.argmax_last(logits)
        got_ids = [token_id]
        for _ in range(fixture["n_positions"] - 1):
            h = backend.embed([token_id])
            masks = backend.make_masks(h, cache)
            for i in range(num_layers):
                h = backend.run_layer_atomic(i, h, cache, masks)
            logits = backend.finalize(h)
            token_id = backend.argmax_last(logits)
            got_ids.append(token_id)
        assert got_ids == expected_ids[:len(got_ids)], (
            f"prompt={case['prompt']!r}: got {got_ids}, expected {expected_ids[:len(got_ids)]}"
        )
```

Also update the placeholder-fixture skip guard if the old test has one — the placeholder format has neither `generated_ids` nor `top_k_per_position`, so the skip on `data.get("_placeholder")` still works unchanged.

- [ ] **Step 3: Ruff + mypy**

```bash
uv run ruff check tests/test_pytorch_tier1.py
```

- [ ] **Step 4: Regression check on Mac (CUDA-gated, so skipped locally)**

```bash
uv run pytest -m slow tests/test_pytorch_tier1.py -v
```

Expected: skipped on Mac (no CUDA). That's fine — Spark run is authoritative.

- [ ] **Step 5: Regression check on Spark**

```bash
ssh ljchang@spark-8c43 "cd ~/Github/model_shard && git pull -q --rebase origin main || echo 'no remote; rsync repo first' ; \
  nohup ~/.local/bin/uv run pytest -m slow tests/test_pytorch_tier1.py -v \
  > /tmp/tier1_recheck.log 2>&1 & echo DISPATCHED"
```

Wait ~6 min for load + test. Poll:

```bash
ssh ljchang@spark-8c43 "tail -10 /tmp/tier1_recheck.log ; pgrep -af pytest | head"
```

Expected: `1 passed` — same top-1 sequence the new fixture records (since both the fixture AND the test now share the `PyTorchBackend` path).

- [ ] **Step 6: Commit**

```bash
git add tests/test_pytorch_tier1.py
git commit -m "Phase 7-C-2 Task 4: test_pytorch_tier1 consumes new top-K fixture format"
```

## Context

- **Predecessor commit:** Task 3.
- **Spec:** §3.3 (format), §5 (Tier 1 still green post-format-bump).
- **The test still asserts top-1** (`ids[0]`). Nothing relaxed. Extra K-1 entries are unused by this test; picked up by Task 5.

## Your Job

1. Follow Steps 1-6.
2. Mac-side skipped cleanly (no CUDA).
3. Spark-side `1 passed`.
4. Commit.
5. Report back.

---

### Task 5: `tests/test_cross_backend_correctness.py` + markdown report

**Files:**
- Create: `tests/test_cross_backend_correctness.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cross_backend_correctness.py`:

```python
"""Phase 7-C-2 Task 5: cross-backend top-K agreement.

Compares the committed MLX and PyTorch tier-1 fixtures without loading
any model. Pure JSON diff — runs anywhere (no Apple Silicon / CUDA
required). Marked slow because it requires both fixtures to be
present and committed.

Agreement metric (graded, per spec §3.4):
  * Min first-token top-1 matches: 2 of 3 prompts' position-0 top-1 agree.
  * Min average top-K overlap: average top-5 intersection size >= 2.0
    across all (prompt, position) pairs.

Bars are conservative MVP values per spec §7. Threshold tuning is a
follow-up once we have observed numbers from real fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MLX_FIXTURE = FIXTURE_DIR / "mlx_tier1_tokens.json"
PT_FIXTURE = FIXTURE_DIR / "pytorch_tier1_tokens.json"
REPORT_FILE = FIXTURE_DIR / "cross_backend_comparison.md"

MIN_FIRST_TOKEN_TOP1_MATCHES = 2
MIN_AVERAGE_TOPK_OVERLAP = 2.0


def _load_or_skip(path: Path, label: str) -> dict:
    if not path.exists():
        pytest.skip(f"{label} fixture missing: {path}")
    data = json.loads(path.read_text())
    if data.get("_placeholder"):
        pytest.skip(f"{label} fixture is a placeholder: {path}")
    return data


def _format_per_position_row(
    mp_pos: dict, pp_pos: dict, position: int,
) -> str:
    mlx_ids = mp_pos["ids"]
    pt_ids = pp_pos["ids"]
    overlap = sorted(set(mlx_ids) & set(pt_ids))
    return (
        f"| {position} | {mlx_ids} | {pt_ids} | "
        f"{len(overlap)} ({overlap}) |"
    )


def _write_report(
    mlx: dict, pt: dict, first_matches: int,
    avg_overlap: float, overlaps: list[int],
) -> None:
    lines: list[str] = []
    lines.append("# Cross-backend Tier-1 comparison")
    lines.append("")
    lines.append(
        f"MLX backend: `{mlx['backend']}` on `{mlx['device']}` "
        f"({mlx['dtype']}), model `{mlx['model_id']}`"
    )
    lines.append(
        f"PyTorch backend: `{pt['backend']}` on `{pt['device']}` "
        f"({pt['dtype']}), model `{pt['model_id']}`"
    )
    lines.append("")
    lines.append(
        f"Position-0 top-1 matches: **{first_matches}/"
        f"{len(mlx['prompts'])}** prompts"
    )
    lines.append(
        f"Average top-{mlx['top_k_recorded']} overlap: "
        f"**{avg_overlap:.2f}** across {len(overlaps)} positions"
    )
    lines.append("")
    for mp, pp in zip(mlx["prompts"], pt["prompts"], strict=True):
        lines.append(f"## Prompt: `{mp['prompt']}`")
        lines.append("")
        lines.append("| position | MLX top-K | PyTorch top-K | overlap |")
        lines.append("|---|---|---|---|")
        for i, (mp_pos, pp_pos) in enumerate(
            zip(mp["top_k_per_position"], pp["top_k_per_position"], strict=True)
        ):
            lines.append(_format_per_position_row(mp_pos, pp_pos, i))
        lines.append("")
    REPORT_FILE.write_text("\n".join(lines) + "\n")


@pytest.mark.slow
def test_cross_backend_agreement():
    mlx = _load_or_skip(MLX_FIXTURE, "MLX")
    pt = _load_or_skip(PT_FIXTURE, "PyTorch")

    # Same prompts on both sides.
    assert [p["prompt"] for p in mlx["prompts"]] == [
        p["prompt"] for p in pt["prompts"]
    ], "Fixture prompt mismatch between MLX and PyTorch sides"

    # Same prompt_ids (tokenizer equivalence).
    for mp, pp in zip(mlx["prompts"], pt["prompts"], strict=True):
        assert mp["prompt_ids"] == pp["prompt_ids"], (
            f"Tokenizer mismatch on prompt={mp['prompt']!r}: "
            f"MLX={mp['prompt_ids']} vs PyTorch={pp['prompt_ids']}"
        )

    # Metric A: first-token top-1 agreement (position 0, no decode drift).
    first_token_matches = sum(
        1 for mp, pp in zip(mlx["prompts"], pt["prompts"], strict=True)
        if (
            mp["top_k_per_position"][0]["ids"][0]
            == pp["top_k_per_position"][0]["ids"][0]
        )
    )

    # Metric B: average top-K overlap across all (prompt, position) pairs.
    overlaps = [
        len(set(mp_pos["ids"]) & set(pp_pos["ids"]))
        for mp, pp in zip(mlx["prompts"], pt["prompts"], strict=True)
        for mp_pos, pp_pos in zip(
            mp["top_k_per_position"], pp["top_k_per_position"], strict=True,
        )
    ]
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    # Write the human-readable report regardless of pass/fail.
    _write_report(mlx, pt, first_token_matches, avg_overlap, overlaps)

    assert first_token_matches >= MIN_FIRST_TOKEN_TOP1_MATCHES, (
        f"Position-0 top-1 agreement: {first_token_matches}/"
        f"{len(mlx['prompts'])} prompts — below minimum "
        f"{MIN_FIRST_TOKEN_TOP1_MATCHES}. See {REPORT_FILE} for per-"
        "position top-K diagnostic."
    )
    assert avg_overlap >= MIN_AVERAGE_TOPK_OVERLAP, (
        f"Average top-{mlx['top_k_recorded']} overlap: {avg_overlap:.2f} "
        f"across {len(overlaps)} positions — below minimum "
        f"{MIN_AVERAGE_TOPK_OVERLAP}. See {REPORT_FILE}."
    )
```

- [ ] **Step 2: Run — expect pass (or informative failure)**

```bash
cd /Users/lukechang/Github/model_shard
uv run pytest -m slow tests/test_cross_backend_correctness.py -v
```

Expected outcomes:
- **Pass:** bars met. Note observed numbers for §7 tuning pass.
- **Fail with tokenizer mismatch:** stop; the fixtures use different tokenizers. Spec §4 violated.
- **Fail with below-threshold:** inspect `tests/fixtures/cross_backend_comparison.md`. Decide if the bars are wrong (tighten after inspection) or there's a real bug (investigate before loosening).
- **Skip:** one fixture missing — Task 3 didn't complete.

- [ ] **Step 3: Ruff + mypy**

```bash
uv run ruff check tests/test_cross_backend_correctness.py
uv run mypy tests/test_cross_backend_correctness.py
```

- [ ] **Step 4: Inspect the markdown report**

```bash
cat tests/fixtures/cross_backend_comparison.md | head -80
```

Verify it renders readably: backend labels on the header, one table per prompt, overlap column shows intersection sets.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cross_backend_correctness.py tests/fixtures/cross_backend_comparison.md
git commit -m "Phase 7-C-2 Task 5: cross-backend top-K agreement test + markdown report"
```

## Context

- **Predecessor commit:** Task 4.
- **Spec:** §3.4 (test body), §7 (threshold tuning).
- **Markdown report is committed** — gets regenerated each test run; small diff, acts as a "last known state" snapshot.
- **If bars fail:** `.md` report is the diagnostic. Follow §7 discipline — don't blindly loosen.

## Your Job

1. Follow Steps 1-5.
2. Report: did bars pass? Observed numbers (`first_token_matches`, `avg_overlap`)?
3. If pass with margin, note recommended tightened bars for §7 follow-up.
4. Commit test + report.

---

### Task 6: README + memory update

**Files:**
- Modify: `README.md`
- Modify: `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`

- [ ] **Step 1: README status paragraph**

Open `README.md`. Find the Phase 7-C-1 paragraph (begins `## Phase 7-C-1 status: Real HF Gemma 4 forward integration — complete`). Insert a Phase 7-C-2 paragraph AFTER it (before the next `## Phase ...` heading). Match existing style: prose, no emojis, ~180 words.

Cover:

- Scope: cross-backend correctness harness. Confirms `MLXBackend` (Mac, 4-bit) and `PyTorchBackend` (Spark, bf16) produce compatible tokens on identical prompts despite the quantization gap.
- Approach: unified fixture generator dispatches on `MODEL_SHARD_BACKEND=mlx|pytorch` and records top-K (K=5) ids + softmax weights per decode position. Two fixtures (`mlx_tier1_tokens.json`, `pytorch_tier1_tokens.json`) committed; device-independent `test_cross_backend_correctness.py` loads both and asserts a graded agreement bar.
- Bars (conservative MVP per spec §7): at least 2/3 prompts agree on position-0 top-1; average top-5 intersection size ≥ 2 across all (prompt, position) pairs.
- Observed numbers (fill in from Task 5 report): first-token matches X/3, avg top-5 overlap Y.YY. Tightening deferred.
- Side-by-side markdown report (`tests/fixtures/cross_backend_comparison.md`) committed for eyeball diagnostics; regenerated every test run.
- Non-goals (deferred): activation-level `allclose`, KL/JS divergence, online cross-machine harness, heterogeneous gossip cluster (7-C-3), tech-debt cleanup (7-C-4).
- Link to spec: `docs/superpowers/specs/2026-04-20-phase7c2-cross-backend-correctness-design.md`.

- [ ] **Step 2: Memory entry**

Edit `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`. Find the Phase 7-C-1 COMPLETE entry. Add a Phase 7-C-2 COMPLETE entry AFTER it. Structure parallel to the 7-C-1 entry.

Cover:

- Date `2026-04-20`, final commit SHA (fill in after Step 4).
- 6 tasks done.
- Plan + spec paths.
- Phase 7-C-2 commit list (`git log --grep "Phase 7-C-2" --oneline`).
- What it enables: empirical agreement bar between MLX and PyTorch backends. Unblocks 7-C-3 (heterogeneous gossip cluster — you can now mix MLX + PyTorch nodes in one pipeline with evidence they agree on outputs).
- Technical additions:
  - `top_k_ids_and_weights(logits, k=5)` helpers in both `pytorch_engine.py` and `mlx_engine.py`.
  - `scripts/generate_tier1_comparison_fixture.py` — unified, `MODEL_SHARD_BACKEND` dispatch.
  - Two committed fixtures with identical top-K schema.
  - `test_cross_backend_correctness.py` — device-independent, slow-marked.
  - Side-by-side markdown report (`tests/fixtures/cross_backend_comparison.md`).
- Observed numbers (from Task 5): first-token matches, avg top-5 overlap. Note for future tuning.
- What didn't change: Backend protocol signatures; MLX slow regression bucket; Spark Tier-1 regression; gossip; wire; provenance; retry; eviction.
- Phase 7-C-3/4 carry-forwards (unchanged):
  - 7-C-3: heterogeneous gossip cluster + 9-tensor↔2-tensor slice_expert bridge + Phase 6-B provenance on PyTorch path.
  - 7-C-4: tech-debt cleanup — `lm` param threading, `_MLX_COMPUTE_LOCK` alias, per-position aggregate-experts signature, `node.py` mlx import gating (Task #85).
- Next: Phase 7-C-3 brainstorm — heterogeneous gossip cluster mixing MLX and PyTorch nodes in one pipeline.

- [ ] **Step 3: Final verification sweep**

```bash
cd /Users/lukechang/Github/model_shard
uv run pytest -q -m "not slow"                                 # fast, Mac
uv run pytest -m slow -q tests/test_cross_backend_correctness.py  # cross-backend
uv run ruff check src tests scripts
uv run mypy src
```

Also run one MLX slow bucket as regression confirmation:

```bash
uv run pytest -m slow -q tests/test_tier1_tokens.py
```

All green.

- [ ] **Step 4: Commit**

```bash
git add README.md "/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md"
git commit -m "Phase 7-C-2 Task 6: README + memory update (7-C-2 COMPLETE)"
```

## Context

- **Predecessor commit:** Task 5.
- **Spec:** §6 (success criteria).

## Your Job

1. Follow Steps 1-4.
2. README paragraph (paste final text in report).
3. Memory COMPLETE entry with observed numbers.
4. Full verification sweep green.
5. Report final commit list: `git log --grep "Phase 7-C-2" --oneline`.

---

## Self-Review Notes

**Spec coverage:**
- §3.1 (unified generator) → Task 2.
- §3.2 (`top_k_ids_and_weights` helpers in both engines) → Task 1.
- §3.3 (fixture format) → Task 2 defines; Task 3 produces; Task 4 consumes.
- §3.4 (comparison test) → Task 5.
- §4 (tokenizer equivalence) → Task 3 Step 4 check + Task 5 test-level assertion.
- §5 (testing tiers) → Tasks 1, 4, 5 cover each tier.
- §6 (success criteria #1–9) → all mapped: #1=Task 1, #2=Task 2, #3=Task 3 Mac, #4=Task 3 Spark, #5=Task 5, #6=Task 4, #7=Task 6 Step 3, #8-9=Task 6 Steps 1-2.
- §7 (threshold tuning) → Task 5 reports observed numbers; Task 6 memory records them; tightening is designed-in follow-up work, not in this plan.
- §8 (decision log) → transparently carried through via task content.

**Placeholder scan:**
- "Fill in observed numbers" in Task 5/6 — these are deliberately lookup-on-completion (we don't know what the bars will produce until we run). Not a plan failure; just a value-substitution at reporting time.
- Task 1 Step 7 has a verify-API step with conditional Step 8 depending on result. Spelled out all three code variants (topk-values-only, topk-tuple, argsort-only) so the engineer has concrete code regardless of MLX version.
- No "TBD" / "add error handling" / "similar to Task N" patterns.

**Type consistency:**
- `top_k_ids_and_weights(logits, k=5) -> tuple[list[int], list[float]]` identical signature across both engines (Task 1) and consumed by the generator (Task 2) and the test (Task 4 indirectly, Task 5 via JSON).
- Fixture shape: `prompts[i].top_k_per_position[j] = {"ids": list[int], "weights": list[float]}` consistent across Tasks 2, 3, 4, 5.
- `fixture["prompts"][i]["top_k_per_position"][j]["ids"][0]` is top-1 — used in Task 4 (Tier-1 regression) and Task 5 (cross-backend position-0 assertion). Same path both places.

No type/signature drift. All referenced functions/fields exist in earlier tasks.

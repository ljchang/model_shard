# Phase 7-C-1 Real HF Integration + DGX Spark Tier-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PyTorchBackend` generate coherent tokens on real Gemma 4 26B A4B weights by threading `position_embeddings` + `position_ids` + `past_key_values` through `run_layer_atomic` and `pt_moe.*`, matching HF's exact `Gemma4TextDecoderLayer.forward` sequence. Single-node Tier 1 passes on DGX Spark with a committed fixture-based regression test.

**Architecture:** Repurpose the Backend protocol's `masks: tuple[Any, Any]` slot to carry `(cos, sin)` rotary embeddings on PyTorch side. `make_masks` computes them once per iteration; `run_layer_atomic` + `pt_moe.*` unpack + derive `position_ids` from cache state internally. MLX path unchanged (still returns `(global_mask, sliding_mask)` in the same slot). Orchestrator gains an outer residual at the end of `run_split_layer`.

**Tech Stack:** Python 3.13, `transformers >= 5.5.0` (`Gemma4ForCausalLM`, `Gemma4TextDecoderLayer`, `DynamicCache`, `Gemma4TextConfig`), `torch >= 2.6`. All Phase 7-B modules extended in place.

**Spec:** `docs/superpowers/specs/2026-04-19-phase7c1-real-hf-integration-design.md` — decisions D1-D7.

---

## File Structure

**Create:**
- `docs/superpowers/reference/2026-04-19-hf-gemma4-forward-signatures.md` — Task 1 output; verbatim HF signatures + layernorm order; subsequent tasks cite this file.
- `tests/test_pytorch_tiny_hf_integration.py` — slow-marked CPU integration test (Task 5).

**Modify:**
- `src/model_shard/pytorch_engine.py` — `make_masks` returns `(cos, sin)`; `run_layer_atomic` uses full HF layer signature (Task 2).
- `tests/test_pytorch_engine.py` — synthetic `_SynthModel` gains `rotary_emb`; `_SynthLayer` accepts HF-shaped kwargs (Task 2).
- `src/model_shard/pt_moe.py` — four functions updated to HF-correct forward (Task 3).
- `tests/test_pt_moe_unit.py` — synthetic `_SynthModel` gains `rotary_emb`; `_SynthDecoderLayer.self_attn` accepts HF-shaped kwargs; new pre-norm assertions (Task 3).
- `src/model_shard/expert_orchestrator.py` — outer residual at end of `run_split_layer` (Task 4).
- `src/model_shard/moe.py` — audit only; adjust only if Task 4 case-analysis requires it.
- `tests/fixtures/pytorch_tier1_tokens.json` — replaces the 7-B placeholder with Spark-generated data (Task 6).
- `README.md` — Phase 7-C-1 status paragraph (Task 6).
- `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md` — Phase 7-C-1 COMPLETE entry (Task 6).

---

## Task ordering

1. HF source research dump — unblocks Tasks 2, 3, 5.
2. `pytorch_engine.run_layer_atomic` + `make_masks` rework (atomic-layer path).
3. `pt_moe.*` HF-correct forward (split-layer path).
4. Orchestrator outer residual (both paths).
5. Tiny-HF integration test (catches any remaining real-HF mismatch on Mac CPU).
6. DGX Spark fixture + Tier-1 regression + README + memory.

---

### Task 1: HF source research dump

**Files:**
- Create: `docs/superpowers/reference/2026-04-19-hf-gemma4-forward-signatures.md`

This task produces a reference document used by Tasks 2, 3, 5. No code changes. No tests.

- [ ] **Step 1: Locate the HF source locally**

The transformers package is installed via `uv sync --extra pytorch`. Find the Gemma4 modeling file:

```bash
cd /Users/lukechang/Github/model_shard
find .venv -path '*transformers/models/gemma4/modeling_gemma4.py'
find .venv -path '*transformers/models/gemma4/configuration_gemma4.py'
```

Note the exact paths (they'll look like `.venv/lib/python3.13/site-packages/transformers/models/gemma4/modeling_gemma4.py`).

- [ ] **Step 2: Extract the exact `Gemma4TextDecoderLayer.forward` signature + body**

Open `modeling_gemma4.py` and locate `class Gemma4TextDecoderLayer`. Record:

1. The full `forward` method signature — parameter names, defaults, ordering.
2. The complete body from `def forward` through `return`. Include every layernorm call, residual, and the MoE-block conditional (`if self.enable_moe_block:` or equivalent).
3. The exact return shape — tuple or plain tensor, what fields.
4. Whether `past_key_value` or `past_key_values` (singular vs plural) is the accepted kwarg name.

- [ ] **Step 3: Extract `Gemma4TextAttention.forward` signature + return**

In the same file, locate `class Gemma4TextAttention`. Record:
- `forward` signature (all params + defaults).
- What the return tuple contains (`attn_output`, `attn_weights`, `past_key_value`?).
- Whether `position_embeddings` is accepted as `(cos, sin)` tuple or some other shape.

- [ ] **Step 4: Extract the router class**

Locate `class Gemma4TextRouter`. Record:
- Attribute names (`norm`, `proj`, `scale`, `per_expert_scale` etc.).
- `forward` signature and return shape (`(top_k_ids, top_k_weights)` or `(top_k_weights, top_k_ids)` order matters).

- [ ] **Step 5: Extract rotary embedding attribute**

Locate `class Gemma4TextModel` (or `Gemma4ForCausalLM`). Find the rotary embedding module — look for `self.rotary_emb = ...` or similar in `__init__`. Record:
- Attribute path: `model.rotary_emb` vs `model.model.rotary_emb` vs `model.model.embed_tokens.rotary_emb` etc.
- `forward` signature: `rotary_emb(hidden_states, position_ids)` or `rotary_emb(x, pos_ids)` — record exact param names.
- Return shape: `(cos, sin)` tuple of `[1, seq_len, head_dim]` tensors (verify).

- [ ] **Step 6: Extract `Gemma4TextConfig` required fields**

Open `configuration_gemma4.py`. Find `class Gemma4TextConfig`. Record:
- Every `__init__` parameter that does NOT have a default (these are required).
- Parameters with defaults that are LIKELY required for a valid model construction (e.g. `num_attention_heads`, `head_dim`, `num_key_value_heads`, `intermediate_size`, `moe_intermediate_size`, `num_experts`, `top_k_experts`, `layer_types`, `sliding_window`, `vocab_size`, `hidden_size`, `num_hidden_layers`, `max_position_embeddings`, `rms_norm_eps`, `rope_theta`, etc.).
- Any `enable_moe_block` or equivalent flag and where it's consumed.

- [ ] **Step 7: Confirm `DynamicCache.get_seq_length()` availability**

In `.venv/.../transformers/cache_utils.py`, confirm `DynamicCache` class has:
- `get_seq_length()` method — returns int.
- Alternative: `seen_tokens` attribute.

Record whichever the installed version exposes.

- [ ] **Step 8: Write the reference file**

Create `docs/superpowers/reference/2026-04-19-hf-gemma4-forward-signatures.md` with this structure:

```markdown
# HF Gemma 4 Forward Signatures (reference for Phase 7-C-1 tasks)

Extracted from `transformers==<version>` at `.venv/.../transformers/models/gemma4/modeling_gemma4.py`.

## Gemma4TextDecoderLayer.forward

### Signature
```python
def forward(
    self,
    <paste exact param list here>,
) -> <paste exact return annotation>:
```

### Body (MoE-layer path)
```python
<paste exact body with indentation preserved>
```

### Body (non-MoE layer path)
```python
<if different, paste>
```

### Return shape
<describe>

## Gemma4TextAttention.forward

### Signature
```python
<paste>
```

### Return tuple
<describe — attn_output? attn_weights? past_key_value?>

## Gemma4TextRouter

### Attributes
<list>

### forward signature + return
```python
<paste>
```

## Rotary embeddings

### Attribute path
`model.<path>.rotary_emb`

### forward signature
```python
<paste>
```

### Return shape
<describe>

## Gemma4TextConfig

### Required fields (no default)
<list>

### Recommended minimum-viable tiny config
```python
Gemma4TextConfig(
    <enumerate every field set + value>
)
```

## DynamicCache

### Seq length API
`cache.<method-or-attr>()` returns int.

## Notes / gotchas
<anything surprising>
```

Write concrete facts, not aspirations. Every claim should be derivable from a file + line ref in the venv. If something is unclear, say "could not determine; investigate in Task X" rather than guessing.

- [ ] **Step 9: Commit**

```bash
git add docs/superpowers/reference/2026-04-19-hf-gemma4-forward-signatures.md
git commit -m "Phase 7-C-1 Task 1: HF Gemma 4 forward-signatures reference"
```

## Context

- **Working directory:** `/Users/lukechang/Github/model_shard`
- **Branch:** `main` (user authorized direct main commits for this phase series)
- **Predecessor commit:** `bc503d8` (Phase 7-C-1 design spec)
- **Plan file:** this file.
- **Spec:** §2.1, §2.2 reference HF signatures that this Task produces.

## Your Job

1. Follow Steps 1-9 exactly. No code writing; this is pure research.
2. The reference file must be specific enough that Tasks 2, 3, 5 can code from it without re-reading HF source.
3. Commit with exact message.
4. Report back with the reference file path and a one-paragraph summary of the most important findings (e.g., "layernorm order is X, router returns (ids, weights), rotary_emb lives at Y").

---

### Task 2: `pytorch_engine.run_layer_atomic` + `make_masks` rework

**Files:**
- Modify: `src/model_shard/pytorch_engine.py`
- Modify: `tests/test_pytorch_engine.py`

Pre-read: Task 1 reference at `docs/superpowers/reference/2026-04-19-hf-gemma4-forward-signatures.md`. Cite attribute paths and kwarg names from there.

- [ ] **Step 1: Update the synthetic test model**

Open `tests/test_pytorch_engine.py`. Find the `_SynthLayer` and `_SynthTextModel` / `_SynthModel` classes. Update:

```python
class _SynthLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.layer_type = "full_attention"
        self.last_kwargs: dict = {}  # for assertion in tests

    def forward(self, hidden_states=None, position_embeddings=None,
                attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=None,
                cache_position=None, **kwargs):
        # Record kwargs for test assertions.
        self.last_kwargs = {
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "past_key_value": past_key_value,
            "use_cache": use_cache,
            "cache_position": cache_position,
        }
        # Behavior: double the input (so existing doubling test still passes).
        return (hidden_states * 2.0,)


class _SynthRotaryEmb(nn.Module):
    """Mimics HF Gemma4's rotary_emb: returns (cos, sin) tuple."""
    def __init__(self, head_dim: int = 4):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, h: torch.Tensor, position_ids: torch.Tensor):
        seq_len = position_ids.shape[-1]
        cos = torch.ones((1, seq_len, self.head_dim))
        sin = torch.zeros((1, seq_len, self.head_dim))
        return cos, sin


class _SynthTextModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_SynthLayer(hidden) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.rotary_emb = _SynthRotaryEmb(head_dim=4)
```

The `_SynthModel.config` already has `num_hidden_layers` and `layer_types`; no change there.

- [ ] **Step 2: Add failing test for `make_masks` returning `(cos, sin)`**

Append to `tests/test_pytorch_engine.py`:

```python
def test_make_masks_returns_cos_sin_tuple():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    cache = pytorch_engine.make_cache(m)
    cos, sin = pytorch_engine.make_masks(m, h, cache)
    # Stub rotary_emb returns [1, seq_len, head_dim=4] tensors.
    assert cos.shape == (1, 3, 4)
    assert sin.shape == (1, 3, 4)


def test_make_masks_advances_position_ids_with_cache():
    """After decoding some tokens, make_masks should use cache_len + seq_len
    to compute position_ids. We verify by constructing a cache with a known
    seq_length and checking that the rotary module was called with shifted
    position_ids."""
    m = _mk_model()
    h = torch.randn((1, 1, 8))
    cache = pytorch_engine.make_cache(m)
    # Simulate 5 tokens already cached via a no-op update.
    # DynamicCache has no public constructor for pre-filled state in all
    # transformers versions, so patch the get_seq_length method.
    class _FakeCache:
        def get_seq_length(self):
            return 5
    cos, sin = pytorch_engine.make_masks(m, h, _FakeCache())
    # Shape for a single new token is [1, 1, head_dim]
    assert cos.shape == (1, 1, 4)


def test_run_layer_atomic_passes_position_embeddings_to_layer():
    m = _mk_model()
    h = torch.ones((1, 3, 8))
    cache = pytorch_engine.make_cache(m)
    cos, sin = pytorch_engine.make_masks(m, h, cache)
    out = pytorch_engine.run_layer_atomic(m, 0, h, cache, cos, sin)
    # Behavior preserved: _SynthLayer still doubles the input.
    assert torch.allclose(out, torch.full((1, 3, 8), 2.0))
    # Kwargs recorded for inspection:
    kwargs = m.model.layers[0].last_kwargs
    assert kwargs["position_embeddings"] is not None
    cos_recv, sin_recv = kwargs["position_embeddings"]
    assert torch.equal(cos_recv, cos)
    assert torch.equal(sin_recv, sin)
    assert kwargs["use_cache"] is True
    assert kwargs["past_key_value"] is cache
    assert kwargs["position_ids"] is not None
    assert kwargs["position_ids"].shape == (1, 3)
    assert kwargs["cache_position"] is not None
    assert kwargs["cache_position"].shape == (3,)
```

Also update the existing `test_run_layer_atomic_doubles_synthetic_layer` test to use the new `make_masks` output:

```python
def test_run_layer_atomic_doubles_synthetic_layer():
    m = _mk_model()
    h = torch.ones((1, 3, 8))
    cache = pytorch_engine.make_cache(m)
    cos, sin = pytorch_engine.make_masks(m, h, cache)
    out = pytorch_engine.run_layer_atomic(m, 0, h, cache, cos, sin)
    assert out.shape == (1, 3, 8)
    assert torch.allclose(out, torch.full((1, 3, 8), 2.0))
```

Update `test_run_layers_delegates_to_run_layer_atomic_for_non_split`:

```python
def test_run_layers_delegates_to_run_layer_atomic_for_non_split():
    m = _mk_model()
    h = torch.ones((1, 3, 8))
    cache = pytorch_engine.make_cache(m)
    masks = pytorch_engine.make_masks(m, h, cache)
    out = pytorch_engine.run_layers(
        m, start_layer=0, end_layer=2, h=h, cache=cache, masks=masks,
        is_split_layer=lambda _: False,
    )
    assert torch.allclose(out, torch.full((1, 3, 8), 4.0))
```

- [ ] **Step 3: Run tests — expect failures**

```bash
uv run pytest tests/test_pytorch_engine.py -v
```

Expected: new tests fail (make_masks returns (None, None)); `test_run_layer_atomic_doubles_synthetic_layer` fails (synth layer now returns tuple; old impl didn't unpack).

- [ ] **Step 4: Rewrite `make_masks` and `run_layer_atomic` in `src/model_shard/pytorch_engine.py`**

Find the existing `make_masks` function. Replace with:

```python
def make_masks(model: Any, h: torch.Tensor, cache: Any) -> tuple[Any, Any]:
    """Compute (cos, sin) rotary position embeddings for the current forward.

    Phase 7-C-1: the Backend protocol's `masks: tuple[Mask, Mask]` slot is
    repurposed for PyTorch to carry (cos, sin). MLX keeps using the slot for
    (global_mask, sliding_mask); same protocol shape, different per-backend
    semantics. Consumers never mix values across backends.
    """
    cache_len = cache.get_seq_length() if cache is not None else 0
    seq_len = h.shape[1]
    device = h.device
    position_ids = torch.arange(
        cache_len, cache_len + seq_len, dtype=torch.long, device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(h, position_ids)
    return cos, sin
```

Find the existing `run_layer_atomic`. Replace with:

```python
def run_layer_atomic(
    model: Any,
    layer_idx: int,
    h: torch.Tensor,
    cache: Any,
    global_mask: Any,
    sliding_mask: Any,
) -> torch.Tensor:
    """Run one decoder layer against the real HF Gemma4TextDecoderLayer.

    Phase 7-C-1: ``global_mask`` / ``sliding_mask`` are repurposed as
    ``(cos, sin)`` rotary embeddings. ``position_ids`` and
    ``cache_position`` derived here from cache state; ``attention_mask``
    left as None (HF builds it from ``cache_position``). ``use_cache=True``
    always (works around transformers bug #45242).
    """
    cos, sin = global_mask, sliding_mask
    layer = model.model.layers[layer_idx]
    cache_len = cache.get_seq_length() if cache is not None else 0
    seq_len = h.shape[1]
    device = h.device
    position_ids = torch.arange(
        cache_len, cache_len + seq_len, dtype=torch.long, device=device,
    ).unsqueeze(0)
    cache_position = torch.arange(cache_len, cache_len + seq_len, device=device)
    with torch.no_grad():
        out = layer(
            hidden_states=h,
            position_embeddings=(cos, sin),
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
        )
    # HF layer returns tuple (hidden_states,) or (hidden_states, attn_weights).
    if isinstance(out, tuple):
        return out[0]
    return out
```

**Note:** if Task 1's reference doc found that HF uses `past_key_values` (plural) as the kwarg name, change `past_key_value=cache` to `past_key_values=cache` both here and in `pt_moe.run_attention_and_route` (Task 3). The research output is authoritative.

- [ ] **Step 5: Run tests to verify pass**

```bash
uv run pytest tests/test_pytorch_engine.py -v
```

Expected: all tests pass (12 existing + 3 new = 15).

- [ ] **Step 6: Ruff + mypy clean**

```bash
uv run ruff check src/model_shard/pytorch_engine.py tests/test_pytorch_engine.py
uv run mypy src/model_shard/pytorch_engine.py
```

Both zero errors. Apply narrow `# noqa` / `# type: ignore` only as needed.

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/pytorch_engine.py tests/test_pytorch_engine.py
git commit -m "Phase 7-C-1 Task 2: pytorch_engine real-HF run_layer_atomic + (cos,sin) via masks tuple"
```

## Context

- **Predecessor commit:** Task 1.
- **Spec:** §2.1.
- **HF bug workaround:** `use_cache=True` always. If HF source shows differently-named kwarg, Task 1 reference has the correct form.

## Your Job

1. Follow Steps 1-7. TDD.
2. 15 tests pass; ruff + mypy clean.
3. Commit.
4. Report back, citing which HF kwarg names you used (e.g. `past_key_value` vs `past_key_values`) based on Task 1's research.

---

### Task 3: `pt_moe.*` HF-correct forward

**Files:**
- Modify: `src/model_shard/pt_moe.py`
- Modify: `tests/test_pt_moe_unit.py`

Pre-read: Task 1 reference document — use it as the source of truth for submodule signatures and layernorm sequence.

- [ ] **Step 1: Update the synthetic test model**

Open `tests/test_pt_moe_unit.py`. Update `_SynthDecoderLayer.__init__` to make `self_attn` accept HF-shaped kwargs (currently it's a bare `nn.Linear`):

```python
class _SynthSelfAttn(nn.Module):
    """Mimics HF Gemma4TextAttention's forward signature."""
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.last_kwargs: dict = {}

    def forward(
        self, hidden_states=None, position_embeddings=None,
        attention_mask=None, position_ids=None,
        past_key_value=None, use_cache=None, cache_position=None,
        **kwargs,
    ):
        self.last_kwargs = {
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "past_key_value": past_key_value,
            "use_cache": use_cache,
            "cache_position": cache_position,
        }
        # Returns (attn_output,) tuple matching HF.
        return (self.proj(hidden_states),)


class _SynthRotaryEmb(nn.Module):
    def __init__(self, head_dim: int = 4):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, h: torch.Tensor, position_ids: torch.Tensor):
        seq_len = position_ids.shape[-1]
        cos = torch.ones((1, seq_len, self.head_dim))
        sin = torch.zeros((1, seq_len, self.head_dim))
        return cos, sin
```

In `_SynthDecoderLayer.__init__`, replace `self.self_attn = nn.Linear(...)` with `self.self_attn = _SynthSelfAttn(hidden)`. Keep all other attributes.

In `_SynthTextModel.__init__`, add `self.rotary_emb = _SynthRotaryEmb(head_dim=4)`.

- [ ] **Step 2: Update existing tests + add new assertions**

Find `test_run_attention_and_route_shapes`. Update to pass `(cos, sin)` in masks:

```python
def test_run_attention_and_route_shapes():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    cos = torch.ones((1, 3, 4))
    sin = torch.zeros((1, 3, 4))
    post_attn, top_k_ids, top_k_weights = pt_moe.run_attention_and_route(
        m, h, layer_idx=0, cache=None, masks=(cos, sin), heat_observer=None,
    )
    assert post_attn.shape == (1, 3, 8)
    assert top_k_ids.shape == (1, 3, 2)
    assert top_k_weights.shape == (1, 3, 2)


def test_run_attention_and_route_passes_kwargs_to_self_attn():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    cos = torch.ones((1, 3, 4))
    sin = torch.zeros((1, 3, 4))

    class _FakeCache:
        def get_seq_length(self):
            return 0

    pt_moe.run_attention_and_route(
        m, h, layer_idx=0, cache=_FakeCache(), masks=(cos, sin), heat_observer=None,
    )
    kwargs = m.model.layers[0].self_attn.last_kwargs
    cos_recv, sin_recv = kwargs["position_embeddings"]
    assert torch.equal(cos_recv, cos)
    assert kwargs["use_cache"] is True
    assert kwargs["position_ids"].shape == (1, 3)
```

Find `test_run_attention_and_route_fires_heat_observer`. Update to pass `(cos, sin)`:

```python
def test_run_attention_and_route_fires_heat_observer():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    cos = torch.ones((1, 3, 4))
    sin = torch.zeros((1, 3, 4))
    calls: list[tuple[int, int, float]] = []
    pt_moe.run_attention_and_route(
        m, h, layer_idx=0, cache=None, masks=(cos, sin),
        heat_observer=lambda L, E, w: calls.append((L, E, float(w))),
    )
    assert len(calls) == 6
    assert all(L == 0 for L, _, _ in calls)
```

Find `test_run_shared_expert_calls_layer_mlp`. Update to assert pre-norm is applied:

```python
def test_run_shared_expert_applies_pre_feedforward_layernorm():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    out = pt_moe.run_shared_expert(m, h, layer_idx=0)
    assert out.shape == (1, 3, 8)
    # Expected value: pre_feedforward_layernorm applied before mlp.
    layer = m.model.layers[0]
    expected = layer.mlp(layer.pre_feedforward_layernorm(h))
    assert torch.allclose(out, expected)
```

Find `test_run_selected_experts_returns_dict_id_to_tensor`. Update to assert pre-norm is applied:

```python
def test_run_selected_experts_returns_dict_id_to_tensor():
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    out = pt_moe.run_selected_experts(m, h, layer_idx=0, expert_ids=[0, 2])
    assert set(out.keys()) == {0, 2}
    for v in out.values():
        assert v.shape == (1, 3, 8)


def test_run_selected_experts_applies_pre_feedforward_layernorm_2():
    """The per-expert MLP should consume pre_feedforward_layernorm_2(h),
    not raw h."""
    m = _mk_model()
    h = torch.randn((1, 3, 8))
    out = pt_moe.run_selected_experts(m, h, layer_idx=0, expert_ids=[1])
    layer = m.model.layers[0]
    normed = layer.pre_feedforward_layernorm_2(h)
    e = layer.experts
    gu = F.linear(normed, e.gate_up_proj[1])
    g, u = gu.chunk(2, dim=-1)
    mid = F.silu(g) * u
    expected = F.linear(mid, e.down_proj[1])
    assert torch.allclose(out[1], expected, atol=1e-5)
```

The old `test_run_selected_experts_per_expert_linear_is_equivalent_to_stacked_index` assumed NO pre-norm; delete it (the new test above supersedes it).

Find `test_aggregate_experts_weights_and_sums_with_shared`. Update:

```python
def test_aggregate_experts_applies_post_feedforward_layernorms():
    """HF structure: dense_normed = post_ff_ln_1(shared); moe_normed =
    post_ff_ln_2(weighted_sum); block_out = dense_normed + moe_normed.
    Outer residual handled by orchestrator, not here."""
    m = _mk_model()
    per_pos_expert_outs = {
        0: torch.full((1, 1, 8), 1.0),
        1: torch.full((1, 1, 8), 2.0),
    }
    ids = [0, 1]
    weights = torch.tensor([[0.25, 0.75]])
    shared = torch.full((1, 1, 8), 10.0)
    out = pt_moe.aggregate_experts(
        m, layer_idx=0,
        expert_outputs=per_pos_expert_outs, top_k_ids=ids,
        top_k_weights=weights, shared_out=shared,
    )
    layer = m.model.layers[0]
    moe_branch = 0.25 * per_pos_expert_outs[0] + 0.75 * per_pos_expert_outs[1]
    expected = (
        layer.post_feedforward_layernorm_1(shared)
        + layer.post_feedforward_layernorm_2(moe_branch)
    )
    assert out.shape == (1, 1, 8)
    assert torch.allclose(out, expected, atol=1e-5)
```

- [ ] **Step 3: Run tests — expect failures**

```bash
uv run pytest tests/test_pt_moe_unit.py -v
```

- [ ] **Step 4: Rewrite `src/model_shard/pt_moe.py`**

Replace the four functions. Full file content (keeping headers / imports unchanged):

```python
"""Phase 7-B + 7-C-1: PyTorch MoE primitives for Gemma 4 split layers.

Mirror of moe.py. Matches HF Gemma4TextDecoderLayer.forward exactly when
the four functions below are composed in order (run_split_layer call flow):
attention+route → shared + selected experts → aggregate.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


HeatObserver = Callable[[int, int, float], None] | None


def _layer(model: Any, layer_idx: int) -> Any:
    return model.model.layers[layer_idx]


def _run_one_expert(
    h: torch.Tensor, gate_up_k: torch.Tensor, down_k: torch.Tensor,
) -> torch.Tensor:
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
    """HF Gemma4TextDecoderLayer attention sub-block + router.

    Returns (post_attn [B,L,H], top_k_ids [B,L,K], top_k_weights [B,L,K]).
    The caller then runs shared + selected experts + aggregate.
    """
    layer = _layer(model, layer_idx)
    cos, sin = masks
    cache_len = cache.get_seq_length() if cache is not None else 0
    seq_len = h.shape[1]
    device = h.device
    position_ids = torch.arange(
        cache_len, cache_len + seq_len, dtype=torch.long, device=device,
    ).unsqueeze(0)
    cache_position = torch.arange(cache_len, cache_len + seq_len, device=device)
    with torch.no_grad():
        residual = h
        x = layer.input_layernorm(h)
        attn_out = layer.self_attn(
            hidden_states=x,
            position_embeddings=(cos, sin),
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h2 = residual + attn_out
        post_attn = layer.post_attention_layernorm(h2)
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
    """HF Gemma4 MoE block: pre_feedforward_layernorm(post_attn) → mlp(). The
    post_feedforward_layernorm_1 wrap happens in aggregate_experts.
    """
    layer = _layer(model, layer_idx)
    with torch.no_grad():
        normed = layer.pre_feedforward_layernorm(h)
        out = layer.mlp(normed)
    return out  # type: ignore[no-any-return]


def run_selected_experts(
    model: Any, h: torch.Tensor, layer_idx: int, expert_ids: list[int],
) -> dict[int, torch.Tensor]:
    """Per-expert MoE path: pre_feedforward_layernorm_2(post_attn) →
    per-expert gated MLP. post_feedforward_layernorm_2 is applied in
    aggregate_experts on the weighted sum. Bypasses HF's
    MixtralExperts.forward so the distributed engine can fan out."""
    layer = _layer(model, layer_idx)
    experts = layer.experts
    with torch.no_grad():
        normed = layer.pre_feedforward_layernorm_2(h)
        out: dict[int, torch.Tensor] = {}
        for k in expert_ids:
            out[int(k)] = _run_one_expert(
                normed, experts.gate_up_proj[k], experts.down_proj[k],
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
    """HF Gemma4 MoE combine: post_ff_ln_1(shared) + post_ff_ln_2(weighted_sum).
    The outer residual (post_attn + block_out) is applied by
    ExpertOrchestrator.run_split_layer, NOT here.
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
    return dense_normed + moe_normed
```

**Note:** if Task 1 research found that router returns `(weights, ids)` order instead of `(ids, weights)`, flip the unpack `top_k_ids, top_k_weights = layer.router(router_in)` accordingly. The research document is authoritative.

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_pt_moe_unit.py -v
```

All pass. Existing Phase 7-B tests that this file supersedes should also pass (or be replaced by the new assertions above).

- [ ] **Step 6: Ruff + mypy**

```bash
uv run ruff check src/model_shard/pt_moe.py tests/test_pt_moe_unit.py
uv run mypy src/model_shard/pt_moe.py
```

- [ ] **Step 7: Commit**

```bash
git add src/model_shard/pt_moe.py tests/test_pt_moe_unit.py
git commit -m "Phase 7-C-1 Task 3: pt_moe HF-correct forward (pre-norms + outer residual deferred to orchestrator)"
```

## Context

- **Predecessor commit:** Task 2.
- **Spec:** §2.2.
- **Outer residual:** intentionally NOT added here — Task 4 adds it in `ExpertOrchestrator.run_split_layer` so MLX and PyTorch share the same convention.

## Your Job

1. Follow Steps 1-7. TDD.
2. Tests pass; ruff + mypy clean.
3. Commit.
4. Report back, noting any deviations from Task 1's research-based expectations (e.g. if router return order differed).

---

### Task 4: `ExpertOrchestrator.run_split_layer` outer residual

**Files:**
- Modify: `src/model_shard/expert_orchestrator.py`
- Modify: `src/model_shard/moe.py` (audit only; edit only if MLX regression fails)

- [ ] **Step 1: Audit MLX `moe.py` aggregate_experts for outer-residual handling**

```bash
cd /Users/lukechang/Github/model_shard
grep -n "aggregate_experts\|post_attn\|residual" src/model_shard/moe.py
```

Read the `aggregate_experts` function body. Determine:

- Case A: MLX `aggregate_experts` already includes `+ post_attn` (outer residual baked in). Adding `+ post_attn` in orchestrator on the PyTorch branch would double-apply; must guard with `isinstance(self.backend, PyTorchBackend)`.
- Case B: MLX relies on orchestrator to add outer residual. Orchestrator adds unconditionally; MLX tests were either passing by accident or MLX already has an equivalent addition elsewhere.
- Case C: MLX doesn't need the outer residual (convention difference). Guard with `isinstance`.

Record which case by reading source AND running one MLX slow bucket in baseline (pre-edit):

```bash
uv run pytest -m slow -q tests/test_tier1_tokens.py
```

Must be green before proceeding — confirms baseline.

- [ ] **Step 2: Apply the outer residual**

Open `src/model_shard/expert_orchestrator.py`. Find `run_split_layer`. The method's Phase C block currently looks (after Phase 7-B refactor) roughly like:

```python
with self._mlx_guard():
    agg = self.backend.aggregate_experts(
        layer_idx, per_pos, ids, weights, per_pos_shared,
    )
# ... agg is returned or fed onward ...
```

Locate where `agg` becomes the final layer output. Add the outer residual:

```python
with self._mlx_guard():
    agg = self.backend.aggregate_experts(
        layer_idx, per_pos, ids, weights, per_pos_shared,
    )
    # Phase 7-C-1: outer residual post_attn + block_out matches HF
    # Gemma4 MoE block structure. MLX path must preserve its existing
    # convention — if Task 4 audit found MLX already bakes this in,
    # guard with isinstance(self.backend, PyTorchBackend).
    agg = <post_attn_for_this_position> + agg
```

The exact variable name for `post_attn_for_this_position` depends on the current orchestrator structure. Likely: Phase A produced `post_attn` for the whole `[B, L, H]` hidden; the per-position aggregate is `agg` for one position; the per-position `post_attn_slice` is `post_attn[b, ll, :]` or similar. Read the current code to match variable scoping exactly.

**If Task 1 audit was Case A or Case C (MLX already baked in):** wrap the addition in `if isinstance(self.backend, PyTorchBackend):` using the existing `from model_shard.backends import PyTorchBackend` import (add if missing).

**If Case B:** add unconditionally.

- [ ] **Step 3: Fast regression on orchestrator tests**

```bash
uv run pytest tests/test_expert_orchestrator.py tests/test_expert_retry_unit.py tests/test_expert_rpc_load_shift.py tests/test_orchestrator_live_owners.py tests/test_routing_correctness.py -v -m "not slow"
```

All pass.

- [ ] **Step 4: MLX slow regression (CRITICAL — non-negotiable)**

```bash
uv run pytest -m slow -q tests/test_tier1_tokens.py
uv run pytest -m slow -q tests/test_partial_load_bit_exact_per_expert.py
uv run pytest -m slow -q tests/test_migration_over_tcp.py
uv run pytest -m slow -q tests/test_expert_retry_bit_exact.py
uv run pytest -m slow -q tests/test_provenance_tier1.py
uv run pytest -m slow -q tests/test_eviction_e2e.py
```

Every bucket must be green. If any fails after the outer-residual edit, either:
- The case analysis was wrong — revise the `isinstance` guard.
- The MLX path needs the addition too — not guarded.

STOP and investigate rather than pushing through.

- [ ] **Step 5: Ruff + mypy**

```bash
uv run ruff check src/model_shard/expert_orchestrator.py
uv run mypy src/model_shard/expert_orchestrator.py
```

- [ ] **Step 6: Commit**

```bash
git add src/model_shard/expert_orchestrator.py
git commit -m "Phase 7-C-1 Task 4: orchestrator outer residual in run_split_layer"
```

If you also had to edit `src/model_shard/moe.py` (rare — audit result), add it to the `git add`.

## Context

- **Predecessor commit:** Task 3.
- **Spec:** §2.3.
- **CRITICAL:** MLX slow regression must stay green. STOP on any regression.

## Your Job

1. Follow Steps 1-6.
2. Report back with: which case (A/B/C) the MLX audit found, how you guarded, per-bucket slow-test results.

---

### Task 5: Tiny-HF integration test on Mac CPU

**Files:**
- Create: `tests/test_pytorch_tiny_hf_integration.py`

Pre-read: Task 1 reference document for `Gemma4TextConfig` minimum-viable tiny-config fields.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pytorch_tiny_hf_integration.py`:

```python
"""Phase 7-C-1 Task 5: real-HF integration test on Mac CPU.

Builds a minimal Gemma4ForCausalLM from a hand-rolled Gemma4TextConfig
(random init, not pretrained) and runs one forward pass through
PyTorchBackend. Catches real-HF integration bugs (wrong kwargs, signature
mismatches, missing layernorms) that synthetic-test coverage misses.

Runs on CPU, takes seconds. Marked slow because it imports transformers
and instantiates a random model.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from model_shard.backends import PyTorchBackend  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Gemma4ForCausalLM
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    # Minimal-viable config. Task 1 reference doc lists required fields;
    # copy any fields it flagged as required here with small values.
    # If transformers raises a "missing field" error, consult
    # docs/superpowers/reference/2026-04-19-hf-gemma4-forward-signatures.md
    # and add the missing field.
    cfg = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=4,
        top_k_experts=2,
        layer_types=["full_attention", "full_attention"],
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        sliding_window=32,
    )
    model = Gemma4ForCausalLM(cfg)
    model.eval()
    return model


@pytest.mark.slow
def test_pytorch_backend_forward_on_tiny_hf_model(tiny_model):
    """End-to-end: embed → make_masks → run_layer_atomic × N → finalize
    on a real (tiny, random-init) HF Gemma4ForCausalLM."""
    b = PyTorchBackend.from_loaded_model(tiny_model, device="cpu")
    token_ids = [5, 6, 7, 8]
    cache = b.make_cache()
    h = b.embed(token_ids)
    assert h.shape == (1, 4, 64)
    masks = b.make_masks(h, cache)
    num_layers = b.num_layers()
    assert num_layers == 2
    for i in range(num_layers):
        h = b.run_layer_atomic(i, h, cache, masks)
        assert h.shape == (1, 4, 64)
    logits = b.finalize(h)
    assert logits.shape == (1, 4, 256)
    # Greedy-decode 1 token via argmax_last.
    token_id = b.argmax_last(logits)
    assert 0 <= token_id < 256
    # No NaN / Inf in logits.
    assert torch.isfinite(logits).all()


@pytest.mark.slow
def test_pytorch_backend_two_step_decode_on_tiny_hf_model(tiny_model):
    """Prefill + one decode step, confirming cache grows correctly."""
    b = PyTorchBackend.from_loaded_model(tiny_model, device="cpu")
    cache = b.make_cache()
    prompt_ids = [1, 2, 3]

    # Prefill
    h = b.embed(prompt_ids)
    masks = b.make_masks(h, cache)
    for i in range(b.num_layers()):
        h = b.run_layer_atomic(i, h, cache, masks)
    logits = b.finalize(h)
    token_id = b.argmax_last(logits)

    # Decode step 1
    h = b.embed([token_id])
    assert h.shape == (1, 1, 64)
    masks = b.make_masks(h, cache)
    for i in range(b.num_layers()):
        h = b.run_layer_atomic(i, h, cache, masks)
    logits = b.finalize(h)
    next_id = b.argmax_last(logits)
    assert 0 <= next_id < 256
    # cache should now reflect prefill_len + 1 = 4 tokens.
    assert cache.get_seq_length() == len(prompt_ids) + 1


@pytest.mark.slow
def test_pytorch_backend_moe_layer_via_run_attention_and_route(tiny_model):
    """Exercise the split-layer path via direct pt_moe calls.

    Confirms the HF MoE-block pieces (run_attention_and_route +
    run_shared_expert + run_selected_experts + aggregate_experts) all
    work against the real HF Gemma4 MoE decoder layer."""
    b = PyTorchBackend.from_loaded_model(tiny_model, device="cpu")
    token_ids = [1, 2, 3]
    cache = b.make_cache()
    h = b.embed(token_ids)
    masks = b.make_masks(h, cache)

    # Run layer 0 through the split-layer primitives.
    post_attn, top_k = b.run_attention_and_route(
        layer_idx=0, h=h, cache=cache, masks=masks,
    )
    top_k_ids, top_k_weights = top_k
    assert post_attn.shape == (1, 3, 64)
    assert top_k_ids.shape == (1, 3, 2)
    assert top_k_weights.shape == (1, 3, 2)

    shared_out = b.run_shared_expert(layer_idx=0, h=post_attn)
    assert shared_out.shape == (1, 3, 64)

    # Get the unique set of selected expert ids across all positions
    unique_ids = sorted({int(e) for e in top_k_ids.reshape(-1).tolist()})
    expert_outputs = b.run_selected_experts(
        layer_idx=0, h=post_attn, expert_ids=unique_ids,
    )
    for eid in unique_ids:
        assert expert_outputs[eid].shape == (1, 3, 64)

    # Aggregate (single-position simplification: just verify no exception + finite)
    # The real orchestrator flattens per-position; this is a smoke test only.
    weights_flat = top_k_weights[:, 0:1, :]  # just position 0, [1, 1, K]
    ids_flat = top_k_ids[:, 0, :].reshape(-1).tolist()  # K ids
    shared_pos = shared_out[:, 0:1, :]
    expert_outputs_pos = {
        int(e): expert_outputs[int(e)][:, 0:1, :] for e in ids_flat
    }
    agg = b.aggregate_experts(
        layer_idx=0,
        expert_outputs=expert_outputs_pos,
        top_k_ids=ids_flat,
        top_k_weights=weights_flat,
        shared_out=shared_pos,
    )
    assert agg.shape == (1, 1, 64)
    assert torch.isfinite(agg).all()
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest -m slow tests/test_pytorch_tiny_hf_integration.py -v
```

Expected: 3 PASS, takes under a minute on CPU.

If a test fails with a `Gemma4TextConfig` kwarg error, add the missing field per Task 1's reference doc and re-run.

If a test fails on a real-HF call (e.g. `self_attn` gets an unexpected kwarg), this is the signal that Task 2 or Task 3 had a wrong kwarg name. Fix in the respective module, NOT in the test.

- [ ] **Step 3: Ruff + mypy**

```bash
uv run ruff check tests/test_pytorch_tiny_hf_integration.py
```

(Mypy not needed for test files generally; leave at default config.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_pytorch_tiny_hf_integration.py
git commit -m "Phase 7-C-1 Task 5: tiny-HF integration test on Mac CPU"
```

## Context

- **Predecessor commit:** Task 4.
- **Spec:** §2.4.
- **Scope reminder:** tiny model is random-init. Output values don't matter; only that the forward pass runs cleanly and produces finite logits with correct shapes.

## Your Job

1. Follow Steps 1-4. TDD.
2. 3 tests pass. If any fail on HF kwarg mismatch, the fix goes in Task 2 or Task 3 code, not the test.
3. Commit.
4. Report back with any `Gemma4TextConfig` fields that needed adding beyond the initial list — useful data for future 7-C work.

---

### Task 6: DGX Spark fixture generation + Tier-1 regression

**Files:**
- Modify: `tests/fixtures/pytorch_tier1_tokens.json` (replaces 7-B placeholder)
- Modify: `README.md`
- Modify: `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`

This task requires DGX Spark access. The **user** executes Steps 1-2 manually (on Spark); the **implementer** handles everything else.

- [ ] **Step 1: USER — generate the fixture on Spark**

On DGX Spark (via Tailscale SSH or direct):

```bash
ssh <spark-host>
cd <path>/model_shard
git pull origin main  # pull Tasks 1-5 + the generator script from Phase 7-B
uv sync --extra pytorch
uv run python scripts/generate_pytorch_tier1_fixture.py
# Script writes tests/fixtures/pytorch_tier1_tokens.json
scp tests/fixtures/pytorch_tier1_tokens.json <mac-host>:<path>/model_shard/tests/fixtures/
```

Alternatively: commit from Spark directly if Spark has git push access.

- [ ] **Step 2: USER — verify the fixture on Spark**

Also on Spark, before committing:

```bash
uv run pytest -m slow tests/test_pytorch_tier1.py -v
```

Should now pass (not skip) since the fixture is real. Expected: 1 test passes with 3 prompts × 10 positions of top-1 token agreement.

If the test fails with a generated-but-wrong-fixture issue, investigate before committing the fixture. The most likely cause is a lingering kwarg mismatch from Tasks 2-3; check `scripts/spark_smoke_test.py` output first.

- [ ] **Step 3: IMPLEMENTER — confirm fixture is present locally**

Back on Mac:

```bash
cat tests/fixtures/pytorch_tier1_tokens.json | head -20
```

Should NOT contain `"_placeholder": true`. Should contain `"prompts": [...]` with 3 entries, each with `"prompt_ids"` and `"generated_ids"` arrays of length matching `n_positions`.

- [ ] **Step 4: IMPLEMENTER — add Phase 7-C-1 paragraph to `README.md`**

Read `README.md`, find the Phase 7-B status paragraph, and insert a Phase 7-C-1 paragraph AFTER it. Match existing style (prose, no emojis, ~180 words).

Cover:

- Scope: closes the 7-B synthetic-test gap — `PyTorchBackend` now generates coherent tokens on real Gemma 4 26B A4B weights.
- Technical summary: `make_masks` returns `(cos, sin)` via the existing `masks` tuple slot; `run_layer_atomic` + `pt_moe.*` call HF layer / submodules with full kwargs (`position_embeddings`, `position_ids`, `past_key_value`, `cache_position`, `use_cache=True`); orchestrator adds outer residual in `run_split_layer`.
- Testing: synthetic unit tests updated for new signatures; new `test_pytorch_tiny_hf_integration.py` exercises a minimal real HF Gemma4 on Mac CPU; committed `tests/fixtures/pytorch_tier1_tokens.json` generated on DGX Spark, consumed by `test_pytorch_tier1.py` as a permanent top-1 regression bar.
- MLX path unchanged: all 6 slow regression buckets green post-change.
- Non-goals (deferred to 7-C-2/3/4): cross-backend correctness harness, heterogeneous cluster, Phase 6-B provenance on PyTorch path, remaining tech-debt cleanup.
- Link to spec: `docs/superpowers/specs/2026-04-19-phase7c1-real-hf-integration-design.md`.

- [ ] **Step 5: IMPLEMENTER — update memory file**

Edit `/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md`. Find the Phase 7-B entry and add a Phase 7-C-1 COMPLETE entry after it.

Cover:

- Date `2026-04-19`, final commit SHA (fill in after Step 7).
- 6 tasks done.
- Links to plan + spec + task commits.
- What it enables: real PyTorch token generation on Spark single-node. Unblocks 7-C-2 (cross-backend correctness harness — now possible because PyTorch actually works).
- Technical changes: `(cos, sin)` threading via `masks` tuple; HF-correct forward replication in `pt_moe.*`; outer residual in orchestrator.
- What didn't change: Backend protocol signatures, MLX slow regression, wire protocol, gossip, provenance, retry, eviction.
- Outstanding for 7-C-2/3/4:
  - `lm` param threading in orchestrator (7-C-4 cleanup).
  - `_MLX_COMPUTE_LOCK` alias (7-C-4 cleanup).
  - Cross-backend correctness harness (7-C-2).
  - Heterogeneous cluster + 9-tensor ↔ 2-tensor slice bridge + Phase 6-B provenance on PyTorch (7-C-3).
- Next: Phase 7-C-2 brainstorm.

- [ ] **Step 6: IMPLEMENTER — final verification**

```bash
uv run pytest -q -m "not slow"                              # fast
uv run pytest -m slow -q tests/test_pytorch_tiny_hf_integration.py  # Mac-only
uv run ruff check src tests scripts
uv run mypy src
```

All pass. Document in the commit message if `test_pytorch_tier1.py` was NOT run from Mac (expected — it skips without CUDA); the Spark run in Step 2 is authoritative.

- [ ] **Step 7: IMPLEMENTER — final commit**

```bash
git add README.md tests/fixtures/pytorch_tier1_tokens.json "/Users/lukechang/.claude/projects/-Users-lukechang-Github-model-shard/memory/project_gossip_moe.md"
git commit -m "Phase 7-C-1 Task 6: Spark fixture + README + memory (7-C-1 COMPLETE)"
```

Record the final commit SHA in the memory file via a second edit + amend if desired, OR leave a "see git log" reference.

## Context

- **Predecessor commit:** Task 5.
- **Spec:** §2.5, §4.
- **Split responsibility:** Steps 1-2 run on Spark (user or SSH session). Steps 3-7 run on Mac.
- **If Spark is delayed:** Tasks 1-5 can land as Phase 7-C-1-pre with a "Spark-fixture follow-up" ticket. Task 6 becomes a later micro-commit when Spark is reachable.

## Your Job

1. Orchestrate Steps 1-2 with the user (fixture generation requires their Spark session).
2. Run Steps 3-7 on Mac.
3. Report back with:
   - Fixture commit SHA.
   - README paragraph text.
   - Per-bucket fast-suite and tiny-HF-integration results.
   - The final Phase 7-C-1 commit list (`git log --grep "Phase 7-C-1" --oneline`).
   - Phase 7-C-1 COMPLETE summary for the user.

---

## Self-Review Notes

**Spec coverage:**
- §2.1 (cos/sin via masks) → Task 2.
- §2.2 (pt_moe HF-correct) → Task 3.
- §2.3 (outer residual) → Task 4.
- §2.4 (tiny-HF integration test) → Task 5.
- §2.5 (Spark fixture + Tier-1) → Task 6.
- §4 success criteria (#1–8) — all have matching tasks; #4 (Spark smoke) is covered by the existing `scripts/spark_smoke_test.py` + Task 6 Step 1 which confirms the fixture generator runs (same code path).
- §5 risks — each has a task-level mitigation (kwarg mismatch caught in Task 5; MLX regression in Task 4; Spark access split in Task 6).

**Placeholder scan:**
- No "TBD", "implement later", "handle edge cases". Task 1 is deliberately non-code research; the output file pattern is prescribed.
- Task 4 Step 2 has variable-name placeholder `<post_attn_for_this_position>` — this is a necessary recursion into actual code structure, and Task 4 Step 1 reads the current orchestrator before writing, so the implementer will know the exact name. I flagged that in the note.
- Task 5 `Gemma4TextConfig` field list may need augmentation — explicit fallback instruction given in Step 2.
- Task 6 `scripts/spark_smoke_test.py` is referenced but not re-specified — it shipped in 7-B Task 7 and is unchanged here.

**Type consistency:**
- `make_masks` returns `tuple[Any, Any]` across Tasks 2/3/5. Content is `(cos, sin)` for PyTorch throughout.
- `run_layer_atomic(model, layer_idx, h, cache, global_mask, sliding_mask)` signature consistent Tasks 2/5.
- `pt_moe.run_attention_and_route` returns `(post_attn, top_k_ids, top_k_weights)` — 3-tuple — consistent Tasks 3/5.
- `PyTorchBackend.run_attention_and_route` returns `(post_attn, (top_k_ids, top_k_weights))` — nested — unchanged from 7-B, consistent Task 5 consumer.
- `aggregate_experts` call signature (`layer_idx`, `expert_outputs`, `top_k_ids`, `top_k_weights`, `shared_out`) consistent Tasks 3/5.
- `isinstance(self.backend, PyTorchBackend)` gating pattern in Task 4 uses the same import path (`model_shard.backends.PyTorchBackend`) as Phase 7-B Task 6.

No type or signature drift. All referenced methods/classes exist in either Phase 7-B code or are defined earlier in this plan.

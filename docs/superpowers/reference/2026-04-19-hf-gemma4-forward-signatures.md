# HF Gemma 4 Forward Signatures — Phase 7-C-1 reference

Extracted from `transformers==5.5.4` at:
- `/Users/lukechang/Github/model_shard/.venv/lib/python3.13/site-packages/transformers/models/gemma4/modeling_gemma4.py`
- `/Users/lukechang/Github/model_shard/.venv/lib/python3.13/site-packages/transformers/models/gemma4/configuration_gemma4.py`
- `/Users/lukechang/Github/model_shard/.venv/lib/python3.13/site-packages/transformers/cache_utils.py`

Scope: the pieces Tasks 2, 3, and 5 need. All verbatim pastes preserve source indentation.

---

## Gemma4TextDecoderLayer (`modeling_gemma4.py:1325`)

### `__init__` — relevant attributes

```python
class Gemma4TextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma4TextConfig | Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4TextRouter(config)
            self.experts = Gemma4TextExperts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
```

### `forward` signature (verbatim)

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
```

### `forward` body (verbatim, full indentation)

```python
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            # Take hidden states before MLP here
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            # Combine mlp and moe outputs
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states
```

### Return

**Plain tensor** — `hidden_states` of shape `(batch, seq, hidden_size)`. NOT a tuple. Annotation is `-> torch.Tensor`.

### Cache kwarg name

- Uses `past_key_values` (plural): **`past_key_values`**
- This is load-bearing for Tasks 2 + 3: our code must call the decoder layer with `past_key_values=...`, not `past_key_value=...`.

### Residual structure (what Task 4 needs)

There are **two** residual adds inside the layer:
1. `residual = hidden_states; ...; hidden_states = residual + hidden_states` (post-attention)
2. `residual = hidden_states; ...; hidden_states = residual + hidden_states` (post-FFN / post-MoE-combine)

Plus an optional third for `hidden_size_per_layer_input`. The final `hidden_states *= self.layer_scalar` multiplies the whole thing by a learned `(1,)` buffer (initialized to 1).

### MoE combining pattern (what Task 3 needs)

```python
if self.enable_moe_block:
    hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)   # normalized dense MLP out
    hidden_states_flat = residual.reshape(-1, residual.shape[-1])          # pre-MLP (post pre_ff_ln) residual
    _, top_k_weights, top_k_index = self.router(hidden_states_flat)
    hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
    hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
    hidden_states_2 = hidden_states_2.reshape(residual.shape)
    hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
    hidden_states = hidden_states_1 + hidden_states_2
```

Key points:
- Dense MLP runs ALWAYS (regardless of MoE flag). It is called on `pre_feedforward_layernorm(residual)`.
- MoE path re-normalizes `residual` (the POST-attention, PRE-`pre_feedforward_layernorm` hidden states) through `pre_feedforward_layernorm_2`. That is, MoE input is `pre_feedforward_layernorm_2(residual_flat)`, NOT the dense MLP output.
- Router input is the RAW `residual_flat` (not normalized).
- Dense and MoE outputs each get their own post-LN (`post_feedforward_layernorm_1`, `post_feedforward_layernorm_2`) before summing.
- The sum then goes through a THIRD post-LN, `self.post_feedforward_layernorm`, before the outer residual add.

---

## Gemma4TextAttention (`modeling_gemma4.py:1126`)

### `forward` signature (verbatim)

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
```

### Return tuple order

1. `attn_output` — `torch.Tensor` shape `(batch, seq, hidden_size)` after `o_proj`.
2. `attn_weights` — `torch.Tensor | None`, only populated when the attention backend returns them (eager), else `None`.

Returned as `return attn_output, attn_weights`. No `past_key_value` in the tuple — cache is mutated in-place on the `Cache` object via `past_key_values.update(key_states, value_states, self.layer_idx)`.

### `position_embeddings` handling

```python
cos, sin = position_embeddings
```

The attention layer UNPACKS `position_embeddings` as a `(cos, sin)` tuple directly. Both tensors are used with `apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2)` applied to `query_states` and `key_states` (only on non-kv-shared layers for `key_states`).

### `apply_rotary_pos_emb` signature (`modeling_gemma4.py:734`)

```python
def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    # ...
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)
```

Attention calls it with `unsqueeze_dim=2` because `query_states` / `key_states` have shape `(batch, seq, heads, head_dim)` BEFORE `.transpose(1, 2)`.

### KV cache update (`modeling_gemma4.py:1223`)

```python
if past_key_values is not None and not self.is_kv_shared_layer:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
if self.store_full_length_kv:
    shared_kv_states[self.layer_idx] = key_states, value_states
```

So for non-kv-shared layers, cache is updated by calling `past_key_values.update(k, v, layer_idx)` and the returned (possibly concatenated) `(k, v)` are what attention is computed on.

### Shared-KV logic (load-bearing gotcha)

The attention layer decides at `__init__` if it is a "kv-shared" layer (`is_kv_shared_layer`) based on `num_kv_shared_layers`. Shared layers do NOT have `k_proj`, `v_proj`, `k_norm`, `v_norm` (they pull k/v from `shared_kv_states[self.kv_shared_layer_index]`). Layers that are the last non-shared instance of their layer-type stash their `(k, v)` into `shared_kv_states[self.layer_idx]` for the shared tail to consume later.

For the current Gemma 4 26B checkpoint we target, `num_kv_shared_layers=0` so every layer is non-shared and `shared_kv_states={}` stays empty throughout — but the arg is REQUIRED (positional with default `None`? NO — see signature above, it has NO default; so callers must pass it, even as `{}`).

---

## Gemma4TextRouter (`modeling_gemma4.py:1289`)

### `__init__` attributes (verbatim)

```python
class Gemma4TextRouter(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5
        self.eps = config.rms_norm_eps

        self.norm = Gemma4RMSNorm(self.hidden_size, eps=self.eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))
```

Attribute summary:
- `norm` : `Gemma4RMSNorm(hidden_size, eps=rms_norm_eps, with_scale=False)` — **no learnable scale** (pure RMS divide, no gamma).
- `proj` : `nn.Linear(hidden_size, num_experts, bias=False)`.
- `scale` : `nn.Parameter(torch.ones(hidden_size))` — applied AFTER `norm`, multiplied element-wise.
- `per_expert_scale` : `nn.Parameter(torch.ones(num_experts))` — applied to top-k weights after topk selection (indexed by `top_k_index`).
- `scalar_root_size` : float `hidden_size ** -0.5` — applied alongside `scale` as a post-norm scalar factor.

### `forward` (verbatim)

```python
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)  # [B*S, E]
        router_probabilities = nn.functional.softmax(expert_scores, dim=-1)

        # topk returns both values (probabilities) and indices directly
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )  # both [B*S, K]

        # Normalize the top-k weights so they sum to 1 per token
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        # Apply per-expert scale directly to the weights
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return router_probabilities, top_k_weights, top_k_index
```

### Return order — WARNING, three values not two

The return is a 3-tuple, NOT 2: **`(router_probabilities, top_k_weights, top_k_index)`**.

The decoder layer discards the first with `_, top_k_weights, top_k_index = self.router(...)`. The python annotation `-> tuple[torch.Tensor, torch.Tensor]` is incorrect (stale annotation vs code). Our code should unpack as `_, top_k_weights, top_k_index = ...` exactly like the HF decoder layer.

Within the (weights, index) pair the order is:
- `top_k_weights` first — the per-token, per-top-k normalized + per-expert-scaled probability.
- `top_k_index` second — the expert ids of shape `[B*S, K]`.

The Router's `Gemma4TextExperts` call site is `self.experts(hidden_states_2, top_k_index, top_k_weights)` — i.e. index first, weights second, when passing to experts.

### Router input

Router is called on `hidden_states_flat = residual.reshape(-1, residual.shape[-1])` where `residual` is the post-attention-residual hidden states (i.e. what exists just before `self.pre_feedforward_layernorm`). Router input is NOT pre-normalized by the caller; the internal `self.norm` in the router handles normalization.

---

## Gemma4TextRotaryEmbedding (`modeling_gemma4.py:1035`)

### Attribute path

`model.model.rotary_emb` (single instance on `Gemma4TextModel`). Verified:
```python
m = Gemma4ForCausalLM(config)
hasattr(m.model, 'rotary_emb')  # True
```
So from a `Gemma4ForCausalLM` the path is `m.model.rotary_emb`. From a `Gemma4TextModel` it is `self.rotary_emb`.

### `forward` signature (verbatim)

```python
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

### Signature summary (load-bearing)

`forward(x, position_ids, layer_type=None) -> (cos, sin)`

- **Does `forward(x, position_ids)` alone hold?** NO. There is a third argument `layer_type` required to select the per-type inv-freq buffer. Passing `layer_type=None` fails because `getattr(self, f"{None}_inv_freq")` does not exist. Gemma 4's rotary is per-layer-type.
- Inside the model, called as `self.rotary_emb(hidden_states, position_ids, layer_type)` for each unique layer type in the config (modeling_gemma4.py:1613).

### Return shape (empirically verified)

Ran with `hidden_size=64, num_attention_heads=4, head_dim=16, global_head_dim=512` (default), `max_pos=64`, seq=8, `layer_type='full_attention'`:

- `cos.shape` = `torch.Size([1, 8, 512])`
- `sin.shape` = `torch.Size([1, 8, 512])`
- `dtype` = `torch.float32` (the default construction dtype of `x`).

Shape: `(batch, seq, rotary_dim)` where `rotary_dim` is determined by the inv-freq length for that layer type. For `full_attention` Gemma 4 uses `rope_type=proportional` and `partial_rotary_factor=0.25`, where the head dim key is `global_head_dim` (512), so the RoPE dim here is the full 512 (doubled via `torch.cat((freqs, freqs), dim=-1)`). This is WHY `apply_rotary_pos_emb` is broadcast-safe even though our `head_dim` might be smaller — in this particular build cos/sin is sized to `global_head_dim` regardless of the per-head split.

**Caveat for Task 3**: the cos/sin last-dim is NOT necessarily equal to `config.head_dim`. For non-sliding layers, it is derived from `global_head_dim`. For sliding layers it uses `head_dim`. The attention layer just does `x * cos + rotate_half(x) * sin` and relies on `x.shape[-1]` matching cos/sin's last dim — which it does because attention itself picks `head_dim = global_head_dim if non-sliding else head_dim` at init (modeling_gemma4.py:1137). Our engine must select the same `head_dim` per layer.

---

## Gemma4TextConfig (`configuration_gemma4.py:87`)

### Full field list (from dataclass)

```python
vocab_size: int = 262_144
hidden_size: int = 2304
intermediate_size: int = 9216
num_hidden_layers: int = 30
num_attention_heads: int = 8
num_key_value_heads: int = 4
head_dim: int = 256
hidden_activation: str = "gelu_pytorch_tanh"
max_position_embeddings: int = 131_072
initializer_range: float = 0.02
rms_norm_eps: float = 1e-6
use_cache: bool = True
pad_token_id: int | None = 0
eos_token_id: int | list[int] | None = 1
bos_token_id: int | None = 2
tie_word_embeddings: bool = True
rope_parameters: dict | None = None      # filled by __post_init__ with default full/sliding dicts
attention_bias: bool = False
attention_dropout: int | float | None = 0.0
sliding_window: int = 512
layer_types: list[str] | None = None     # filled by __post_init__ with 5:1 sliding pattern
final_logit_softcapping: float | None = None
use_bidirectional_attention: Literal["all", "vision"] | None = None
vocab_size_per_layer_input: int = 262_144
hidden_size_per_layer_input: int = 256
num_global_key_value_heads: int | None = None
global_head_dim: int = 512
attention_k_eq_v: bool = False
num_kv_shared_layers: int = 0
enable_moe_block: bool = False
use_double_wide_mlp: bool = False
num_experts: int | None = None
top_k_experts: int | None = None
moe_intermediate_size: int | None = None
```

### Strictly required fields

None — every field has a default. `Gemma4TextConfig()` with no args is valid.

### Minimum-viable tiny config for MoE layers (verified)

Verified to instantiate `Gemma4ForCausalLM(c)` with ~905k params and run a tiny forward on random input to produce logits of shape `(1, 8, 256)`:

```python
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
c = Gemma4TextConfig(
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
    sliding_window=32,
    enable_moe_block=True,
    hidden_size_per_layer_input=0,   # disables the per-layer-input residual branch
)
```

Forward was verified with:
```python
import torch
from transformers import Gemma4ForCausalLM
m = Gemma4ForCausalLM(c).eval()
ids = torch.randint(0, 256, (1, 8))
with torch.no_grad():
    out = m(ids)
# out.logits.shape == (1, 8, 256)
```

### Fields that needed iteration / surprises

- Stdout during construction emits: `Unrecognized keys in rope_parameters for 'rope_type'='default': {'sliding_attention', 'full_attention'}` — a harmless warning from `PreTrainedConfig` post-init that peeks at `rope_parameters` globally. Safe to ignore.
- `hidden_size_per_layer_input=0` must be set to disable the per-layer-input branch (default is 256). Leaving it at default forces you to also provide `per_layer_inputs` or have `get_per_layer_inputs` wire itself up via `vocab_size_per_layer_input`. For our tiny Mac test, set to 0.
- `layer_types` can be explicitly set to `["full_attention", "full_attention"]`. If you omit it, `__post_init__` builds a 5:1 sliding pattern where the last layer is forced to full_attention. For tests with only `num_hidden_layers=2`, the default pattern becomes `["sliding_attention", "full_attention"]` — which is fine but means you need `sliding_window` tuned.
- `num_experts` / `top_k_experts` / `moe_intermediate_size` are optional at the config level (default None) but REQUIRED if `enable_moe_block=True`; otherwise Gemma4TextExperts init and the router's `nn.Linear(..., num_experts)` fail.
- `rope_parameters` leave as None — `__post_init__` fills with correct default per-layer-type dict.

---

## DynamicCache (`cache_utils.py`)

### Verified API

```python
from transformers import DynamicCache
c = DynamicCache()
c.get_seq_length()   # -> 0        (API exists)
hasattr(c, 'seen_tokens')   # False — attribute removed
c.key_cache   # None          (attribute exists but None until first update)
c.update(k, v, layer_idx)  # standard update API, returns (k_concat, v_concat)
```

Relevant public attrs/methods observed on a fresh instance:

```
batch_repeat_interleave, batch_select_indices, crop, early_initialization,
get_mask_sizes, get_max_cache_shape, get_seq_length, has_previous_state,
is_compileable, is_initialized, is_sliding, layer_class_to_replicate,
layers, max_batch_size, max_cache_len, offload, offloading, prefetch,
reorder_cache, reset, update, update_conv_state, update_recurrent_state
```

### Takeaways

- Use `cache.get_seq_length()`, NOT `cache.seen_tokens`. The latter no longer exists on `transformers==5.5.4`.
- `key_cache` is `None` until first update; do not assume it is a list.
- Gemma4TextModel constructs its own cache via `past_key_values = DynamicCache(config=self.config)` when `use_cache=True and past_key_values is None`.

---

## Additional notes / gotchas for Tasks 2, 3, 5

### Decoder layer call shape (what Task 2's `run_layer_atomic` must match)

When Task 2 calls an HF decoder layer instance directly, the call must be:

```python
hidden_states = decoder_layer(
    hidden_states,                         # positional arg 0
    per_layer_input,                       # positional arg 1 (None if hidden_size_per_layer_input == 0)
    shared_kv_states=shared_kv_states,     # keyword; pass {} for non-kv-shared models
    position_embeddings=position_embeddings[layer_type],   # (cos, sin) tuple for THIS layer's type
    attention_mask=causal_mask_mapping[layer_type],
    position_ids=position_ids,
    past_key_values=past_key_values,       # PLURAL
    **kwargs,
)
```

The return is a plain tensor, not a tuple. Do not unpack `[0]`.

### Rotary embedding call is per-layer-type, NOT per-layer-index

The orchestrator / caller must:
1. For each unique layer type in config, call `rotary_emb(hidden_states, position_ids, layer_type)` and store in a dict keyed by layer_type.
2. At each decoder layer, look up `position_embeddings[self.config.layer_types[layer_idx]]`.

This matters for Task 2 — caching the `(cos, sin)` tuple once per unique layer type, not reconstructing per-layer.

### Attention mask is a dict-per-layer-type

`Gemma4TextModel.forward` builds `causal_mask_mapping = {"full_attention": ..., "sliding_attention": ...}` via `create_causal_mask` / `create_sliding_window_causal_mask` and then indexes by layer_type. Task 2's `make_masks` rework must produce this same dict structure.

### Gemma4TextMLP (`modeling_gemma4.py:1016`) — the dense MLP always runs

```python
def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
```

Standard gated MLP (gate_proj + up_proj, hadamard, act_fn, then down_proj). Even when `enable_moe_block=True`, this dense MLP branch runs alongside the MoE branch — they sum. This is different from "typical" MoE patterns where the MLP is replaced.

### Gemma4TextExperts (`modeling_gemma4.py:1249`) weight layout

```python
self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size))
self.down_proj   = nn.Parameter(torch.empty(num_experts, hidden_size, moe_intermediate_size))
```

- 3D parameters, sharded at dim 0 for expert sharding.
- Per-expert loop: `gate, up = nn.functional.linear(x, gate_up_proj[e]).chunk(2, dim=-1)` — note `gate_up_proj[e]` is 2D shape `(2*moe_intermediate, hidden)`, so the Linear produces `(N, 2*moe_intermediate)` and is then chunked.
- `final_hidden_states = torch.zeros_like(hidden_states); final_hidden_states.index_add_(0, token_idx, weighted_expert_out)` — Task 3 must preserve this exact combining logic for bit-exact correctness.

### `use_experts_implementation` decorator

`Gemma4TextExperts` is decorated with `@use_experts_implementation` which may swap the forward for a fused kernel at runtime. For Mac CPU tests this should fall back to the reference Python impl; verify before asserting bit-exactness.

### `use_kernelized_func(apply_rotary_pos_emb)` decorator on attention

`Gemma4TextAttention` is decorated with `@use_kernelized_func(apply_rotary_pos_emb)` (line 1125). This may swap in a kernelized rotary. Mac CPU test should use the reference Python `apply_rotary_pos_emb`.

### `layer_scalar` is a buffer, not a parameter

```python
self.register_buffer("layer_scalar", torch.ones(1))
```

Shape `(1,)`, initialized to 1. Applied via `hidden_states *= self.layer_scalar` at end of decoder layer forward. Trained-model checkpoints may have non-1 values. Our custom layer implementation MUST include this multiply to match HF output bit-exactly.

### Stale return annotation on router

`Gemma4TextRouter.forward` annotation says `-> tuple[torch.Tensor, torch.Tensor]` but actually returns 3 tensors. Trust the code, not the annotation.

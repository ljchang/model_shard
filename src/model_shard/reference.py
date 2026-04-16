"""Single-process reference oracle.

Wraps the loaded Gemma 4 model into a simple API focused on what Phase 1
acceptance tests need:
  * generate_greedy — Tier 1 (exact-match generated tokens)
  * prefill_trace  — Tier 2 (per-layer hidden-state capture)

Both operations run on a fresh KV cache per call — the oracle is stateless
across invocations.
"""

from dataclasses import dataclass

import mlx.core as mx

from .mlx_engine import (
    LoadedModel,
    embed_tokens,
    finalize,
    load_model,
    make_cache,
    make_masks,
    run_layers,
)


@dataclass
class PrefillTrace:
    """One capture per layer boundary plus the final hidden state and logits."""

    prompt_token_ids: list[int]
    layer_inputs: list[mx.array]  # hidden state BEFORE each layer (len == num_layers)
    final_hidden: mx.array         # after text_model.norm
    logits: mx.array               # after LM head + softcap, [1, seq, vocab]


class ReferenceModel:
    def __init__(self, lm: LoadedModel) -> None:
        self._lm = lm

    @classmethod
    def load(cls, hf_id: str) -> "ReferenceModel":
        return cls(load_model(hf_id))

    @property
    def num_layers(self) -> int:
        return self._lm.num_layers

    def tokenize(self, text: str) -> list[int]:
        tokenizer = self._lm.processor.tokenizer
        return list(tokenizer.encode(text, add_special_tokens=False))

    def detokenize(self, token_ids: list[int]) -> str:
        tokenizer = self._lm.processor.tokenizer
        return str(tokenizer.decode(token_ids, skip_special_tokens=True))

    def generate_greedy(
        self, prompt_tokens: list[int], max_new_tokens: int
    ) -> list[int]:
        lm = self._lm
        cache = make_cache(lm)

        # Prefill over the full prompt.
        tokens = mx.array([prompt_tokens])
        h = embed_tokens(lm, tokens)
        global_mask, sliding_mask = make_masks(lm, h, cache)
        h = run_layers(lm, h, 0, lm.num_layers, cache, global_mask, sliding_mask)
        logits = finalize(lm, h)
        next_token = int(mx.argmax(logits[0, -1, :]).item())

        generated: list[int] = [next_token]

        # Decode loop.
        for _ in range(max_new_tokens - 1):
            step_tokens = mx.array([[next_token]])
            h = embed_tokens(lm, step_tokens)
            global_mask, sliding_mask = make_masks(lm, h, cache)
            h = run_layers(lm, h, 0, lm.num_layers, cache, global_mask, sliding_mask)
            logits = finalize(lm, h)
            next_token = int(mx.argmax(logits[0, -1, :]).item())
            generated.append(next_token)

        return generated

    def prefill_trace(self, prompt_tokens: list[int]) -> PrefillTrace:
        """Runs prefill layer-by-layer, capturing the hidden state seen by each layer."""
        lm = self._lm
        cache = make_cache(lm)

        tokens = mx.array([prompt_tokens])
        h = embed_tokens(lm, tokens)
        global_mask, sliding_mask = make_masks(lm, h, cache)

        layer_inputs: list[mx.array] = []
        for i in range(lm.num_layers):
            # Snapshot INPUT to layer i (so consumers can compare shard-boundary state).
            layer_inputs.append(h)
            h = run_layers(lm, h, i, i + 1, cache, global_mask, sliding_mask)

        final_hidden = lm.text_model.norm(h)
        logits = finalize(lm, h)  # applies norm internally, but idempotent

        return PrefillTrace(
            prompt_token_ids=list(prompt_tokens),
            layer_inputs=layer_inputs,
            final_hidden=final_hidden,
            logits=logits,
        )

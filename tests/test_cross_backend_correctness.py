"""Phase 7-C-2 / 7-C-3a: cross-backend top-K agreement.

Compares the committed MLX and PyTorch tier-1 fixtures without loading
any model. Pure JSON diff — runs anywhere (no Apple Silicon / CUDA
required). Marked slow because it requires both fixtures to be present
and committed (generating them requires the real models on two separate
hosts).

Agreement metric (graded, per spec §3.4):

  * Min first-token top-1 matches: 3 of 3 prompts' position-0 top-1
    tokens agree. After the bf16 rebaseline (Phase 7-C-3a) MLX bf16
    on M5 and PyTorch bf16 on Spark consume the same source weights,
    so the position-0 distributions agree on top-1 across all prompts.
    Any disagreement here means something structural broke.
  * Min average top-K overlap: average top-5 intersection size >= 2.5
    across all (prompt, position) pairs. The remaining drift comes from
    MLX vs PyTorch kernel rounding differences and accumulating decode
    divergence (positions 6+ on prompts that branch into low-confidence
    paths see the worst overlap).

Observed agreement post-Phase 7-C-3a (MLX bf16 on M5 vs PyTorch bf16 on
DGX Spark GB10, both consuming google/gemma-4-26B-A4B-it):

  * Position-0 top-1 matches: 3/3 prompts
  * Average top-5 overlap: ~3.07 across 30 (prompt, position) pairs

The pre-rebaseline 4-bit-vs-bf16 numbers were 1/3 and ~1.03 respectively
— the bf16 rebaseline closed most of that gap. The remaining drift is
MLX/PyTorch implementation-level rounding, not weight-level precision.

Floors are below observed values to leave headroom for normal kernel-
rounding variance; tightening to "exact observed" would cause flaky
failures from harmless reruns. A regression (e.g., one backend's
forward path silently breaking) drops these numbers dramatically and
fails loudly.

Markdown side-by-side report is regenerated every test run at
``tests/fixtures/cross_backend_comparison.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MLX_FIXTURE = FIXTURE_DIR / "mlx_tier1_tokens.json"
PT_FIXTURE = FIXTURE_DIR / "pytorch_tier1_tokens.json"
REPORT_FILE = FIXTURE_DIR / "cross_backend_comparison.md"

# See docstring for rationale. These are floors, not targets. Tightened
# in Phase 7-C-3a after the bf16 rebaseline closed the precision gap;
# observed values are 3/3 first-token matches and ~3.07 avg top-5 overlap.
MIN_FIRST_TOKEN_TOP1_MATCHES = 3
MIN_AVERAGE_TOPK_OVERLAP = 2.5


def _load_or_skip(path: Path, label: str) -> Any:
    if not path.exists():
        pytest.skip(f"{label} fixture missing: {path}")
    data = json.loads(path.read_text())
    if data.get("_placeholder"):
        pytest.skip(f"{label} fixture is a placeholder: {path}")
    return data


def _format_per_position_row(
    mp_pos: Any, pp_pos: Any, position: int,
) -> str:
    mlx_ids = mp_pos["ids"]
    pt_ids = pp_pos["ids"]
    overlap = sorted(set(mlx_ids) & set(pt_ids))
    return (
        f"| {position} | {mlx_ids} | {pt_ids} | "
        f"{len(overlap)} ({overlap}) |"
    )


def _write_report(
    mlx: Any, pt: Any, first_matches: int,
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
    lines.append(
        f"Agreement floors: first-token top-1 matches >= "
        f"{MIN_FIRST_TOKEN_TOP1_MATCHES}; "
        f"avg top-{mlx['top_k_recorded']} overlap >= "
        f"{MIN_AVERAGE_TOPK_OVERLAP}."
    )
    lines.append("")
    for mp, pp in zip(mlx["prompts"], pt["prompts"], strict=True):
        lines.append(f"## Prompt: `{mp['prompt']}`")
        lines.append("")
        lines.append("| position | MLX top-K | PyTorch top-K | overlap |")
        lines.append("|---|---|---|---|")
        for i, (mp_pos, pp_pos) in enumerate(
            zip(
                mp["top_k_per_position"], pp["top_k_per_position"], strict=True,
            )
        ):
            lines.append(_format_per_position_row(mp_pos, pp_pos, i))
        lines.append("")
    REPORT_FILE.write_text("\n".join(lines) + "\n")


@pytest.mark.slow
def test_cross_backend_agreement() -> None:
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
        f"{len(mlx['prompts'])} prompts — below floor "
        f"{MIN_FIRST_TOKEN_TOP1_MATCHES}. See {REPORT_FILE} for per-"
        "position top-K diagnostic."
    )
    assert avg_overlap >= MIN_AVERAGE_TOPK_OVERLAP, (
        f"Average top-{mlx['top_k_recorded']} overlap: {avg_overlap:.2f} "
        f"across {len(overlaps)} positions — below floor "
        f"{MIN_AVERAGE_TOPK_OVERLAP}. See {REPORT_FILE}."
    )

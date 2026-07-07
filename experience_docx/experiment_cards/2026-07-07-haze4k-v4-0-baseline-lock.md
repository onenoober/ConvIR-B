# Haze4K v4.0 Baseline Lock Route

Date: 2026-07-07

Status: PLANNED

## Scope

- Project: ConvIR-B Haze4K v4.
- Route identity: new route initialization and A0 baseline lock for `SFAD-ConvIR-B`.
- Fixed pain points:
  1. spatially non-uniform haze and polluted feature transfer;
  2. low-frequency dehazing versus high-frequency detail preservation conflict.
- Method family: SFAD, with later SDFM-GST and DCFSB-lite variants.
- Starting source: `github/codex/haze4k-official-arch-anchor`.
- Starting commit: `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-0-baseline-lock`.
- Evidence root: `experience_docx/experiment_logs/haze4k_v4_0_baseline_lock_20260707/`.
- Protocol package: `docs/ai_text_packages/haze4k_v4_sfad/`.

## Fact Sources

- GitHub `main` at `fd5ac2b3740b3a60f1084b0a7372f12b202bb9a9` for current route memory and closed-route status.
- GitHub anchor branch `codex/haze4k-official-arch-anchor` at `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898` for clean ConvIR-B architecture source.
- Current `convir-4090` cloud state for runtime paths, data, checkpoint, GPU availability, and raw outputs.
- User-supplied v4 SFAD text package as design input, now staged under `docs/ai_text_packages/haze4k_v4_sfad/`.

## Route Framing

This is a new v4 route, not a continuation, rescue, or expansion of v2/v3. It is allowed to create new architecture branches from the official anchor. It is not allowed to continue v3.0/v3.2 canary expansion, selector/alpha tuning, bridge/generator work, or locked-test-selected policy search.

## Locked-Test Policy

Locked Haze4K test is blocked for model selection. A0 will first use anchor evidence plus train-derived/internal validation contracts. Any future locked-test confirmation requires a fixed candidate, fixed checkpoint policy, and a written gate that has already passed on internal splits.

## Resource Contract

- Runtime host: `convir-4090`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Haze4K data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Checkpoint sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- A0 command script: `experience_docx/experiment_logs/haze4k_v4_0_baseline_lock_20260707/run_v4_a0_preflight.sh`.
- Status file: `experience_docx/experiment_logs/haze4k_v4_0_baseline_lock_20260707/status.txt`.

## Metric Contract

A0 is the fixed ConvIR-B official-anchor baseline. Later v4 deltas must use the same sample/crop/split view as A0. Single-seed deltas below `+0.10 dB` are directional/mechanism evidence, not promotion evidence. A1/A2/A3 must report PSNR/SSIM, preservation/tail evidence, module statistics, and failure samples before any later phase is authorized.

## Stage Ladder

1. A0: strict baseline/preflight lock, no model edits, no locked-test selection.
2. A1: SDFM only.
3. A2: GST only.
4. A3: SDFM + GST.
5. A4: optional weak density auxiliary loss only if A3 justifies it.
6. B1/B2: DCFSB bottleneck and bottleneck+1/2 decoder.
7. C1/C2/C3: final spatial/final-lite/final-full candidates.

## Stop Rules

- If A0 resource or metric contract is not reproducible, do not launch module experiments.
- If A3 is not better than A0 and R/G maps are not interpretable, do not continue to DCFSB.
- If DCFSB introduces high-frequency artifacts or sky noise, prefer bottleneck-only or drop it.
- Do not treat mean PSNR alone as sufficient evidence.

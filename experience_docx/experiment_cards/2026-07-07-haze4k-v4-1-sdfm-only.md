# Haze4K v4.1 SDFM-Only Route

Date: 2026-07-07

Status: COMPLETED_PREFLIGHT_PASS

## Scope

- Project: ConvIR-B Haze4K v4.
- Route identity: new architecture route, A1 SDFM-only.
- Fixed pain points:
  1. spatially non-uniform haze and polluted feature transfer;
  2. low-frequency dehazing versus high-frequency detail preservation conflict.
- This branch tests only pain point 1 through spatial degradation field modulation at the two ConvIR multi-scale fusion points.
- Starting source: `github/codex/haze4k-official-arch-anchor`.
- Starting commit: `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Branch: `codex/haze4k-v4-1-sdfm-only`.
- Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-1-sdfm-only`.
- Evidence root: `experience_docx/experiment_logs/haze4k_v4_1_sdfm_only_20260707/`.
- Shared protocol package: `docs/ai_text_packages/haze4k_v4_sfad/`.

## Fact Sources

- GitHub `main` at `fd5ac2b3740b3a60f1084b0a7372f12b202bb9a9` for current route memory and v2/v3 closure decisions.
- GitHub anchor branch `codex/haze4k-official-arch-anchor` at `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898` for clean ConvIR-B architecture source.
- A0 v4 preflight branch `codex/haze4k-v4-0-baseline-lock` at `7b09159891addd6b0deac98f0139d2884cbc3b27` for v4 baseline/resource contract.
- Current `convir-4090` runtime state for paths, GPU, data, checkpoint, and raw outputs.

## Hypothesis

A learned single-channel spatial degradation field at the 1/2 and 1/4 multi-scale fusion points can modulate ConvIR-B feature fusion without disturbing the official baseline at initialization. If useful, A1 should show non-collapsed `R_s` maps and target-region movement before any GST or DCFSB module is added.

## Architecture Contract

- New route arch: `--arch sfad_sdfm`.
- Official FAM modules remain present and keep their checkpoint key names.
- New modules: `SFAD_SDFM2` at 1/2 scale and `SFAD_SDFM1` at 1/4 scale.
- Allowed missing prefixes during partial load: `SFAD_` only.
- Unexpected checkpoint keys: fatal.
- Official key shape mismatch: fatal.
- Initialization: FiLM residual scale `alpha=0`, so the route must be no-op versus A0 at initialization.
- No GST, no DCFSB, no density auxiliary loss, no color/airlight branch.

## Training Contract

Stage 0 preflight must prove strict partial-load, finite forward, and no-op equivalence to A0. Stage 1 uses `--sfad_train_scope adapter_only`, training only `SFAD_` parameters first. Any wider scope requires a card update and Stage 1 evidence.

## Resource Contract

- Runtime host: `convir-4090`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Haze4K data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Checkpoint sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Command script: `experience_docx/experiment_logs/haze4k_v4_1_sdfm_only_20260707/run_v4_a1_sdfm_preflight.sh`.
- Status file: `experience_docx/experiment_logs/haze4k_v4_1_sdfm_only_20260707/status.txt`.

## Metric Contract

A1 deltas must be compared against A0 on the same sample/crop/split view. Stage 0 uses train-only finite/no-op checks and does not make a quality claim. Stage 1/2 quality gates require PSNR/SSIM, per-image deltas, SDFM `R_s` statistics, and preservation/tail evidence before moving to A2/A3.

## Locked-Test Policy

Locked Haze4K test is blocked. Do not use locked test to choose checkpoint, scale, seed, active modules, or thresholds.

## Stop Rules

- Stop if partial load has unexpected keys, official shape mismatch, or bad missing non-`SFAD_` keys.
- Stop if no-op max absolute difference versus A0 is not near zero at initialization.
- Stop if `R_s` collapses to all 0 or all 1 in the preflight/audit.
- Do not continue to GST/DCFSB if A1 cannot pass preflight.

## Stage 0 Preflight Result

Status: `COMPLETED_PREFLIGHT_PASS`.

Primary evidence file: `experience_docx/experiment_logs/haze4k_v4_1_sdfm_only_20260707/v4_a1_sdfm_preflight.json`.

Key checks:

- Code commit: `73643c4a965e6399ebfb5362c8fff668c4d8e518`.
- Total params: `8,831,629`; added params: `200,964`.
- Adapter-only trainable params: `200,964`; frozen official params: `8,630,665`.
- Trainable prefixes: `SFAD_SDFM1`, `SFAD_SDFM2`.
- Partial load: `602` official keys loaded; `22` missing new-module keys; unexpected `[]`; shape mismatch `[]`.
- No-op max abs vs A0: synthetic `0.0`, train crop `0.0`.
- One train-crop multiscale L1: `0.01309124380350113` on the sampled crop.
- `R_s` preflight stats:
  - `SDFM_1_2`: mean `0.5245807`, std `0.0071737`, min/max `0.4805435/0.5622713`, alpha `0.0`.
  - `SDFM_1_4`: mean `0.5315088`, std `0.0017463`, min/max `0.5227934/0.5384336`, alpha `0.0`.
- Locked test touched: `false`; test split enumerated: `false`.

Decision: A1 Stage 0 passes. Stage 1 adapter-only training up to 5 epochs is authorized with the same data/checkpoint contract and locked test blocked.

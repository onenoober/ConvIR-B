# ConvIR-Dehaze-v1.4-UDP-Lite

Date: 2026-06-04

Status: v1.4A adapter-only completed; internal regular+hard gate failed; locked Haze4K test blocked.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: depth/prior inside ConvIR feature learning, replacing the stopped DPGA-v1.3 HSDF bottleneck form.
- Dataset or task: Haze4K train-derived internal splits first (`train_inner`, `val_regular`, `val_hard`).
- Primary objective: improve hard-bottom recovery with real multi-scale depth-guided feature fusion while preserving regular/easy/strong-reference cases.
- Main metric: PSNR delta versus official ConvIR-B A0 on internal validation.
- Secondary metrics: SSIM delta, positive ratio, strong-reference regression ratio, worst `<= -0.20 dB` count, latency/memory, fusion activity, depth-quality failure signatures.
- Execution environment: cloud server `dehaze1`; local WSL checkout is for editing and compile/syntax-only checks.
- Artifact root: `experience_docx/experiment_logs/haze4k_udp_lite_v14_20260604/`.
- Branch or isolated workspace: `codex/haze4k-convir-v1-4-udp-lite-depth-fusion`.

## Current Run Status

- v1.4A adapter-only run launched on `dehaze1` at `2026-06-04T22:02:25+08:00` and completed at epoch `20`, iter `300/300`.
- Tmux sessions `v14a_udp_lite_train` and `v14a_udp_lite_post` are no longer active.
- Checkpoints produced on cloud: `Best.pkl`, `Final.pkl`, `model.pkl`, and `model_20.pkl`; `Final.pkl` mtime `2026-06-04 22:32:53 +08:00`.
- Training log: `experience_docx/experiment_logs/haze4k_udp_lite_v14_20260604/train_ConvIR-Haze4K-v1.4A-UDP-Lite-DPFM123-adapter-only-seed3407-20260604.log`; latest logged validation PSNR was `39.50 dB`.
- Internal Best/Final `val_regular` and `val_hard` eval completed in `v14a_eval_regular_hard/`; gate file is `v14a_gate_eval_regular_and_hard.json`.
- Intermediate module ablation and depth-quality audits completed for both `val_regular` and `val_hard` under `v14a_intermediates/`.
- UDPNet external repo audit found `ConvIR_UDPNet.py`, README Haze4K ConvIR-B baseline `34.15`, ConvIR+UDP `34.82`, and one checkpoint-like URL; official eval reproduction is still pending.

## Baseline Contract

- Baseline implementation: official ConvIR-B Haze4K A0 checkpoint and unchanged `Dehazing/ITS/models/ConvIR.py` path.
- Candidate initialization: load A0 into UDP-Lite DPGA wrapper with only `DPGA_*` missing keys.
- Evaluation entrypoint: `experience_docx/tools/eval_haze4k_checkpoint_compare.py` with `--candidate_arch dpga --candidate_dpga_fusion_mode udp_lite`.
- Training entrypoint: `Dehazing/ITS/main.py --arch dpga --dpga_fusion_mode udp_lite --mode train`.
- Dataset and split: v1.3 train-derived internal split JSON reused unless a new split card is written; no locked Haze4K test for v1.4A/B pre-gate.
- Preprocessing and decoding: existing Haze4K loader plus cached DepthAnything V2 depth prior.
- Metric implementation: same PSNR/SSIM implementation as v1.3 compare tooling.
- Reference entrypoints that must remain stable: plain `--arch convir` remains unchanged; legacy DPGA remains `--dpga_fusion_mode legacy`.
- Checkpoint/export/resume contract: candidate checkpoints save a standard `model` state dict; A0 load must be strict except for `DPGA_*` keys.

## Most Valuable Attempt

- v1.3 evidence shows hard-gate detection and bottleneck activation are not the bottleneck; corrected ablation found bottleneck-only mean contribution about `+0.0008 dB`.
- The next high-value variable is the fusion mechanism: replace shallow/bottleneck/skip small residual adapters with A0-equivalent multi-scale DPFM/DGAM-Lite fusion.
- The cheapest decisive preflight is zero-init A0 equivalence plus projection-gradient liveness before any training.
- Failure of zero-init or grad liveness blocks training and is an engineering failure, not a route result.
- Failure of v1.4A quality/hard gates after valid preflights rules out adapter-only UDP-Lite at the current capacity; it does not authorize scale search.

## Hypothesis

If cached depth/prior features are encoded into a pyramid and fused into ConvIR encoder features at 1x, 1/2x, and 1/4x using zero-init depth-guided channel attention plus local window cross-attention, then hard-bottom PSNR should improve more than DPGA-v1.3 because depth is now a multi-scale feature-conditioning signal instead of a small single-path residual adapter.

## Change

- Code branch: `codex/haze4k-convir-v1-4-udp-lite-depth-fusion`.
- Exact code/config change:
  - `--dpga_fusion_mode udp_lite` selects the new path;
  - `DPGA_prior_encoder` creates prior pyramid `D1/D2/D4`;
  - `DPGA_dpfm1`, `DPGA_dpfm2`, `DPGA_dpfm4` fuse after `Encoder[0]`, `Encoder[1]`, `Encoder[2]`;
  - each DPFM has zero-init depth-guided channel attention and zero-init local window cross-attention;
  - `DPGA_agf1`, `DPGA_agf2` provide optional zero-init AGF-lite skip gating but are inactive by default.
- Enabled mechanisms for v1.4A: `DPGA_prior_encoder`, `DPGA_dpfm1`, `DPGA_dpfm2`, `DPGA_dpfm4`.
- Explicitly disabled mechanisms for v1.4A: legacy shallow/bottleneck/skip DPGA, hard gate supervision, DPGA scale-only search, APDR/output residual, full ConvIR finetune, locked Haze4K test.
- Parameter/runtime/memory impact expected: higher than v1.3 DPGA due window attention; must be recorded by internal eval before promotion.
- Initialization or no-op behavior: all DPFM/AGF output projections are zero-init; initial outputs must match A0 within `<= 1e-6` max absolute difference.
- Resume policy: resume only same config/split; any changed active modules/components are a new diagnostic.
- Defaults changed: none for legacy DPGA or plain ConvIR.
- Defaults intentionally preserved: `--arch convir` and `--dpga_fusion_mode legacy` remain backward-compatible.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| Python compile | edited model/train/data/tools compile locally | pass |
| dataloader engineering fix | hard-aware sampler uses `batch_sampler`; dataset stores `image_dir` | pass; both typos patched and compiled |
| zero-init equivalence | `max_abs_diff(output, A0) <= 1e-6` on cloud | pass; `max_abs_diff=0.0` in `preflight/v14_zero_init_equivalence.json` |
| projection-gradient liveness | at least one active UDP-Lite zero-init projection has nonzero grad on cloud | pass; DPFM1/2/4 channel and cross projections all nonzero in `preflight/v14_zero_init_equivalence.json` |
| UDPNet repo text audit | external repo exposes ConvIR UDP code and checkpoint/protocol clues | pass; `v14_udpnet_repro_audit.md` |
| split/intermediate audit | v1.4 split/depth/mask metadata available before training | pass; post-checkpoint module/depth audits completed for `val_regular` and `val_hard` |
| no locked test | all pre-gate commands use train-derived internal splits | pass; locked Haze4K test was not run |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| zero-init A0 equivalence | proves route preserves anchor before training | one synthetic or internal sample | `v14_zero_init_equivalence.json` |
| DPFM module ablation | identifies which scale supplies useful depth fusion | `val_regular`, `val_hard` | `v14_depth_fusion_module_ablation_val.csv` |
| depth-quality failure audit | separates depth-estimator failures from fusion-capacity failures | `val_hard` plus strong/worst regressions | `v14_depth_quality_failure_audit.csv` |
| fusion delta norm | checks branch activity and prevents uncontrolled correction | training logs | train log `DPGA_TC ... fusion_delta` |
| regular+hard gate | prevents easy-safe configs from being selected as hard gains | Best/Final on both splits | v1.4 gate JSON |

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| A0 with `--arch convir` | reference baseline | unchanged from prior accepted baseline |
| UDP-Lite zero-init | no-op anchor preservation | `max_abs_diff <= 1e-6` |
| DGCA-only vs full DPFM | distinguish channel conditioning from local cross-attention | both logged before route interpretation |
| DPFM1/2/4 only | scale attribution | all three single-scale rows written |
| DPFM+AGF-lite row | tail-control reserve diagnostic only | not used for v1.4A selection unless predeclared v1.4C |

## Fair Run Contract

- Training budget: v1.4A adapter-only internal run first; no locked test.
- Batch/sample policy: hard/medium/easy balanced sampler from v1.3 unless a new split audit supersedes it.
- Optimizer: Adam; v1.4A trains `DPGA_*` only; v1.4B may use `--dpga_train_scope fusion_neighbor` with smaller neighbor LR.
- Loss weights: Charbonnier reconstruction; FFT `0.03~0.05`; anchor/chroma preservation only as tail guard; fusion delta norm around `1e-4`.
- Random seed policy: seed `3407` for first v1.4A.
- Evaluation cadence: Best/Final evaluated on `val_regular` and `val_hard`.
- Hardware/runtime assumptions: cloud `dehaze1`; local WSL compile/static checks only.
- Locked evaluation policy: locked Haze4K test only once after v1.4 internal regular+hard gates pass.
- Exception budget: if v1.4A is mean-positive/tail-safe but hard `< +0.04 dB`, do not scale-search; move to v1.4B partial unfreeze.


## v1.4A Results

### Gate Summary

- Gate result: `pass=false`; `locked_test_allowed=false` in `v14a_eval_regular_hard/v14a_gate_eval_regular_and_hard.json`.
- Best `val_regular`: mean delta `+0.028294 dB`, easy top-25 `+0.019686 dB`, SSIM `+0.000001232`, positive ratio `0.586667`, strong regression ratio `0.253333`, worst `<= -0.20 dB` count `19`.
- Best `val_hard`: mean delta `+0.020340 dB`, hard bottom-25 `+0.022275 dB`, easy top-25 `+0.036567 dB`, SSIM `-0.000000505`, positive ratio `0.583333`, worst count `14`.
- Final checkpoint was weaker: `val_regular` mean `+0.026026 dB`, `val_hard` mean `+0.016454 dB`, `val_regular` worst count `22`.

### Module Attribution

- `DPFM1-only` was the strongest safer contributor: `val_regular` mean `+0.030553 dB`, hard25 `+0.022737 dB`, worst count `2`; `val_hard` mean `+0.026774 dB`, hard25 `+0.024597 dB`, worst count `0`.
- Full `DPFM1+2+4` was not better than `DPFM1-only` and raised tail risk: `val_regular` worst count `19`, `val_hard` worst count `14`.
- `DPFM2-only` was negative on both splits: `val_regular` mean `-0.005455 dB`, `val_hard` mean `-0.011348 dB`.
- `DPFM4-only` was small positive but insufficient; `AGF-lite` row matched full `DPFM1+2+4`, so it had no active effect in v1.4A.

### Interpretation

v1.4A confirms that UDP-Lite depth fusion has a real but weak positive signal,
with the useful action concentrated at the 1x DPFM1 scale. The full multi-scale
adapter-only composition is not promotion-ready because it misses hard-gain
thresholds and introduces unacceptable tail risk. This rules out full DPFM123
scale/gate micro-tuning, but it leaves a DPFM1-focused diagnostic or v1.4B
fusion-neighbor partial unfreeze as evidence-supported follow-ups.

## Gates

### v1.4A Pass Line

- `val_regular` mean delta `>= +0.040 dB`.
- `val_regular` easy top-25 delta `>= 0`.
- `val_regular` SSIM delta `>= 0`.
- `val_regular` strong regression ratio `<= 0.16`.
- `val_regular` worst `<= -0.20 dB` count `<= 12/300`.
- `val_regular` positive ratio `>= 0.62`.
- `val_hard` mean delta `>= +0.030 dB`.
- `val_hard` hard bottom-25 delta `>= +0.050 dB`.

### Stop/Continue

- If zero-init equivalence fails: stop and fix engineering before training.
- If v1.4A mean `<= +0.02 dB` and hard `<= +0.02 dB`: adapter-only UDP-Lite is insufficient; do not micro-tune v1.4A.
- If v1.4A is mean-positive and tail-safe but hard remains below gate: proceed only to v1.4B fusion-neighbor partial unfreeze.
- If v1.4A/B mean and hard pass but tail is high: add v1.4C AGF-lite skip gating as a separate predeclared variable.
- Only after internal gates pass: allow one locked Haze4K test.

## Analysis Plan

- Run UDPNet ConvIR repo audit separately to verify external code/checkpoint availability and Haze4K protocol compatibility.
- Run zero-init and grad liveness before training.
- Train v1.4A on `train_inner` only after preflight passes.
- Best/Final on `val_regular` and `val_hard` completed.
- Module ablation and depth-quality failure audits completed for both internal splits.
- This card, `EXPERIMENT_INDEX.md`, and the family summary record the failed v1.4A gate and module attribution.

## Decision

- Decision label: `FAIL_V14A_ADAPTER_ONLY_FULL_DPFM123`.
- Image/global metric reason: Best `val_regular` mean `+0.028294 dB` and Best `val_hard` mean `+0.020340 dB` are positive but below gate thresholds; hard bottom-25 is only `+0.022275 dB`.
- Mechanism reason: module attribution shows useful action is concentrated in `DPFM1-only`; `DPFM2-only` is negative and full `DPFM1+2+4` raises tail risk.
- Preservation or regression reason: Best full DPFM123 has `val_regular` worst count `19` and strong regression ratio `0.253333`, so locked test remains blocked.
- Cost/deployability reason: adapter-only UDP-Lite adds runtime complexity without passing internal gates.
- Evidence strength label: completed internal diagnostic with regular/hard eval, module ablation, and depth-quality audit.
- Reopen condition, if stopped: a DPFM1-focused diagnostic, v1.4B fusion-neighbor partial unfreeze, or external UDPNet reproduction result must explain how it avoids DPFM2/tail failure.
- What this decides next: do not run locked Haze4K test or full DPFM123 scale/gate micro-tuning; consider only evidence-supported DPFM1-focused or fusion-neighbor follow-up.

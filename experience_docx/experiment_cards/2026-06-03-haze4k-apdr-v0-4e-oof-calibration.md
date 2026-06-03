# Haze4K APDR-v0.4E 5-Fold OOF Calibration

Date: 2026-06-03

Status: E1 OOF audit completed on `autodl-dehaze4`, but a post-sync
implementation audit found a clean-reproducibility mismatch in commit
`ed38afb`. Current locked thresholds remain blocked; exact v0.4E numbers are
fixed-code rerun pending. No stop20, local correction, full spatial router, or
trainable residual head.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-v0.4E frozen action-bank OOF diagnostics.
- Dataset or task: Haze4K train split, 5 stratified OOF folds.
- Primary objective: test whether the locked E0 confidence/no-op rules and
  target-free risk features remain safe when each held-out fold is evaluated by
  basis and mappers derived only from the other folds.
- Main metric: OOF severe/strong regressions, easy preservation, hard gain,
  mean gain, accepted coverage, oracle recovery, and risk-feature calibration.
- Execution environment: `autodl-dehaze4`,
  `/root/miniconda3/envs/convir-cu128/bin/python`.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_apdr_v0_4e_oof_calibration_20260603/`.
- Branch or isolated workspace:
  `codex/haze4k-apdr-v0-4b-mapping-triage`.

## Baseline Contract

- Baseline implementation: frozen ConvIR-B official Haze4K checkpoint through
  the APDR-v0.2RC selector checkpoint.
- Training entrypoint: none.
- Evaluation entrypoint:
  `experience_docx/tools/audit_haze4k_apdr_v0_4e_oof_calibration.py`.
- Dataset and split: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`;
  5 stratified folds by anchor PSNR, `P_benefit`, `M_safe_mean`, and file-orde
  chunk.
- Reference assets that must remain stable: APDR-v0.2RC selector, sigma `3.0`
  correctability, v0.4D frozen spatial feature hooks, and v0.4E locked rules.
- Checkpoint/export/resume contract: no checkpoint; rerun shell script.

## Most Valuable Attempt

- Why this is the highest-value next attempt: E0 passed on an independent
  confirm slice, but a single confirm slice cannot prove risk calibration.
- Target failure or opportunity: OOF selective reliability of candidate actions.
- Cheap preflight evidence: E0 locked thresholds passed with Rule A strong/severe
  `0/0` and Rule B `1/0`.
- Earliest decisive gate: E1 OOF locked-rule calibration.
- Expected cost or attempt-count saving: avoids any long training or stop20 run
  if OOF tail safety fails.
- What success decides: authorize a locked-policy held-out E2 audit.
- What failure decides: close current v0.4E-RSAB thresholds or redesign risk
  features before any deployable module.
- Why a cheaper diagnostic is not enough: E0 is confirm-slice evidence, not
  per-fold out-of-fold calibration.

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| E1 OOF pass | severe `0`, strong rate `<=1%`, easy `>=-0.02`, hard `>=+0.25`, mean `>0`, coverage `>=10%`, oracle recovery `>=0.15` | locked target-free confidence transfers across OOF folds | authorize E2 locked held-out policy audit |
| E1 OOF fail | any pass-line item fails | selector confidence is not yet reliable enough | stop or redesign risk features; do not train full router |
| Intermediate evidence complete | all OOF JSON/CSV outputs exist | later decisions can inspect fold stability and failure signatures | continue only after reading E1 |

## Current Decision

Decision label:

```text
E1_FAIL_STOP_CURRENT_LOCKED_THRESHOLDS
```

Reproducibility status:

```text
FIXED_CODE_RERUN_REQUIRED_BEFORE_NUMERIC_SEAL
```

Post-sync implementation audit found `align_coners` and
`kenel_size/kernel_size` mismatches in the submitted v0.4E tools. The failure
direction remains a useful stop signal, but the exact E1 numeric evidence must
be reproduced from a clean fixed-code checkout before archival sealing.

E1 completed on `autodl-dehaze4` with `exit_code=0`
(`2026-06-03T21:50:26+08:00` to `2026-06-03T22:39:02+08:00`).

Locked Rule A/B OOF result:

| Rule | Keep | Coverage | Mean gain | Hard gain | Easy gain | Strong/severe | Oracle recovery | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Rule A: `global_plus_spatial_kenel_knn_9`, K16 | `239/3000` | `0.0797` | `+0.0749 dB` | `+0.2596 dB` | `+0.0000 dB` | `0/5` | `0.1654` | fail: severe tail and coverage |
| Rule B: `spatial_priors_ridge_10`, K16 | `150/3000` | `0.0500` | `+0.0378 dB` | `+0.1352 dB` | `+0.0000 dB` | `0/1` | `0.0835` | fail: severe, hard, recovery, coverage |

Post-hoc OOF threshold policy search retained `4000` low-capacity candidates
and found `0` gate-passing policies. The best retained policy
(`spatial_priors_ridge_10`, scale `1.0`,
`weighted_residual_norm >= 0.009209617488433332` and
`nn_distance <= 11.120915794372559`) removed severe regressions and reached hard
`+0.2527 dB`, but coverage was only `0.0877`, below the pre-registered
`0.10` line.

Do not launch E2, full spatial router, local correction, dense residual
training, or stop20 from the current locked-threshold v0.4E route. A future
route may be justified only if it pre-registers a safe-subset policy and tests
it on a fresh held-out split.

# Haze4K v5 CHD-RM v3n Conservative First-Step Calibration

Date: 2026-07-12

Status: `PLANNED_A0_LABEL_ONLY_PREFLIGHT`

Branch: `codex/haze4k-v5-v3n-conservative-first-step-calibration`

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3n_conservative_first_step_calibration_20260712/`

## Route Identity

v3n is a new target-semantics diagnostic route after v3m, not a rescue of the
failed v3m A3 calibrated policy. Parent evidence is GitHub `main` commit
`260dccc97c308cf8d0a3ce856086bca928a5a46a`. Runnable source starts from v3m
route commit `2c9cb511627895981c4c489cacd990326185ced6`.

v3m established that block16 oracle value and direct-step-energy label
observability are real, but mean/ordinal aggressive action calibration creates
unsafe image-level tails. v3n therefore tests only a stricter false-intervention
semantics before any policy replay: default `alpha=0.125`, allow only a single
first-step escalation to `alpha=0.25`, and set the threshold as a fixed
train-fold negative quantile.

## Forbidden

- no Haze4K locked-test access;
- no canary or route-confirm audit;
- no model, controller, ranker, backbone, direct-head, or adapter training;
- no policy replay in A0;
- no threshold-family search or post-hoc threshold tuning;
- no physics/proxy continuation;
- no checkpoint, image, raw per-image, raw block, or selected-action table sync
  to GitHub.

## A0 Objective

Run a label-only preflight over the v3m-A1 cloud-only block table. For each
operator and held-out fold:

1. define negative blocks as `oracle_alpha <= 0.125`;
2. compute the 99th percentile of `direct_step_energy` over train-fold
   negatives;
3. select held-out blocks above that threshold for `alpha=0.25`;
4. keep all other blocks at `alpha=0.125`.

A0 does not replay images and cannot claim PSNR utility. It can only decide
whether this conservative target semantics is label-feasible enough to justify
a separate A1 32-image replay smoke.

## A0 Gate

Both `D_ref` and `D_rep` must satisfy:

- held-out negative false rate `<= 0.0125`;
- max per-fold negative false rate `<= 0.02`;
- selected coverage `>= 0.005`;
- min per-fold selected coverage `>= 0.0025`;
- selected precision `>= 0.60`;
- positive recall `>= 0.01`.

Pass decision:
`V3N_A0_CONSERVATIVE_LABEL_PREFLIGHT_PASS_AUTHORIZE_A1_REPLAY_SMOKE_ONLY`.

Fail decision:
`V3N_A0_CONSERVATIVE_LABEL_PREFLIGHT_FAIL_STOP_NO_REPLAY`.

No A0 result can authorize formal replay, route-confirm, canary, locked test,
training, learned ranker, physics/proxy continuation, or deployment.

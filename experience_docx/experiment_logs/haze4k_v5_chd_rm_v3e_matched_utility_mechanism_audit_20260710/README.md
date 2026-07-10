# v3e D7c-FAM2 Matched Utility Mechanism Audit

Date: 2026-07-10

Route id:
`haze4k_v5_chd_rm_v3e_matched_utility_mechanism_audit_20260710`

Route branch:
`codex/haze4k-v5-v3e-matched-utility-mechanism-audit`

Parent/source:
`origin/codex/haze4k-v5-v3d-rarm-adapter-only-preflight`
at `c52f65fcfe413e47bbf2324f02c968ed19c50980`.

## Purpose

Audit why v3d D7c-gated FAM2 adapter-only training is tail-safer than the
ungated matched control but does not beat the control on mean PSNR.

This is a mechanism audit, not a continuation of v3d training.

## Fact Sources

- GitHub `main`:
  `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`
- GitHub `main` v3d compact evidence:
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710/`
- Cloud runtime/raw source:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/`
- v3e cloud workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3e-matched-utility-mechanism-audit/`

## Forbidden Flows

- No 20-epoch continuation.
- No v4/RARM expansion.
- No neighbor, FAM1, or backbone unfreeze.
- No new generic D7c probe.
- No locked Haze4K test.
- No NH-HAZE or Dense-Haze transfer.
- No optimizer, scheduler, clipping, or loss change mixed into this audit.
- No checkpoint, image, tensor, array, or raw feature-table commit.

## Metric Contract

All v3e phases use the same internal Haze4K train-derived val-inner 600 split
used by v3d:

`experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json`

Baseline is A0 official Haze4K checkpoint:

`/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`

Metric direction:

- higher PSNR/SSIM delta is better;
- fewer `<= -0.2 dB` regressions is safer;
- lower negative/exterior output change is safer.

## Completed Phases

- v3e-A: paired statistical reanalysis from existing v3d CSV only.
- v3e-B: no-training `W_D/W_U x G_D/G_1` replay.
- v3e-C: operator-gain alignment and boundary leakage audit.
- v3e-D: no-step gradient/optimizer/scheduler contract audit.

No phase authorizes training or locked-test access.

## Key Results

v3e-A:

- paired mean CI95: `[-0.01676, -0.00365, +0.00930]`;
- tail-regression reduction CI95: `[26, 41, 57]`;
- decision: `V3E_A_SINGLE_SEED_MEAN_INCONCLUSIVE_TAIL_SAFETY_STABLE`.

v3e-B:

- `W_D + G_D`: mean `+0.02947`, `<= -0.2 dB` regressions `50`;
- `W_D + G_1`: mean `+0.03899`, `<= -0.2 dB` regressions `113`;
- `W_U + G_D`: mean `+0.01278`, `<= -0.2 dB` regressions `23`;
- `W_U + G_1`: mean `+0.03307`, `<= -0.2 dB` regressions `91`.

v3e-C:

- D7c score vs ungated FAM2 positive gain AUROC: `0.4921`;
- D7c score vs D7c-gated FAM2 positive gain AUROC: `0.4904`.

v3e-D:

- all audited batches clipped for both D7c and control;
- effective Adam weight decay was `0` despite CLI `0.0001`;
- resume checkpoints have no scheduler state.

## Final Decision

`V3E_OPERATOR_CORRECTABILITY_MISMATCH_PRIMARY_HARD_GATE_SAFETY_TRADEOFF_SECONDARY_NO_RARM_EXPANSION`

The next route should not continue v3d. It should design/audit a D7c safety
veto plus FAM2 operator-correctability ranker using internal/OOF actual FAM2
marginal gain targets.

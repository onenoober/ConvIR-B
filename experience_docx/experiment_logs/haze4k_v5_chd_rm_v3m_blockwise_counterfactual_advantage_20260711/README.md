# v3m Blockwise Counterfactual Advantage

Status: `A3_FAIL_STOP_NO_ROUTE_CONFIRM`.

Route card:
`experience_docx/experiment_cards/2026-07-11-haze4k-v5-chd-rm-v3m-blockwise-counterfactual-advantage.md`

Central index:
`experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

This route starts from the v3l frozen direct operators and rechecks control
granularity under a common action ladder before any local actuation, physics,
proxy, or controller work. It is train-derived only and keeps locked test and
canary blocked.

## A0a Outputs

- `v3m_a0_common_action_summary.json`
- `v3m_a0_common_action_gate.csv`
- `v3m_a0_granularity_retention_group_bootstrap.csv`
- `v3m_a0_common_action_oracle_summary.csv`
- `v3m_a0_confirm_audit_only_policy_summary.csv`
- `v3m_a0_operator_agreement.csv`
- `v3m_a0_source_manifest.json`

Cloud-only per-image outputs live under `cloud_only_raw_common_action/` and
must not be committed or synced to GitHub.

`v3m_deviation_log.md` records the initial summary-only engineering failure and
the constrained rebuild command.

## A0a Closeout

`v3m_a0a_closeout.md` is the canonical compact A0a result and integrity
record. It records the dual-operator gate values, the raw-table preservation
checks, and the repaired finite operator-agreement statistics.

Decision:
`V3M_A0_COMMON_ACTION_GRANULARITY_PASS_AUTHORIZE_A0B_DENSE_AND_CONTINUOUS_MECHANISM_ONLY`.

Only the A0b dense-grid and continuous-pixel mechanism cross-audit is
authorized. `v3m_a0b_metric_contract.md` freezes its paired OOF source hashes,
policy mappings, and quantization-gap gate before it reads any paired outcome.
No model/inference rerun is needed. It cannot promote a policy, controller,
training, canary, or locked-test access; only its own written pass closeout may
authorize the A1 feasible-local-actuation audit.

The initial A0b output is retained as a metric-contract-mismatched diagnostic:
continuous alpha is solved before final output clamping, so a pointwise nested
candidate check is invalid for that pair. `v3m_a0b_metric_contract.md` records
the correction and A0b-r1 remains read-only.

## A0b-r1 Closeout

`v3m_a0b_r1_closeout.md` records the valid corrected result. All eight paired
quantization-gap checks passed with exact fixed-alpha replay, input SHA checks,
and existing tail-safety preservation. Decision:
`V3M_A0B_QUANTIZATION_GAP_SMALL_AUTHORIZE_A1_FEASIBLE_LOCAL_ACTUATION_ONLY`.

Only A1 feasible-local-actuation audit is now authorized. It needs a separate
metric contract and must not train a controller, select route-confirm choices,
launch a canary, or access locked test.

## A1 Preflight

`v3m_a1_metric_contract.md` freezes a no-training block16 local-observability
audit. It writes its large block table only on cloud and tests four fixed
signals with grouped per-image OOF AUC. `run_v3m_a1_smoke32.sh` is the required
32-image replay template; the first attempt is retained as an engineering
failure, r1 preserved the wrong subset fold map, and
`run_v3m_a1_smoke32_r2.sh` is the corrected screen. Only its
explicit pass marker authorizes
`run_v3m_a1_formal.sh` on all 1,200 OOF images.

## A1 Smoke Result

The corrected r2 smoke reproduced both frozen operators' 32 fixed-alpha rows
exactly and wrote the expected 40,000 cloud-only block records. Its compact
summary, replay CSV, signal CSV, and source manifest are retained under
`a1_smoke32_r2/`; the raw block table is cloud-only. Decision:
`V3M_A1_SMOKE_REPLAY_PASS_AUTHORIZE_FORMAL_OOF_ONLY`.

## A1 Formal Closeout

`v3m_a1_formal_closeout.md` records exact dual-operator replay and the formal
signal gate. Direct-step energy and D7c score passed; the next and only
authorized stage is A2 OOF calibration audit. No training or test access is
authorized.

## A2 Closeout

`v3m_a2_closeout.md` records the fold-separated OOF label-calibration result.
Using only A1's fixed primary signal, `direct_step_energy`, the 16-target-bin
monotone calibration rule collapsed deterministically to 9 actual bins because
of duplicate calibration-fold score quantiles. It still passed by a wide
margin on labels:

- `D_ref` ordinal MAE improvement CI95 low `0.6734292`;
- `D_rep` ordinal MAE improvement CI95 low `0.6719839`;
- escalation AUROC CI95 lows `0.8518716` and `0.8514065`;
- AP-lift CI95 lows `0.3123022` and `0.3117141`.

Decision:
`V3M_A2_OOF_CALIBRATION_PASS_AUTHORIZE_A3_FROZEN_POLICY_REPLAY_ONLY`.

A2 is label-only evidence and does not claim actual PSNR utility.

## A3 Closeout

`v3m_a3_closeout.md` records the actual frozen policy replay. A3 smoke r0
failed engineering before image replay because the wrapper omitted
`confirm_key`; r1 fixed the wrapper and passed fixed-alpha replay. A3 formal
then completed all 1,200 train-derived OOF images for both operators with exact
fixed `alpha=0.125` replay.

The calibrated policy had positive mean PSNR lift over fixed alpha but failed
retention and tail safety:

| Operator | Mean lift vs fixed | Lift CI95 low | Retention CI95 low | Paired lift p10 | Severe policy/fixed | Hard policy/fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `D_ref` | `+0.0828431 dB` | `+0.0659919 dB` | `0.1911805` | `-0.2209025 dB` | `148 / 0` | `39 / 0` |
| `D_rep` | `+0.0826054 dB` | `+0.0662187 dB` | `0.1922018` | `-0.2306944 dB` | `146 / 0` | `41 / 0` |

Decision:
`V3M_A3_FROZEN_POLICY_REPLAY_FAIL_STOP_NO_ROUTE_CONFIRM`.

No route-confirm audit, canary, locked test, training, learned controller,
ranker, physics/proxy continuation, or policy deployment is authorized.

## A3 Failure-Decomposition Diagnostic

`v3m_a3_failure_decomposition_closeout.md` records a corrected r1
diagnostic-only post-fail audit over the already completed A3/A2 cloud rows.
It does not train, tune thresholds, rerun inference, replay a new policy, use
route-confirm, touch canary, or touch locked test. The first r0 diagnostic used
the wrong tail definition (`policy_lift_vs_fixed` instead of actual
`policy_psnr_delta`) and is recorded only as a metric-mismatched operational
deviation; r1 matches the A3 closeout severe/hard counts.

Key r1 facts:

- cross-operator tail failures are highly stable: severe overlap `140` of union
  `154` (Jaccard `0.9090909`), hard overlap `38` of union `42` (Jaccard
  `0.9047619`);
- policy-lift and selected-alpha correlations across the two frozen operators
  are `0.9930474` and `0.9962972`;
- severe images still retain positive block16-oracle headroom on average
  (`+0.2754318 dB` for `D_ref`, `+0.2682955 dB` for `D_rep`);
- A2 calibration action confusion shows the current direct-step-energy bins are
  unsafe for aggressive actions: selected `alpha=0.25`/`0.5` match the held-out
  oracle action only about `6%`/`9%`, with roughly balanced over- and
  under-escalation; selected `alpha=1.0` still over-escalates about `42%`.

Decision:
`V3M_A3_FAILURE_DECOMPOSITION_DIAGNOSTIC_ONLY_NO_AUTHORIZATION`.

The diagnostic sharpens the bottleneck to safe utility calibration / action
semantics under aggressive local escalation. It does not authorize
route-confirm, canary, locked-test access, controller training, learned ranker,
physics/proxy continuation, or policy deployment.

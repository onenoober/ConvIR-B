# Haze4K DTA-v3.6 HRCS Route

Date: 2026-06-13

Status: `PHASE_A_COMPLETED_RELAXED_PASS_STRICT_FAIL_FORMAL_QUEUE_PENDING`

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Family: Depth-transmission adapters.
- Route name: DTA-v3.6 HRCS, High-Coverage Risk-Calibrated Selector for Conservative FDF.
- Branch: `codex/haze4k-dta-v3-6-hrcs`.
- Anchor commit: `2d529d4` from `github/codex/haze4k-official-arch-anchor`.
- Continuation base: DTA-v3.5 FDF-RCS-Lite commit `1e1d87a`; this is a selector/calibration continuation because v3.5 already fixed much of the v3.4 over-action failure and exposed oracle selector headroom.
- Runtime host: `convir-4090` only.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-6-hrcs`.
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`.
- v3.5 source evidence: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-5-fdf-rcs-lite/experience_docx/experiment_logs/haze4k_dta_v3_5_fdf_rcs_lite_20260612/`.
- Evidence root: `experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/`.

## Locked-Test Policy

Default governance rule remains: locked Haze4K test is blocked until the fixed train-derived policy is selected without test feedback.

The 2026-06-13 user instruction explicitly requests a relaxed continuation through later locked test. This card records that request as `USER_EXPLICIT_RELAXED_LOCKED_TEST_OVERRIDE_PENDING`. Any locked-test run from this route must be one-shot, fixed from train-derived evidence, and labeled exploratory/relaxed rather than strict promotion. The locked-test result must not be used to tune features, thresholds, checkpoints, or action-bank membership.

## Why This Route Exists

DTA-v3.5 FDF-RCS-Lite showed that conservative feature-level depth fusion has useful action: L1/L3 passed the positive-ratio line and kept strong true-vs-zero/shuffle/normal surplus, while L2 reduced worst regressions to `60.5/600`. It still failed strict all-image tail gates, and the nested v3.5 selector was only a single-feature threshold that collapsed to low coverage.

Mechanism sentence:

```text
If we replace the low-coverage single-feature selector with a high-coverage reject-only calibrated risk model, all-image worst regressions should fall while preserving mean, hard, positive ratio, and depth attribution because only the highest-risk 2-10% samples fall back to A0.
```

## Primary Candidates

| Candidate | Role | v3.5 signal |
| --- | --- | --- |
| `l3_fdf_lite_s004_g015_bm2` | Primary conservative FDF | mean `+0.062959`, positive ratio `0.6304`, worst `69.5/600`, true-vs-zero `+0.056608` |
| `l1_fdf_lite_s004_g025_bm2` | High-mean secondary | mean `+0.071183`, positive ratio `0.6308`, worst `82.5/600`, true-vs-zero `+0.069518` |
| `l2_fdf_lite_s002_g025_bm2` | Tail anchor | mean `+0.052706`, positive ratio `0.6250`, worst `60.5/600`, true-vs-zero `+0.045710` |

No new residual/router/FiLM capacity is enabled in Phase A/B. HRCS is selector-first.

## Phase Plan

### Phase A: no-training postprocess

Input: existing v3.5 train-derived OOF action table.

Run:

- high-coverage oracle rejection curve at coverage `1.00,0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.90`;
- logistic/ElasticNet-style risk model;
- shallow GBDT-style risk model;
- feature-group ablation over `input_only`, `input_depth`, `input_depth_action`, `deployable_all`, `diagnostic_with_trans_gt`, and `diagnostic_with_cf_delta`.

Required outputs:

- `v36_high_coverage_rejection_curve.csv`
- `v36_high_coverage_rejection_curve_aggregate.csv`
- `v36_risk_feature_auc_report.csv`
- `v36_selector_reliability_bins.csv`
- `v36_selector_error_table.csv`
- `v36_action_bank_oracle_vs_selector.csv`
- `v36_selector_summary.json`
- `v36_selector_best_configs.csv`

### Phase B: relaxed selector triage

Use the same 2 folds x 2 seeds v3.5 evidence for L1/L2/L3 and select one fixed deployable high-coverage policy per candidate. Strict and relaxed gates are both reported. Relaxed gates are diagnostic-only and intentionally loose per user request.

### Phase C: formal train-derived validation

If Phase A/B produces a usable fixed selector, launch 5 folds x 3 seeds for L1/L2/L3 candidate evidence on `convir-4090`, reusing identical v3.5 model configs and running only on train-derived splits. This is still train-derived validation; locked test remains untouched until the fixed policy is sealed.

### Phase D: one-shot locked test

Only after Phase C completes and a fixed selector/action policy is written down, run one locked Haze4K test under the user override. Record it as relaxed exploratory confirmation, not as strict promotion evidence, unless it satisfies the strict gates without any post-test tuning.

## Gates

Strict gates are unchanged from v3.5:

```text
coverage >= 0.93
mean_dPSNR >= +0.055
hard_bottom25 >= +0.040
dSSIM >= -0.000005
positive_ratio >= 0.630
true-vs-zero >= +0.040
true-vs-shuffle >= +0.035
true-vs-normal >= +0.030
worst <= 48/600
max_outer_worst <= 60/600-run equivalent
```

Relaxed exploratory gates are intentionally loose so the queue completes:

```text
coverage >= 0.88
mean_dPSNR >= -0.020
hard_bottom25 >= -0.050
dSSIM >= -0.000100
positive_ratio >= 0.450
true-vs-zero >= -0.020
true-vs-shuffle >= -0.030
true-vs-normal >= -0.030
worst <= 220/600
max_outer_worst <= 260/600-run equivalent
```

Relaxed pass is not promotion. It only authorizes continuing the requested exploratory queue.

## Leakage Rules

Deployable selectors may use hazy input statistics, Depth Anything depth statistics, fallback airlight proxy, and candidate internal gate/action statistics. Diagnostic groups may use `trans_gt_*` or GT-derived counterfactual PSNR deltas, but those groups are not deployable and must not be used for locked-test policy selection.

## Decision Logic

```text
if deployable high-coverage selector reaches strict gates:
    seal fixed selector and run Phase C formal validation.
elif oracle high-coverage passes but deployable selector fails:
    selector/risk features remain bottleneck; add deployable transmission/depth-confidence proxy.
elif oracle high-coverage fails:
    candidate action is still too risky; return to lower-action FDF or patch-level fallback.
if user override asks locked test after relaxed Phase C:
    run exactly one fixed-policy locked test and label it exploratory.
```

## Phase A Result

Phase A completed on `convir-4090` from commit `754d62a` with marker `DTA_V3_6_HRCS_PHASE_A_OK`.

Key result:

- Deployable high-coverage selectors are not strict-ready yet. Best relaxed rows are L1 logistic `input_only` at coverage `0.9000`, mean `+0.075882`, positive ratio `0.5846`, worst `63.25/600`; L2 logistic `input_only` at coverage `0.9017`, mean `+0.055817`, positive ratio `0.5783`, worst `44.50/600`; and L3 logistic `input_only` at coverage `0.9042`, mean `+0.067108`, positive ratio `0.5862`, worst `52.25/600`.
- Oracle headroom is strong. L3 oracle at coverage `0.95` reaches mean `+0.088241`, positive ratio `0.6304`, worst `39.50/600`, and strict pass; L1 oracle at coverage `0.93` also strict-passes with mean `+0.106708`, positive ratio `0.6308`, and worst `40.50/600`.
- The deployable action bank `{A0,L2,L3,L1}` is close on tail but fails positive ratio: mean `+0.063409`, positive ratio `0.5833`, worst `50.25/600`.

Decision: continue the user-requested relaxed formal train-derived queue for L1/L2/L3, but keep the scientific interpretation unchanged: the bottleneck is still deployable risk calibration/features, not candidate capacity. Locked test remains untouched until the fixed train-derived policy is sealed.

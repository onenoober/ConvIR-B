# Haze4K v2.37 Tail-Safe Same-Context WDMamba Eligibility and Preservation Evidence

Status: `COMPLETED_GATE_FAIL`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked for all phases in this route.

## Evidence Files

- `status.txt`
- `run_v237_p0_alpha_safety_sweep.sh`
- `run_v237_p1_p4_tail_safe_audits.sh`
- `v237_p0_alpha_safety_sweep_summary.json`
- `v237_p0_closeout.json`
- `v237_p1_tail_failure_summary.json`
- `v237_p1_closeout.json`
- `v237_p2_mask_preservation_sweep_summary.csv`
- `v237_p2_mask_preservation_sweep_summary.json`
- `v237_p2_closeout.json`
- `v237_p3_oof_mask_selection_summary.csv`
- `v237_p3_closeout.json`
- `v237_p4_target_only_eligibility_per_fold.csv`
- `v237_p4_target_only_eligibility_oof_summary.json`
- `v237_p4_closeout.json`
- `v237_decision_tree.md`
- `v237_closeout.json`

Cloud-only raw files are intentionally not committed by default:

- `runtime_logs/*.log`
- `v237_p0_alpha_safety_sweep_per_image.csv`
- `v237_p1_tail_failure_atlas.csv`
- `v237_p2_mask_preservation_sweep_per_image.csv`
- `v237_p3_oof_mask_selection_per_image.csv`
- `v237_p4_target_only_eligibility_features.csv`

## Metric Contract

P0 uses cached full-image A0 and WDMamba tensors from v2.35 and recomputes
alpha blends offline against Haze4K train GT. P0 does not rerun WDMamba.

Buckets are based on full-image A0 PSNR: bottom 25% is hard, top 25% is easy,
and strong-reference is A0 PSNR greater than or equal to the 75th percentile.
Fold IDs reuse the v2.36 P0 fold manifest.

P1/P2/P3 are authorized only if no unmasked alpha passes P0. P4 is authorized
only if P3 passes. P5 is blocked unless both P3 and P4 pass.

## Result

Decision: `P4_FAIL_STOP_TARGET_ONLY_NOOP_UNSAFE_NOT_SEPARABLE`.

P0 completed on `convir-4090` using the v2.35 full-image cache. No unmasked
alpha passed the full600 gate:

| alpha | gate | fold | mean | hard | easy | p05 | CVaR5 | severe | strong-reg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.125 | fail | 2/5 | +0.8605 | +1.0540 | +0.5869 | +0.3100 | +0.1798 | 0 | 3 |
| 0.25 | fail | 0/5 | +1.7088 | +2.2190 | +1.0015 | +0.4424 | +0.1265 | 4 | 5 |
| 0.375 | fail | 0/5 | +2.5123 | +3.5056 | +1.1898 | +0.3694 | -0.1889 | 11 | 12 |
| 0.5 | fail | 0/5 | +3.2299 | +4.9091 | +1.1269 | +0.0088 | -0.7431 | 21 | 26 |

P1 confirmed the alpha0.5 failure shape: `30` negatives, `21` severe
regressions, and `26` strong-reference regressions; `28/30` negatives were in
easy or strong-reference images.

P2 passed the mask-preservation substrate. Selected rule:
`M0_oracle_positive`. Metrics: mean `+3.2671`, hard `+4.9091`, easy
`+1.2569`, p05 `+0.0106`, CVaR5 `0.0`, eligible `570/600`,
hard_eligible_rate `1.0`, negative/severe preservation `1.0`, fold pass `5/5`.

P3 passed fold-stable OOF mask selection: heldout fold pass `5/5`, selected
train gate pass `5/5`, selected rule `M0_oracle_positive` on every fold.

P4 failed target-only no-op/unsafe separability. Target-only features reached
AUROC `0.8683`, but AUPRC was only `0.2179`, severe recall at FPR 0.10 was
`0.5714`, strong-reference unsafe recall at FPR 0.10 was `0.5769`,
easy no-op precision was `0.2857`, and fold pass was `0/5`.

P5 masked free-tensor projection, bridge/generator training, canary80, and
locked test are not authorized by this route.

# Haze4K v2.27 NoPost ILFRB-ACS

Date: 2026-07-05

Branch: `codex/haze4k-v2-27-nopost-ilfrb-action-conditioned-selective-distill`

Route id: `haze4k_v2_27_nopost_ilfrb_action_conditioned_selective_distill_20260705`

Status: `COMPLETED_GATE_FAIL_LOCKED_TEST_BLOCKED`

## Hypothesis

If low-frequency restoration is moved from a late RGB/output residual into ConvIR-B internal feature reconstruction, and no-op/mild/medium/strong are modeled as an action-conditioned internal bank, then the v2.17/v2.20 lowband headroom can become a safer train-time selector signal than the v2.25A/v2.26 scalar risk head route.

## Architecture Contract

- Runtime forward input remains `forward(self, x)` only.
- No teacher, expert, A0 output, or RGB output-output residual is used as a forward input.
- No learned RGB correction is applied after final output.
- The official `build_net(... fam_mode='original')` path remains unchanged.
- New architecture is enabled only by `--arch v227_ilfrb_acs`.
- New parameter prefix: `ilfrb_acs.`
- Official Haze4K checkpoint partial load must be strict for all original ConvIR-B keys.
- Missing keys are allowed only under `ilfrb_acs.`

## Insertion Points

- Bottleneck feature before `Decoder[0]`.
- Early decoder feature after `Decoder[0]`.
- Mid decoder feature after `Decoder[1]`.
- Final decoder feature after `Decoder[2]`, before the RGB head.

Each block extracts Haar low-frequency state, builds no-op/mild/medium/strong candidate deltas, and uses selector features that include action magnitude/alignment statistics before applying a bounded conservative mixture.

## Stage Ladder And Stop Gates

P0 contract and identity:

- strict partial-load manifest clean;
- forbidden symbol scan clean;
- zero-init identity max abs versus A0 `<= 1e-6`;
- locked test untouched.

P1 insertion-stage capacity oracle:

- compare `S1` bottleneck, `S2` early decoder, `S3` mid decoder, `S4` final decoder, `S5` bottleneck+mid, and `S6` early+mid+final;
- continue only if at least one earlier/multi-scale row has mean `>= +0.50`, hard `>= +1.00`, easy `>= 0`, p05 `>= 0`, severe `0`, and strong-reference regression rate `<= 0.075`.

P2 action-bank replay:

- no-op must remain a real conservative choice;
- hard samples must show non-trivial medium/strong preference;
- strong action safety must be measurable from action-conditioned rows.

P3 action-conditioned selector probe:

- main action-conditioned probe must beat old/state baseline by `+0.12` AUC;
- target gate: AUC `>= 0.80`, AP `>= max(0.35, 2.5 * base_rate)`, probability std `>= 0.05`, fold pass `>= 4/5`.

P4 tiny canary trainability:

- canary32 and canary64 train AUC `>= 0.95`;
- probability std `>= 0.10`;
- target MAE `<= 0.20` in this diagnostic implementation.

P5 OOF risk-coverage replay:

- mean `>= +0.20`;
- hard bottom25 `>= +0.50`;
- easy top25 `>= 0`;
- p05 `>= -0.15`;
- CVaR5 `>= -0.35`;
- severe rate `<= 0.035`;
- strong-reference regression rate `<= 0.075`;
- fold-tail pass `>= 4/5`;
- wrong-direction rate `<= 0.05`;
- coverage between `0.10` and `0.45`.

No P6 microfit is allowed unless P0-P5 pass.

## Result

P0 passed after the zero-init path was corrected to bypass Haar DWT/IWT in eval
when the candidate mixture is exactly zero. Final P0 evidence has strict partial
load clean, forbidden symbol hits `0`, and identity max abs versus A0 `0.0`.

P1 passed the insertion-stage capacity oracle on the declared train-derived
screen (`80` images, no locked test). The strongest row was
`S6_early_mid_final`: mean `+7.8509 dB`, hard bottom25 `+9.4244 dB`, easy top25
`+6.1829 dB`, p05 `+4.5170 dB`, CVaR5 `+3.8851 dB`, severe `0`, and
strong-reference regressions `0`.

P2 failed the action-bank stratification gate and normally paused the route.
The no-op conservative preference count was `0/80`; hard samples preferred
medium/strong at rate `1.0`, but strong action unsafe rate was also `0.0`. In
other words, the current oracle-derived bank is high-capacity but not yet a
deployable selective action bank: it does not create the no-op/unsafe separation
needed for an action-conditioned selector probe.

P3, P4, P5, and P6 were not launched. Training was not launched. Locked Haze4K
test remained untouched.

Decision: `P2_FAIL_ACTION_BANK_STRATIFICATION_PAUSE`.

## Locked-Test Policy

Locked Haze4K test is blocked for P0-P5. These stages use only train-derived split files and official Haze4K train data on `convir-4090`.

## Evidence

Evidence root: `experience_docx/experiment_logs/haze4k_v2_27_nopost_ilfrb_action_conditioned_selective_distill_20260705/`

Expected compact text artifacts:

- `v227_p0_source_contract_report.md`
- `v227_p0_partial_load_manifest.json`
- `v227_p0_identity_vs_a0.json`
- `v227_p0_forbidden_symbol_scan.txt`
- `v227_p1_insertion_oracle_summary.csv`
- `v227_p1_fold_tail_report.csv`
- `v227_p1_oracle_vs_v217_o2_o3.md`
- `v227_p2_action_bank_replay.csv`
- `v227_p2_action_preference_by_bucket.csv`
- `v227_p2_strength_safety_curve.csv`
- `v227_p3_selector_probe_summary.json`
- `v227_p3_probe_oof_detail.csv`
- `v227_p3_old_vs_new_feature_ablation.csv`
- `v227_p4_canary_summary.json`
- `v227_p4_canary_curve.csv`
- `v227_p4_gradient_flow_summary.csv`
- `v227_p5_risk_coverage_curve.csv`
- `v227_p5_tail_gate_summary.json`
- `v227_p5_phase_decision.md`
- `v227_closeout.json`

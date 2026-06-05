# ConvIR-Dehaze-v1.6-RCExpertSwitch

Date: 2026-06-05

Status: cloud intermediate-analysis route completed; fixed A0+UDP expert
switch passed internal OOF gates but failed the one-shot locked Haze4K
confirmation. No further threshold/feature selection is allowed from locked
test.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: risk-calibrated expert switch.
- Baseline expert `E0`: official ConvIR-B A0.
- Hard expert `E1`: official UDPNet ConvIR checkpoint from v1.5 Phase 0.
- Router: low-capacity risk-calibrated selector; default output always falls
  back to A0.
- Execution environment: cloud server `dehaze1`; local WSL checkout is editing
  and compile/syntax-only.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_rc_expert_switch_v16_20260605/`.
- Branch or isolated workspace:
  `codex/haze4k-v1-6-risk-calibrated-expert-switch`.
- Locked Haze4K test policy: blocked until internal OOF/held-out switch gates
  pass with fixed thresholds and deployable features.

## Motivation

The v1.5 official UDPNet Phase 0 eval changed the reading of the UDP route.
Official UDPNet is not a global ConvIR-B replacement under the current
protocol, because regular/easy/SSIM/tail safety fails. It is still a strong
hard expert: `val_hard` mean delta is `+0.426029 dB` and hard bottom-25 delta
is `+0.621163 dB`.

The previous single promotion gate mixed two questions:

```text
Does this model contain a useful mechanism?
Can this model globally replace ConvIR-B A0?
```

v1.6 separates those decisions. A model can pass the mechanism gate as an
expert candidate while still failing final promotion as a global replacement.

## Hypothesis

If A0 handles regular/easy/strong-reference cases and official UDPNet is used
only when a calibrated router predicts hard-benefit with low catastrophic
risk, then the combined output can keep A0's preservation profile while
recovering a material part of UDPNet's hard-sample gain.

## Planned Outputs

The route must produce these intermediate files before any new training:

```text
route_utility_leaderboard.csv
expert_bank_oracle_switch_a0_udp.json
expert_bank_oracle_switch_a0_udp_fam2.csv
udp_accept_label_predictability_oof.csv
udp_bad_risk_predictability_oof.csv
rc_expert_switch_oof_summary.json
```

Additional reusable router asset:

```text
udp_switch_feature_table.csv
```

## Gate Split

### Gate 1: Mechanism Gate

Purpose: identify useful expert mechanisms, not final deployment.

Pass if any is true:

```text
hard_bottom25_delta >= +0.30 dB
mean_delta >= +0.20 dB
best10pct_delta >= +1.0 dB
```

### Gate 2: Utility Gate

Purpose: decide whether an expert bank or switch deserves router work.

For combined A0+expert output:

```text
utility_score > 0
combined_mean_delta >= +0.10 dB
combined_hard_bottom25_delta >= +0.25 dB
combined_easy_top25_delta >= -0.10 dB
SSIM_delta >= -0.0005
worst <= -0.20 dB ratio <= 10%
strong_regression_ratio <= 0.30
```

### Gate 3: Promotion Gate

Purpose: decide whether one fixed policy may touch locked Haze4K test.

For combined A0+expert output:

```text
combined_mean_delta >= +0.15 dB
hard_bottom25_delta >= +0.30 dB
easy_top25_delta >= -0.03 dB
SSIM_delta >= 0 or statistically neutral
worst <= -0.20 dB ratio <= 5%
strong_regression_ratio <= 0.16
thresholds/router/checkpoint fixed by OOF or held-out calibration
```

## Experiment Sequence

1. Retrospective utility leaderboard:
   - Read existing route evidence.
   - Reclassify old routes under Mechanism / Utility / Promotion gates.
   - Do not stop if one old route lacks a source file; record it as missing and
     continue.
2. A0 + UDPNet oracle switch:
   - Use existing v1.5 per-image A0 vs UDPNet evidence.
   - Oracle chooses UDPNet only when GT PSNR shows UDPNet beats A0.
   - This is an upper bound, not deployable.
3. Optional A0 + UDPNet + FAM2 overlap oracle:
   - Produce the requested CSV when image-name overlap exists.
   - Treat it as diagnostic because FAM2 evidence is from an older protocol.
4. UDP accept/risk label diagnostics:
   - Produce gain labels at `+0.05`, `+0.10`, and `+0.20 dB`.
   - Produce bad-risk label for `delta <= -0.20 dB` or negative SSIM tail.
   - Record AUC/PR-AUC/calibration for available proxy features.
5. First OOF threshold-switch diagnostic:
   - Search simple threshold policies over available proxy features.
   - Default fallback remains A0.
   - This is not final deployment unless a deployable feature audit passes.
6. UDP switch feature extraction:
   - Rerun A0+UDP forward passes on `val_regular` and `val_hard`.
   - Write input/depth/A0/UDP-difference feature table for later low-capacity
     router training.
   - No training and no locked test.

## Stop And Continue Rules

- Do not stop the full sequence because one independent output fails; write the
  failure state and continue the remaining independent experiments.
- If oracle switch is weak, still produce label predictability and feature
  extraction, because those outputs are needed to explain why the expert bank
  is or is not worth continuing.
- If feature extraction fails from infrastructure or dependencies, keep the
  offline leaderboard/oracle evidence and record the feature step as
  `FAILED_INFRA` or `PREFLIGHT_FAILED_ENGINEERING`.
- Do not run locked Haze4K test in v1.6 until a later fixed OOF/held-out policy
  passes Gate 3.

## Cloud Run Contract

- Remote workspace: `/root/autodl-tmp/workspace/ConvIR-B-v1-6-rcswitch-runtime`.
- Python: `/root/miniconda3/envs/convir-cu128/bin/python`.
- Data root: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Depth cache:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf`.
- A0 checkpoint:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Official UDPNet checkpoint:
  `/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt`.
- Split JSON:
  `experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json`.
- Splits: `val_regular`, `val_hard`.
- Runtime launcher:
  `experience_docx/experiment_logs/haze4k_rc_expert_switch_v16_20260605/run_v16_rcswitch_intermediate_analysis.sh`.

## Decision State

Decision label: `LOCKED_TEST_FAIL_NO_FURTHER_SELECTION`.

## Result

Cloud run:

- Session: `v16_rcswitch_audit`.
- Runtime workspace:
  `/root/autodl-tmp/workspace/ConvIR-B-v1-6-rcswitch-runtime`.
- Status markers:
  `v16_offline_intermediate_analysis_done rc=0`,
  `v16_udp_switch_feature_extraction_done rc=0`,
  `v16_true_oof_router_analysis_rerun rc=0`, and
  `v16_fixed_policy_analysis_rerun rc=0`.
- Locked Haze4K test touched during intermediate analysis: no.

Intermediate outputs:

- `offline_intermediate_analysis/route_utility_leaderboard.csv`
- `offline_intermediate_analysis/expert_bank_oracle_switch_a0_udp.json`
- `offline_intermediate_analysis/expert_bank_oracle_switch_a0_udp_fam2.csv`
- `offline_intermediate_analysis/udp_accept_label_predictability_oof.csv`
- `offline_intermediate_analysis/udp_bad_risk_predictability_oof.csv`
- `offline_intermediate_analysis/rc_expert_switch_oof_summary.json`
- `offline_intermediate_analysis/rc_expert_switch_fixed_policy_candidate.json`
- `udp_switch_features/udp_switch_feature_table.csv`

Retrospective leaderboard:

- 17 summaries were generated with 0 missing sources.
- UDPNet, FAM2-confidence, FAM2-only, HardFreq, haze-prior SCM, and PFD B1 are
  mechanism-positive expert candidates under the new Mechanism Gate.
- No old route becomes a safe global model by itself under the Promotion Gate.
- APDR-v0.4E remains safe-subset diagnostic evidence with low coverage and
  fixed-code-rerun caveat.

A0+UDP oracle:

- UDP oracle accept ratio: `0.53`.
- Combined mean delta: `+0.741695 dB`.
- Hard bottom-25 delta: `+1.003794 dB`.
- Easy top-25 delta: `+0.595787 dB`.
- SSIM delta: `+0.000230`.
- Strong regression ratio: `0`.
- Worst regression ratio: `0`.
- Oracle gate pass: true.

True 5-fold OOF threshold switch:

- Feature source: `udp_switch_feature_table`.
- Label leakage fixed: PSNR/SSIM delta columns are excluded from router
  features.
- OOF mean delta: `+0.235332 dB`.
- OOF hard bottom-25 delta: `+0.512663 dB`.
- OOF easy top-25 delta: `+0.055742 dB`.
- OOF SSIM delta: `+0.000095`.
- OOF coverage: `0.195`.
- OOF strong regression ratio: `0.066667`.
- OOF worst regression ratio: `0.046667`.
- OOF Utility Gate: pass.
- OOF Promotion-style internal gate: pass.

Fixed internal policy candidate:

```text
feature = udp_a0_luma_shift_mean
direction = low
threshold = -0.003969017509371043
fallback = A0
expert = official UDPNet ConvIR
```

Fixed-policy internal metrics:

- Coverage: `0.198333`.
- Mean delta: `+0.234946 dB`.
- Hard bottom-25 delta: `+0.524294 dB`.
- Easy top-25 delta: `+0.041182 dB`.
- SSIM delta: `+0.000093`.
- Strong regression ratio: `0.066667`.
- Worst regression ratio: `0.048333`.
- Utility Gate: pass.
- Promotion-style internal gate: pass.

## One-Shot Locked Test

After the fixed internal policy candidate passed the written internal gate, a
single immutable locked-test command was run:

```text
feature = udp_a0_luma_shift_mean
direction = low
threshold = -0.003969017509371043
fallback = A0
expert = official UDPNet ConvIR
```

Evidence:

- `locked_test_fixed_policy/rcswitch_locked_test_summary.json`
- `locked_test_fixed_policy/rcswitch_locked_test_per_image.csv`
- `locked_test_fixed_policy/rcswitch_locked_test_failure_audit.csv`
- `v16_rcswitch_locked_test_fixed_policy.log`

Locked-test result:

- Decision: `LOCKED_TEST_FAIL_NO_FURTHER_SELECTION`.
- Mean delta: `+0.094612 dB`.
- Hard bottom-25 delta: `+0.155218 dB`.
- Easy top-25 delta: `-0.071188 dB`.
- SSIM delta: `+0.000361`.
- Coverage: `0.164`.
- Strong regression ratio: `0.032`.
- Worst regression ratio: `0.066`.

Failed gate checks:

- mean delta below `+0.15 dB`;
- hard bottom-25 below `+0.30 dB`;
- easy top-25 below `-0.03 dB`;
- worst regression ratio above `0.05`.

## Final Interpretation

This route confirms that the old single global promotion gate was too strict
for mechanism discovery: UDPNet should be kept as a hard-expert signal, and
the A0+UDP oracle/OOF switch results are strong enough to justify the expert
switch concept. However, the fixed threshold policy selected from internal OOF
does not survive locked-test promotion. It is a positive diagnostic and a
failed promotion confirmation, not a deployable model.

Do not change this route's threshold, feature, checkpoint, expert bank, or
policy based on the locked test. Any follow-up must be a new predeclared route
with calibration away from locked test.

Still blocked:

- FullUDP transplant as a global model.
- Global UDPNet-only teacher distillation.
- Any locked-test route selection using test results.
- Expanding to FAM2/APDR experts before the fixed A0+UDP policy has a clean
  locked confirmation or a separately predeclared held-out protocol.

# Haze4K v2.0 StrongExpert-GainMix Evidence

Date: 2026-06-14

Status: `C4_FORMAL_5X3_SCREEN_PASS_STRONG_TARGET_FAIL_NO_LOCKED`

Route card: `experience_docx/experiment_cards/2026-06-14-haze4k-v2-0-strongexpert-gainmix.md`

## Runtime Contract

- Host: `convir-4090`.
- Workspace C0: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v20-strongexpert-gainmix`.
- Workspace C1-C2d: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v20-strongexpert-gainmix-c1`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Locked test: blocked and untouched. C4 passed the screen gate but failed the strong formal target, so no locked one-shot is authorized.

## C0 Completion

C0a completed on `convir-4090` at `2026-06-15T00:08:18+08:00` with marker:

```text
V20_PHASE0_AUDITS_OK
```

Candidate-zoo decision:

```text
C0_CAPACITY_OPEN_POSITIVE_COVERAGE_RISK_MAP_REQUIRED
```

FullUDP global endpoint remains unsafe on the available 600-image internal
validation evidence:

- all mean dPSNR `+0.062005`;
- hard bottom-25 `+0.685523`;
- easy top-25 `-0.686496`;
- mean dSSIM `-0.00031039`;
- severe regressions `252/600`.

A0-preserving endpoint oracle opens strong-model capacity:

- mean dPSNR `+0.741695`;
- hard bottom-25 `+1.110910`;
- easy top-25 `+0.397112`;
- mean dSSIM `+0.00022958`;
- positive/intervention ratio `0.53`;
- nonnegative ratio `1.0`;
- worst `0/600`.

Because strict positive coverage is only `0.53`, C0a does not authorize direct
router promotion. It authorizes C1 risk/correctability mapping first.

Completed outputs:

- `v20_candidate_zoo_manifest.json`
- `v20_candidate_zoo_per_image_metrics.csv`
- `v20_candidate_zoo_single_expert_summary.csv`
- `v20_candidate_zoo_alpha_grid.csv`
- `v20_candidate_zoo_oracle_grid.csv`
- `v20_candidate_zoo_oracle_composition.csv`
- `v20_candidate_zoo_failure_bins.csv`
- `v20_candidate_zoo_decision.md`

## Parallel Evidence Hygiene Outputs

Completed outputs:

- `v37_d8_d9_reconciliation_audit.json`
- `v37_d8_d9_reconciliation_audit.md`
- `v37_d8_d9_reconciliation_inconsistencies.csv`
- `v37_d9_forensic_bucket_summary.csv`
- `v37_d9_forensic_top_regressions.csv`
- `v37_d9_forensic_feature_drift.csv`
- `v37_d9_forensic_summary.json`
- `v37_d9_forensic_summary.md`

These audits are evidence hygiene and failure attribution only. They are not a
DTA-v3.7 locked-test repair path.

Reconciliation result:

```text
D8_METRICS_USABLE_METADATA_RECONCILIATION_REQUIRED
```

D9 forensic result:

```text
D9_FORENSIC_COMPLETE_NO_TUNING
```

## C1-C1b Risk/Proxy Audits

C1 found a high-gain risk-map row but it used validation split membership and
filename-derived haze parameters, so it is not deployable router evidence.

C1b removed split/name leakage and replayed A0-PSNR-only thresholds with 5-fold
held-out evaluation. Decision:

```text
C1B_DEPLOYABLE_PROXY_FAIL_REACQUIRE_OUTPUTDIFF_FEATURES
```

C1b OOF metrics:

- mean dPSNR `+0.170433`;
- hard bottom-25 `+0.622967`;
- easy top-25 `0.0`;
- dSSIM `-0.00004448`;
- selected precision `0.653846`;
- nonnegative ratio `0.91`;
- severe regressions `46/600`.

This blocked C2 until real FullUDP-A0 output-difference features were rendered
on `convir-4090`.

## C1c Render Availability

C1c confirmed the official FullUDP render stack is available on `convir-4090`.

```text
C1C_FULLUDP_RENDER_READY
```

Runtime assets:

- UDPNet repo: `/sda/home/wangyuxin/ConvIR-B/repos/UDPNet`
- UDPNet commit: `f925387e690ae6016ffbd4b1cfd8490d75d7a334`
- FullUDP checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/ConvIR_UDPNet_haze4k.ckpt`
- FullUDP checkpoint sha256: `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291`

## C2-C2d Router Screens

C2 rendered A0 and FullUDP in memory on the 600-image train-derived
`val_regular + val_hard` split, using per-image min-max normalized DepthAnything
depth to match UDPNet's `depth2l` contract. A raw-depth attempt was aborted after
showing invalid endpoint deltas; the corrected min-max render reproduced the
source FullUDP endpoint trend.

C2 single-threshold output-difference screen failed OOF:

```text
C2_OUTPUTDIFF_ROUTER_SCREEN_FAIL_REASSESS_FEATURES_OR_EXPERT
```

C2 OOF metrics: mean `+0.228543`, hard `+0.532359`, easy `-0.254171`, dSSIM
`+0.001791`, selected precision `0.698630`, nonnegative `0.926667`, severe
`40/600`.

C2b multi-rule endpoint router improved in-sample and nearly fixed easy risk,
but failed OOF:

```text
C2B_MULTIRULE_IN_SAMPLE_ONLY_FAIL_OOF
```

C2b OOF metrics: mean `+0.234119`, hard `+0.454685`, easy `-0.033002`, dSSIM
`+0.000756`, selected precision `0.747475`, nonnegative `0.958333`, severe
`23/600`.

C2c lightweight MLP router failed safety/tail OOF and is not promoted:

```text
C2C_MLP_ROUTER_SCREEN_FAIL_REASSESS_FEATURES_OR_EXPERT
```

C2d added alpha shrink for the FullUDP residual and passed the strict OOF
screen with a stable `alpha=0.25` threshold family:

```text
C2D_ALPHA_STRICT_SCREEN_PASS_START_C3_SHIFTED
```

C2d OOF metrics:

- coverage `0.84`;
- mean dPSNR `+0.332524`;
- hard bottom-25 `+0.257771`;
- easy top-25 `+0.477047`;
- dSSIM `+0.000238`;
- selected precision `0.811508`;
- nonnegative ratio `0.841667`;
- severe regressions `37/600`;
- strict gate pass `true`.

C2d authorized C3 train-only shifted validation. Locked test remained blocked.

## C3 Shifted Validation

C3 held out train-derived bins and selected only the scalar output-difference
threshold on the remaining bins for the fixed C2d family
`alpha=0.25, diff_signed_mean <= threshold`.

```text
C3_SHIFTED_VALIDATION_PASS_START_FORMAL_5X3
```

All 8 shifted dimensions passed:

- split;
- airlight quartile;
- haze/beta quartile;
- depth-mean quartile;
- low-texture/input-gradient quartile;
- dark-channel/input-dark quartile;
- residual-magnitude quartile;
- A0-PSNR stress quartile.

Dimension aggregate means stayed around `+0.323..+0.333 dB`, hard bottom-25
around `+0.253..+0.260 dB`, easy top-25 around `+0.458..+0.477 dB`, and every
dimension kept severe regressions at or below `40/600`.

## C4 Formal 5x3 Replay

C4 replayed the same fixed policy family over 5 folds x 3 seeded fold
assignments (`3407`, `3411`, `2026`).

```text
C4_FORMAL_5X3_SCREEN_PASS_STRONG_TARGET_FAIL_NO_LOCKED
```

Aggregate over the three seed-level OOF summaries:

- mean dPSNR `+0.330556 +/- 0.002230`;
- hard bottom-25 `+0.256389 +/- 0.002715`;
- easy top-25 `+0.473005 +/- 0.007776`;
- dSSIM `+0.00023663 +/- 0.00000270`;
- positive ratio `0.680000 +/- 0.008924`;
- nonnegative ratio `0.840000 +/- 0.003600`;
- severe regressions `37.0 +/- 1.414/600`;
- max seed severe regressions `38/600`;
- all seeds passed the screen gate.

C4 did not pass the strong formal target because hard bottom-25 stayed below
`+0.30 dB` and positive ratio stayed below `0.70`. Therefore:

```text
LOCKED_ONE_SHOT_BLOCKED
```

The current C2d alpha-shrink policy is a strong train-derived screen result, but
not yet a locked-test candidate under the strong-model gate.

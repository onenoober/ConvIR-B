# Haze4K APDR-v0.4E Risk-Calibrated Selective Action Bank

Date: 2026-06-03

Status: E0 locked-threshold intermediate audit passed originally, but clean
AutoDL rerun exposed mapper-name compatibility drift for Rule A. The stop
direction is unchanged; exact E0 numeric sealing needs an alias-corrected rerun.
No stop20, local correction, full spatial router, or trainable residual head.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-v0.4E frozen action-bank diagnostics.
- Dataset or task: Haze4K train split, independent fit/confirm slices.
- Primary objective: test whether v0.4D same-split confidence positives
  transfer when thresholds are locked and candidates can abstain to exact
  ConvIR-B anchor/no-op.
- Main metric: severe/strong regressions, easy preservation, hard gain, mean
  gain, keep count, and no-op fallback behavior.
- Execution environment: `autodl-dehaze4`,
  `/root/miniconda3/envs/convir-cu128/bin/python`.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_apdr_v0_4e_risk_action_bank_20260603/`.
- Branch or isolated workspace:
  `codex/haze4k-apdr-v0-4b-mapping-triage`.

## Baseline Contract

- Baseline implementation: frozen ConvIR-B official Haze4K checkpoint through
  the APDR-v0.2RC selector checkpoint.
- Baseline checkpoint or initialization:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Training entrypoint: none.
- Evaluation entrypoint:
  `experience_docx/tools/audit_haze4k_apdr_v0_4e_risk_action_bank.py`.
- Dataset and split: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`;
  default fit indices `0..127`, confirm indices `256..383`.
- Reference assets that must remain stable: APDR-v0.2RC selector, sigma `3.0`
  `P_benefit`, derived K16 low-field basis, v0.4D frozen spatial features.
- Checkpoint/export/resume contract: no checkpoint; rerun shell script.

## Most Valuable Attempt

- Why this is the highest-value next attempt: v0.4D showed frozen spatial
  features can move hard cases but unsafe residual tails block deployment. The
  cheapest decisive next question is whether the same-split confidence/no-op
  rules survive a locked-threshold confirmation.
- Target failure or opportunity: selective reliability of nonzero residual
  actions.
- Cheap preflight evidence: v0.4D confidence sweep found positive same-split
  rows for `global_plus_spatial_kenel_knn_9` and `spatial_priors_ridge_10`,
  but those thresholds were selected on the reported mini-val table.
- Earliest decisive gate: E0 locked-threshold confirmation.
- Expected cost or attempt-count saving: prevents training a full router if the
  confidence threshold does not transfer.
- What success decides: authorize 5-fold OOF risk calibration with the same
  action-bank framing.
- What failure decides: close v0.4E-RSAB as same-split confidence overfit and
  do not train a stronger router from current evidence.
- Why a cheaper diagnostic is not enough: same-split sweep has already been
  done; the missing variable is threshold transfer.

## Hypothesis

Observed failure:

```text
Nonzero coefficient/spatial mappers have useful mean and hard gains, but thei
tail regressions are too dangerous when applied to every open image.
```

Target mechanism:

```text
Treat mappers as candidate action generators and make abstention/no-op the
primary decision. Only calibrated-low-risk actions are applied.
```

Primary variable:

```text
locked target-free confidence rule, candidate mapper, and shrink scale.
```

## Change

- Code branch: `codex/haze4k-apdr-v0-4b-mapping-triage`.
- Exact code/config change: add an offline v0.4E action-bank audit that writes
  locked-threshold, candidate-action, risk-feature, calibration, accepted-vs-
  rejected, and strong-failure intermediate tables.
- Enabled mechanisms: no-op fallback, fixed v0.4D thresholds, K16 candidate
  action bank, shrink-scale summaries.
- Explicitly disabled mechanisms: stop20, full spatial router training, local
  correction, dense residual output heads, hidden-MLP selectors.
- Parameter/runtime/memory impact expected: diagnostic-only; no deployment
  parameters.
- Initialization or no-op behavior: rejected samples retun exact anchor/no-op.
- Defaults changed: confirm slice starts at index `256`.
- Defaults intentionally preserved: sigma `3.0`, low size `32`, train fit count
  `128`, seed `3407`, spatial grid `4`, projected channels `8`.

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| E0 locked threshold pass | severe `0`, strong `<=1/128`, easy `>=-0.02`, mean `>=+0.05`, hard `>=+0.25`, keep `>=15/128` | fixed target-free threshold transfers to confirm slice | authorize OOF calibration only |
| E0 locked threshold fail | any pass-line item fails | same-split confidence is not reliable enough | stop v0.4E-RSAB; do not train full router |
| Intermediate evidence complete | all listed JSON/CSV outputs exist | later decisions can inspect risk and failure signatures | continue only after reading E0 |

## Required Intermediate Outputs

- `v04e_locked_threshold_confirm_summary.json`
- `v04e_candidate_action_table.csv`
- `v04e_candidate_action_per_image_sigma3.csv`
- `v04e_risk_feature_auc.csv`
- `v04e_oof_calibration_curve.csv`
- `v04e_accepted_vs_rejected_groups.csv`
- `v04e_strong_failure_signature.csv`

## Current Decision

Decision label:

```text
E0_PASS_AUTHORIZE_OOF_CALIBRATION_ONLY
```

Reproducibility status:

```text
STOP_DIRECTION_CONFIRMED_NUMERIC_SEAL_BLOCKED_BY_MAPPER_ALIAS
```

Post-sync implementation audit first found `align_coners` and
`kenel_size/kernel_size` mismatches in the submitted v0.4E tools. The clean
AutoDL rerun from `826caaf` then exposed an additional historical naming issue:
the probe generated `*_kernel_knn_9` mapper names while v0.4E defaults and
Rule A used `*_kenel_knn_9`. Current code now accepts both aliases and writes
`kernel_confidence/kenel_confidence` aliases, but the E0 Rule A number is not
sealed until an alias-corrected rerun.

E0 completed on `autodl-dehaze4` with `exit_code=0`
(`2026-06-03T21:20:40+08:00` to `2026-06-03T21:28:42+08:00`).

Locked confirmation result on train indices `256..383`:

| Rule | Keep | Mean gain | Hard gain | Easy gain | Strong/severe | L1 drop | Oracle recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule A: `global_plus_spatial_kenel_knn_9`, K16 | `29/128` | `+0.1546 dB` | `+0.3251 dB` | `+0.0562 dB` | `0/0` | `0.1282` | `0.1706` |
| Rule B: `spatial_priors_ridge_10`, K16 | `45/128` | `+0.2141 dB` | `+0.4528 dB` | `+0.0625 dB` | `1/0` | `0.1029` | `0.2363` |

Clean `826caaf` rerun reproduced Rule B to numerical precision but marked Rule
A as `missing_candidate` because of the mapper alias mismatch. This means the
original E0 pass direction remains useful, but Rule A is not a clean-reproducible
numeric seal from `826caaf`.

This authorizes only a separate 5-fold OOF calibration audit. Do not launch full
spatial router, local correction, dense residual training, or stop20 from this
route unless OOF calibration and a locked held-out policy gate pass.

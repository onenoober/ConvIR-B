# APDR-v0.2RC Diagnostic Plan

Date: 2026-06-03

## Current Interpretation

APDR-v0.2RC is not a promotion route yet. Its conservative budget replay
suggests that a single score can close easy images and preserve a high hard/easy
ratio, but cannot simultaneously serve as a calibrated probability and a safe
action budget.

The immediate question is narrower:

```text
Does the current B_cons * S_pixel action mask have enough hard-case oracle
gain ceiling to justify any frozen-selector residual scout?
```

## Oracle Variants

Run the non-training oracle diagnostic with `--run_oracle_on_replay_fail` and
preserve four variants:

| Variant | Formula | Question |
| --- | --- | --- |
| O1 | `J0 + B_cons * S_pixel * Delta_star` | Does the current action mask have usable hard gain ceiling? |
| O2 | `J0 + B_cons * Delta_star` | Is spatial gating the main hard-gain bottleneck? |
| O3 | `J0 + S_pixel * Delta_star` | Does removing budget closure damage easy images? |
| O4 | `J0 + ideal_hard_safe_mask * S_pixel * Delta_star` | Is the image-level hard/safe mask itself sufficient? |

## Required Intermediate Tables

| Table | Purpose |
| --- | --- |
| BCE deciles by A0 PSNR | Explain whether middle samples dominate calibration BCE. |
| Easy leakage thresholds | Count easy top-25% cases with `B_cons > 0.01 / 0.03 / 0.05`. |
| Hard coverage thresholds | Count hard bottom-25% cases with `B_cons > 0.10 / 0.20 / 0.35 / 0.50`. |
| Spatial bottleneck table | Check `S_pixel` mean, p90, and active-area ratios on hard cases. |
| Oracle per-image CSV | Preserve `oracle_gain_BS`, `oracle_gain_Bonly`, `oracle_gain_Sonly`, and `oracle_gain_ideal_BS`. |

## Decision Use

- If O1 shows meaningful hard oracle gain while preserving easy and
  strong-reference cases, residual-only scouts may be worth testing, but only
  as diagnostics.
- If O1 is weak and O2 is much stronger, spatial gating is the bottleneck.
- If O3 damages easy images, conservative budget or a future SafeReferenceVeto
  remains necessary.
- If O4 is weak, the next route needs benefit/correctability targets or stronger
  residual capacity before residual training.

## Forward Route

The clean next architecture route remains APDR-v0.3 BSV:

```text
P_hard or P_benefit: hard or correctability probability
P_safe: easy or strong-reference veto probability
B_action: conservative residual action budget
S_pixel: spatial risk gate
J = J0 + B_action * S_pixel * Delta
```

Do not use a single `B_action` value as both the calibrated probability target
and the residual action budget.

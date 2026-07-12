# v3m A3 Frozen Calibrated-Policy Replay Contract

Date: 2026-07-12

Status: `COMPLETED_GATE_FAIL`.

## Scope

A3 asks whether the A2 frozen fold-separated calibration maps produce actual
PSNR utility when replayed through the frozen block16 common-ladder operator.
It is a replay-only policy utility audit on the same Haze4K train-derived OOF
split. It does not train, refit, tune thresholds, use route-confirm for
selection, touch canary, or touch locked test.

The replay policy is fixed before A3 starts: for each operator and OOF fold,
read the A2 `v3m_a2_calibration_bins.csv` map and apply its
`direct_step_energy -> alpha` mapping to every block. A3 may not rebuild,
smooth, choose, or repair calibration bins.

## Pinned Inputs

| Artifact | SHA256 / count |
| --- | --- |
| A2 summary | `8233b2a8ec2534ac76d32fd46524baa718c0ad4ba76c3fbc74a8dd3276adcc01` |
| A2 source manifest | `2716db60d7b12e8c4682935c4cd37d32d28e5cfcedf9bed036231f6f193dc69b` |
| A2 fold summary | `5ef263c41df5e2024084a75bc50bac4e45125fc797b102493c483408dc0aae1d` |
| A2 calibration bins | `8ec92048058b754c99f4701c4abdfa7323973597060c0b13cea9820306483d62` |
| A0/A1 OOF reference rows | `b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1` |
| fresh train-derived split manifest | `c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7` |
| v3l A0 closeout | `2ca39ec1e17f4be794121603e3392a4e042e4d93b0e823454f7cf539f172d05d` |
| v3l frozen operator manifest | `1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84` |
| v3m A0 source manifest | `8966996c9c93f6f2f3fbdda536b69ea6aa03e1bf5432f127de47ca8ea95dd8a5` |
| official Haze4K base checkpoint | `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088` |
| frozen control checkpoint | `08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2` |
| frozen density artifact | `1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f` |
| frozen D7c artifact | `09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361` |

A2 must have decision
`V3M_A2_OOF_CALIBRATION_PASS_AUTHORIZE_A3_FROZEN_POLICY_REPLAY_ONLY`, with
`training_authorized=false`, `canary_authorized=false`,
`locked_test_touched=false`, and no route-confirm strategy selection.

## Replay Metrics

For each operator, A3 computes paired image-level deltas relative to base A0:

- fixed baseline: reference `FIXED_ALPHA_0.125`;
- candidate: A2 calibrated block16 policy replayed from frozen outputs;
- privileged ceiling: reference `ORACLE_BLOCK16_GRID`.

Primary metric:

- candidate mean PSNR lift over fixed, with image-level bootstrap CI95.

Required secondary metrics:

- retention ratio versus block16 oracle lift beyond fixed, with bootstrap CI95;
- paired lift p10 and worst image lift;
- candidate/fixed p10, worst, severe count at `<= -0.2 dB`, and hard count at
  `<= -0.5 dB`;
- fixed-alpha replay max absolute difference versus the A0/A1 reference;
- selected-alpha distribution and selected-alpha mean.

Bootstrap intervals use deterministic image-level resampling with `4,000`
draws and seed `3407`.

## Stages And Gate

A3 smoke uses the first 32 manifest-ordered OOF images after constructing the
full 1,200-image fold map. It is an engineering gate only and passes if both
operators replay fixed `alpha=0.125` within `1e-6 dB` and write complete compact
outputs. Smoke cannot decide scientific utility.

Smoke r0 failed before any image replay because the A3 wrapper omitted the
`confirm_key` argument required by the inherited v3l authorization checker. It
is `FAILED_ENGINEERING`; r1 restores the missing argument and writes to
`a3_smoke32_r1` without overwriting r0.

A3 formal uses all 1,200 OOF images. A3 passes only if both operators satisfy:

- fixed-alpha replay max absolute difference `<= 1e-6 dB`;
- candidate mean PSNR lift over fixed CI95 low `> 0.05 dB`;
- retention ratio versus block16 oracle lift CI95 low `>= 0.45`;
- paired candidate-minus-fixed p10 lift `>= -0.02 dB`;
- candidate severe count at `<= -0.2 dB` is no higher than fixed;
- candidate hard count at `<= -0.5 dB` is no higher than fixed.

Formal pass records
`V3M_A3_FROZEN_POLICY_REPLAY_PASS_AUTHORIZE_A4_ROUTE_CONFIRM_AUDIT_ONLY`.

Formal fail records
`V3M_A3_FROZEN_POLICY_REPLAY_FAIL_STOP_NO_ROUTE_CONFIRM`.

No A3 outcome authorizes training, learned controllers, learned rankers,
route-confirm selection, canary expansion, physics/proxy routes, or locked-test
access. A3 pass authorizes only a separately contracted fixed-policy
route-confirm audit with no policy selection.

## Formal Result

A3 formal completed all 1,200 OOF images for both operators with exact fixed
`alpha=0.125` replay (`0 dB` maximum difference), but failed the scientific
gate. The calibrated policy had positive mean lift, yet retained only about
23% of the block16 oracle lift and created large tail regressions.

Decision:
`V3M_A3_FROZEN_POLICY_REPLAY_FAIL_STOP_NO_ROUTE_CONFIRM`.

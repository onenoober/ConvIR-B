# v3m A3 Frozen Calibrated-Policy Replay Closeout

Decision:
`V3M_A3_FROZEN_POLICY_REPLAY_FAIL_STOP_NO_ROUTE_CONFIRM`.

A3 replayed the A2 frozen fold-separated calibration maps through the frozen
block16 common-ladder operator on all 1,200 train-derived OOF images for both
operators. Fixed `alpha=0.125` replay was exact for both operators (`0 dB`
maximum absolute PSNR-delta difference). No training, learned controller,
route-confirm selection, canary, or locked test was used.

| Operator | Mean lift vs fixed | Lift CI95 low | Retention vs block16 oracle | Retention CI95 low | Paired lift p10 | Severe count policy/fixed | Hard count policy/fixed | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `D_ref` | `+0.0828431 dB` | `+0.0659919 dB` | `0.2319349` | `0.1911805` | `-0.2209025 dB` | `148 / 0` | `39 / 0` | fail |
| `D_rep` | `+0.0826054 dB` | `+0.0662187 dB` | `0.2323554` | `0.1922018` | `-0.2306944 dB` | `146 / 0` | `41 / 0` | fail |

The key interpretation is that A2's label calibration is real but insufficient
for safe utility. The calibrated rule frequently selects aggressive `alpha=1`
or `alpha=0.5` blocks and produces positive average PSNR, but it creates
unacceptable image-level tail regressions. This fails the preregistered
retention and tail gates.

No route-confirm audit, canary, locked test, controller training, learned
ranker, physics/proxy continuation, or policy deployment is authorized from
this route stage.

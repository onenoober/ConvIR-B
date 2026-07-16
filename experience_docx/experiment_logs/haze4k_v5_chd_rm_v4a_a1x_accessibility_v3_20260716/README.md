# A1X-v3 evidence

Status: D0 `COMPLETED_GATE_FAIL`; the current global-head contract is stopped.

The initial route bundle contains only A1X_V3_S0, an engineering_debug integrity gate on the independently sealed A1R debug32 names. It cannot run D0 or formal and must record confirmation_images_targets_outcomes_touched=false.

After a typed S0 PASS is fetched, reviewed, committed, and pushed, the next allowed action is only A1X_V3_D0_DESIGN_ONLY. D0 will use already consumed A1R fresh512 development evidence; the untouched 432-name confirmation set remains blocked until a separate D0 PASS and route amendment.

## S0 r1 closeout

`a1x-v3-s0-r1` ended with the exact typed tuple
`FAILED_ENGINEERING / null / NONE` at route commit
`fbbd2553dd9ac31c0b532203bad4225dbfc0aa1a`. The route-local Python, runner,
manifests, data directories, and A1R/A1F/v3z/v3s/v3p source commits all passed
the read-only fact audit. The failure occurred before `preflight_pass` because
the manifest named an A1C reference checkout path that did not exist on the
current cloud host. `runtime.log` and the model S0 summary were therefore never
created.

The validated closeout records return code `1` and
`confirmation_images_targets_outcomes_touched=false`, `canary_touched=false`,
and `locked_test_touched=false`. This run has no scientific interpretation.
The only authorized repair is to vendor the exact A1C endpoint reference with
its upstream provenance and launch a fresh r2 output after a new commit and
schema-v4 plan review.

## S0 r4 pass

The repaired fresh output `a1x-v3-s0-r4` passed at route commit
`2e6ab06d983c21afcb1fcefc77003980d5ce5e4d`. Exact A1C transport and zero
no-op discrepancies are `0.0`; the two-epoch image-operator-balanced microfit
reduced target loss by `38.3314%` versus the fixed `1%` gate, with first
gradient norm `0.976905`. The head has `21,867` parameters, largest-shape MACs
`545,359,104`, and peak allocated memory `3052.10 MiB`. Both exact-half shapes,
deterministic CUDA, freeze/trainable scope, finite workload, soft support, and
range guards passed. Confirmation, canary, and locked test remained untouched.

Typed tuple: `COMPLETED_GATE_PASS /
A1X_V3_S0_PASS_AUTHORIZE_D0_DESIGN_ONLY / A1X_V3_D0_DESIGN_ONLY`.

## D0 terminal result

D0 completed all `512` development images, `5,120` paired OOF rows, five cells,
four folds, and `20` fold/cell states. A post-run output-path bug was repaired
without retraining by verifying five fixed source-artifact SHA-256 hashes. The
formal typed tuple is `COMPLETED_GATE_FAIL /
A1X_V3_D0_GLOBAL_HEAD_CONTRACT_FAIL_STOP / NONE`.

The proposed A1X-global cell passed true-minus-shuffle LCB95 (`+0.007963 dB`)
and paired global-minus-local LCB95 (`+0.000890 dB`), with all structural and
safety guards passing. It failed the material gain LCB95 (`+0.013200 dB` versus
`+0.020`) and oracle-retention LCB95 (`0.080069` versus `0.25`). Therefore the
current global-head contract stops and the 432-name confirmation stage remains
untouched and unauthorized.

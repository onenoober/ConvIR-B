# A1X-v3 evidence

Status: S0 r1 `FAILED_ENGINEERING` during asset preflight; no model workload ran.

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

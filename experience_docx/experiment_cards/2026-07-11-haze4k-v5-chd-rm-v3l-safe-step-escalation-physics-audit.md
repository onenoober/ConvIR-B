# Haze4K v5 CHD-RM v3l Safe-Step Escalation And Physics Audit

Date: 2026-07-11

Status: `CLOSED_B_WEAK_STOP`

Branch: `codex/haze4k-v5-v3l-safe-step-escalation-physics-audit`

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711/`

## Route Identity

v3l is a new diagnostic continuation after v3k. It is not a canary, not a
promotion route, and not a continuation of v3j/v3k direct residual as a model
candidate.

The purpose is to test whether correction can be safely escalated beyond the
provisional `context alpha=0.125` safe step. The first gate freezes canonical
direct operators so that later risk targets cannot drift.

## A0 Gate

Fit only fixed context direct heads:

- `D_ref`: seed `3407`;
- `D_rep`: seed `3408`.

Artifacts are saved only on `convir-4090`; GitHub should receive only compact
manifests and SHA evidence.

Pass criteria:

- exact replay row identity/order;
- max PSNR delta replay difference `<= 1e-6`;
- exact severe set at `<= -0.2 dB`;
- max direct tensor replay difference `<= 1e-7`;
- stable artifact SHA.

## Forbidden

- no Haze4K locked test;
- no canary;
- no alpha/clip/energy/risk-threshold selection from v3j route-confirm;
- no confidence/router training;
- no backbone, RARM, FAM, or larger direct-head training;
- no GitHub sync of weights, checkpoints, feature tensors, or raw outputs.

## Next Stage

Only if A0 passes, authorize v3l-A1 oracle granularity audit on the frozen
operators. Physics-risk audit remains blocked until A1 shows meaningful
escalation utility beyond fixed `alpha=0.125`.

## A0 Result

A0 passed on convir-4090. Both `D_ref` and `D_rep` artifacts had stable SHA
hashes and exact double-replay agreement: row identity/order exact, max PSNR
delta diff `0`, max direct tensor diff `0`, and exact severe set at
`<= -0.2 dB`. The artifacts remain cloud-only under `cloud_only_artifacts/`.

## A1 Gate

A1 may read the frozen A0 artifacts and compute oracle step-size upper bounds at
image, block, and pixel granularity. The OOF gate compares oracle policies
against fixed `alpha=0.125` for both `D_ref` and `D_rep`; route-confirm output
is confirm-audit-only and must not be used for strategy selection. A1 cannot
authorize canary, locked test, confidence/router training, or new model search.

## A1 Result

A1 passed on OOF for both frozen operators. All oracle granularities cleared the
pre-registered meaningful-escalation gate against fixed `alpha=0.125`; even the
image-level oracle had zero severe regressions and large paired lift. This
supports the diagnosis that the current bottleneck is step-size/risk
observability, not lack of direct-correction upside.

## B Gate

B may audit available physics metadata and test privileged transmission-map risk
signals. It may not train a deployable estimator, select route-confirm
thresholds, touch locked test, or authorize canary. If privileged transmission
cannot identify direct-severe / low-optimal-alpha / wrong-or-harmful risk on OOF
for both `D_ref` and `D_rep`, physics-risk work stops.

## B Result

B stopped the route. Haze4K has `train/trans` and `test/trans`, but no
airlight/beta/depth/atmos metadata were found. Privileged transmission features
were moderately informative for low optimal alpha and wrong-or-harmful cases,
but failed the pre-registered direct-severe risk gate on both frozen operators:
best OOF direct-severe AUC was about `0.635` for `D_ref` and `0.631` for
`D_rep`, below the `0.65` threshold.

Final decision:
`V3L_B_PRIVILEGED_TRANSMISSION_RISK_WEAK_STOP_NO_PHYSICS_POLICY`.

No next stage is authorized.

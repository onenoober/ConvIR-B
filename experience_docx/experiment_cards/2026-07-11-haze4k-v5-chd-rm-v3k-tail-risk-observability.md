# Haze4K v5 CHD-RM v3k Tail-Risk Observability

Date: 2026-07-11

Route ID: `haze4k_v5_chd_rm_v3k_tail_risk_observability_20260711`

Status: provisional diagnostic complete; no canary authorized.

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3k_tail_risk_observability_20260711/`

## Decision

`V3K_PROVISIONAL_MICRO_ALPHA_SAFE_STEP_SUPPORTED_NO_CANARY_NO_NEW_SEALED_SPLIT`

v3k supports the diagnosis that the direct residual bottleneck is not simply
missing tail-risk information. The sharper bottleneck is A0-relative correction
advantage sign plus safe-step observability. Full direct residual steps mix
wrong-direction cases with harmful overshoot, while small fixed alpha replay is
tail-safer in grouped OOF and historical open holdout.

This route does not authorize canary, promotion, or locked-test access because:

- strict reconstruction of v3j-B direct-head replay failed;
- subsequent v3k outputs are labeled new-replicate/provisional evidence;
- `val_inner` is historical open holdout, not a new sealed train-derived split.

## Key Results

Strict reconstruction gate:

- OOF row count/identity matched, but max direct PSNR delta difference was
  `0.2168 dB`; severe sets differed.
- CONFIRM row count/identity matched, but max direct PSNR delta difference was
  `0.3162 dB`; severe sets differed.

Direction diagnosis:

- OOF context: wrong-direction `328/1200`, harmful overshoot `264/1200`,
  alpha* median `0.519`, severe `279`.
- OOF linear: wrong-direction `342/1200`, harmful overshoot `326/1200`,
  alpha* median `0.344`, severe `233`.
- Confirm context: wrong-direction `155/600`, harmful overshoot `135/600`,
  alpha* median `0.552`, severe `125`.
- Confirm linear: wrong-direction `170/600`, harmful overshoot `145/600`,
  alpha* median `0.458`, severe `116`.

OOF fixed-alpha replay:

- `alpha=0.125 context`: mean `+0.0298 dB`, p10 `-0.0279 dB`, severe `0`,
  paired mean vs hard `+0.0129 dB`, CI95 `[+0.0067,+0.0195]`.
- `alpha=0.25 context`: mean `+0.0536 dB`, p10 `-0.0636 dB`, severe `15`,
  paired mean vs hard `+0.0366 dB`, CI95 `[+0.0290,+0.0444]`.
- `alpha>=0.375` starts failing tail gates.

Historical open `val_inner` holdout:

- `alpha=0.125 context`: mean `+0.0287 dB`, p10 `-0.0324 dB`, severe `0`,
  paired mean vs hard `+0.0159 dB`, CI95 `[+0.0062,+0.0255]`.
- `alpha=0.25 context`: mean `+0.0512 dB`, p10 `-0.0745 dB`, severe `10`,
  paired mean vs hard `+0.0384 dB`, CI95 `[+0.0275,+0.0493]`.
- Seed `3408` context-only stability repeated the same conclusion.

## Next Gate

Do not launch canary. A real promotion step requires a genuinely new sealed
train-derived or external validation split plus deterministic saved head
artifacts, then validation of the OOF-selected micro-alpha strategy before any
canary or locked-test access.

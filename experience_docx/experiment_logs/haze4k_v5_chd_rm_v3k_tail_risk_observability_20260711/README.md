# v3k Tail-Risk Observability

Route card:
`experience_docx/experiment_cards/2026-07-11-haze4k-v5-chd-rm-v3k-tail-risk-observability.md`

Central index:
`experience_docx/EXPERIMENT_INDEX.md`

Decision:
`V3K_PROVISIONAL_MICRO_ALPHA_SAFE_STEP_SUPPORTED_NO_CANARY_NO_NEW_SEALED_SPLIT`

## Scope

v3k reconstructs v3j-B direct bounded residual evidence, then diagnoses
A0-relative correction sign, harmful overshoot, block-local risk, and fixed
micro-alpha replay. It does not touch locked test, save probe weights, save raw
feature tensors, authorize canary, or authorize training continuation.

## Primary Evidence

- `v3k_closeout.md` and `v3k_closeout.json`: compact final decision and metrics.
- `v3k_reconstruction_compare.json`: strict v3j-B reconstruction comparison.
- `v3k_direct_direction_summary.csv`: alpha* direction and overshoot summary.
- `v3k_block_risk_summary.csv`: spatial block-risk concentration summary.
- `v3k_oof_fixed_alpha_policy_summary.csv`: grouped OOF fixed-alpha replay.
- `v3k_oof_fixed_alpha_bootstrap_vs_hard.csv`: OOF bootstrap against hard D7c.
- `v3k_c_open_holdout_fixed_alpha_summary.json`: historical `val_inner` open
  holdout diagnostic.
- `v3k_d_context_seed3408_open_holdout_stability/`: context-only seed stability
  repeat on historical `val_inner`.

Raw per-image replay rows and block rows remain cloud/runtime artifacts and are
not required for the compact GitHub reading path.

## Key Results

Strict reconstruction failed: row identity/count matched v3j-B, but direct-head
PSNR deltas and severe sets differed. All subsequent v3k outputs are therefore
new-replicate/provisional evidence.

Full direct residual bottleneck:

- OOF context wrong-direction `328/1200`, harmful overshoot `264/1200`.
- OOF linear wrong-direction `342/1200`, harmful overshoot `326/1200`.
- Confirm context wrong-direction `155/600`, harmful overshoot `135/600`.
- Confirm linear wrong-direction `170/600`, harmful overshoot `145/600`.

OOF fixed-alpha replay:

- `alpha=0.125 context`: mean `+0.0298 dB`, p10 `-0.0279 dB`, severe `0`,
  paired mean vs hard `+0.0129 dB`, CI95 `[+0.0067,+0.0195]`.
- `alpha=0.25 context`: mean `+0.0536 dB`, p10 `-0.0636 dB`, severe `15`,
  paired mean vs hard `+0.0366 dB`, CI95 `[+0.0290,+0.0444]`.

Historical open `val_inner` holdout:

- `alpha=0.125 context`: mean `+0.0287 dB`, p10 `-0.0324 dB`, severe `0`,
  paired mean vs hard `+0.0159 dB`, CI95 `[+0.0062,+0.0255]`.
- `alpha=0.25 context`: mean `+0.0512 dB`, p10 `-0.0745 dB`, severe `10`,
  paired mean vs hard `+0.0384 dB`, CI95 `[+0.0275,+0.0493]`.
- Seed `3408` context-only repeat preserved both conclusions.

## Gate

No canary is authorized. A real promotion step requires a genuinely new sealed
train-derived or external validation split and deterministic saved head
artifacts before any canary or locked-test access.

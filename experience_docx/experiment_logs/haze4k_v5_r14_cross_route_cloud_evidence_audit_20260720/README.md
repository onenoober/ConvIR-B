# Haze4K v5 R14 Cross-Route Cloud Evidence Audit

Date: 2026-07-20

Status: `COMPLETED_READ_ONLY_AUDIT`. The authoritative model terminal remains
R13 `COMPLETED_GATE_FAIL / R13_A0_RELATIVE_CONTEXT_UTILITY_FAIL_STOP / NONE`.
This audit does not replace or amend any R5-R13 terminal.

## Identity And Access

- Cloud host/root: `convir-4090:/sda/home/wangyuxin/ConvIR-B/`.
- Successful audit source: `fe4005f00128346872bccd2107a16973e52a417b`.
- Successful output: `runs/haze4k_v5_r14_cross_route_cloud_evidence_audit_20260720/r14-cross-route-audit-r2`.
- r1 failed only while serializing a NumPy integer into strict JSON. It was not
  overwritten; r2 changed only JSON scalar conversion and used a fresh checkout
  and output directory. r1 is engineering history, not scientific evidence.
- No training, checkpoint selection/loading, candidate generation or protected
  role access occurred. Confirmation, canary and locked test remained sealed.

## Formal Reproduction

- R3 manifest SHA-256 values and the R5/R11 preregistered raw-input hashes match
  their official identities. Every fixed raw-table key, fold and row-count check
  passes. R5/R10/R11/R12/R13 status ledgers each contain one complete terminal
  workload event at the expected count.
- R10, R11 and R12 decisive 4,000-draw bootstrap points and intervals reproduce
  exactly from cloud raw rows; maximum absolute difference is `0.0`.
- Every R10-R13 closeout-bound compact Git file matches its recorded SHA-256.
- R13 did not persist a row-prediction/action-map artifact. Its status is
  `COMPACT_HASH_VERIFIED_ONLY`, not raw-row reproduction.

## Main Forensic Findings

- R5 whole-image oracle-to-policy gap is `0.124540 dB`: frozen 20% coverage
  accounts for `0.057708`, coverage ranking for `0.046351`, and chosen action
  direction for `0.020481 dB`. Only 31/78 selected names overlap the true
  fixed-budget set; 19/78 selected actions use the wrong sign/direction.
- With the exact privileged R10 regional action budget, R11 predicts the exact
  tile action only `51.1271%` of the time. Accuracy is still `50.4706%` among
  14,662 tiles whose true best-second margin is at least `0.02 dB`. Prediction
  versus actual worst utility has Pearson `0.2914`; eligibility AUROC is
  `0.5785`. This places the dominant failure before policy coverage.
- Label precision is not the main explanation: float32 versus float64 produces
  zero action flips; only 31/49,152 action rows lie within `1e-4 dB` of the
  `+0.005 dB` gate. But 33.23% of tiles have best-second margin at most
  `0.005 dB`, and oracle maps have mean four-neighbor boundary disagreement
  `0.3488`, leaving region/label semantics a material upstream concern.
- R12 primary AUROC is `0.7221`, yet fixed top-20% severe capture is only
  `0.4437`, retained prevalence ratio `0.6954`, row ECE `0.2134`, and
  score-versus-damage Spearman `0.1968`. Post-hoc action strata differ sharply
  (AUROC `0.6170` positive versus `0.8121` negative).
- Historical NH-HAZE v2.7 fixed-Haze4K blend evidence has mean dPSNR
  `-0.0182 dB`, 13/55 severe losses, positive mean dSSIM, and 50.9% PSNR/SSIM
  sign disagreement. It is a different action family, so this is only
  directional evidence for domain/target mismatch, never R10-R13 validation.

## Decision

The exact R5-R13 selector/critic/context paradigm remains closed. H1
(insufficient observable candidate-specific signed utility) is the strongest
in-domain mechanism; H2 (decision/risk conversion) remains secondary. H3
(target/region/action misspecification) is strengthened as an upstream strategic
bottleneck, while its narrow numerical-precision version is weakened. H4
(real-domain shift) remains unresolved and directionally concerning. The total
research objective requires strategic reconstruction; this audit authorizes only
read-only evidence sync and no protected-data access.

Cloud-only artifacts remain the original R3/R5/R10-R12 row tables, cache units
and all raw runtime sources. Git receives only this README, compact JSON
summaries, status and a small runtime receipt.

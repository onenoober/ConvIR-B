# Haze4K v4.7 DCFSB Adapter4 Candidate-Lock Validation

Date: 2026-07-08

Branch: `codex/haze4k-v4-7-dcfsb-candidate-validation`

Status: completed locked-test confirmation fail; no promotion.

Route identity: continuation/audit of the v4.6 DCFSB-bottleneck `adapter4` candidate, not a new architecture route and not an A3/v4.5 continuation.

Parent/source: `github/codex/haze4k-v4-6-dcfsb-bottleneck-independent` at closeout commit `1277b61788fd2969e2bfdac9455a1a317db61f48`.

Forbidden flows:

- Do not continue A3.
- Do not expand v4.5 SDC-Lite or connect R to skip/restoration outputs.
- Do not alter model structure, loss, train split, seed, or checkpoint selection during candidate-lock validation.
- Do not run additional locked-test commands for this candidate.
- Do not tune model structure, checkpoint, epoch, thresholds, or variants from locked-test results.

Candidate under validation: fixed v4.6 `adapter4` checkpoint from cloud path `Dehazing/ITS/results/ConvIR-Haze4K-v4A6-DCFSB-Bottleneck-adapter4-notest-seed3407-20260708/Training-Results/Final.pkl`.

## Internal Candidate-Lock Gate

Status: pass.

- mean dPSNR `0.044404`
- positive ratio `0.625000`
- p5 dPSNR `-0.216141`
- mean dHighL1 `-0.0000005702`
- bootstrap 95% CI low `0.024481`
- sign-test one-sided p `3.802649e-05`
- systematic failure flags `0`

## Locked-Test Confirmation

Status: fail.

Exactly one metric-producing locked-test command was run for the fixed checkpoint. Prediction images were not saved. A prior post-v4.7 directory-count preflight enumerated the locked test split but produced no metric; see `post_v47_locked_test_policy_note.md`.

- A0 mean PSNR `34.145502`
- candidate mean PSNR `34.149328`
- mean dPSNR `0.003826`
- median dPSNR `-0.003686`
- positive ratio `0.484000`
- p5 dPSNR `-0.210819`
- mean dSSIM `0.00002084`

Gate failure reason: locked-test positive ratio `0.484000` was below the prewritten `0.50` threshold, despite a tiny positive mean dPSNR.

## Decision

Do not promote adapter4. Treat v4.6/v4.7 as useful internal evidence that bottleneck frequency calibration can help on train-derived holdout, but not as a reliable deployable or paper-result improvement. Future work must not tune from locked-test results. If continuing this family, use train-derived K-fold/tail-safe validation or a separately justified R-only calibration probe.

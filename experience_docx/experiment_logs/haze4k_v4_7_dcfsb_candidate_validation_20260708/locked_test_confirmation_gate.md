# Haze4K v4.7 Locked-Test Confirmation Gate

Date: 2026-07-08

Status: authorized one metric-producing locked-test command.

Precondition: v4.7 train-derived candidate-lock validation passed on internal_holdout256 with mean dPSNR `+0.044404`, positive ratio `0.625000`, p5 `-0.216141`, bootstrap CI low `+0.024481`, sign-test p `3.802649e-05`, and zero systematic failure flags.

Operational note: after the v4.7 internal audit completed, an evaluation-entry preflight listed Haze4K test directories and counts. No image was opened and no metric was produced, but this is recorded as a locked-test enumeration under repository policy in `post_v47_locked_test_policy_note.md`.

Authorized command: run exactly one paired A0-vs-adapter4 locked-test evaluation script, `run_v47_locked_test_once.sh`.

Fixed checkpoints:

- A0: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- Candidate: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent/Dehazing/ITS/results/ConvIR-Haze4K-v4A6-DCFSB-Bottleneck-adapter4-notest-seed3407-20260708/Training-Results/Final.pkl`

Output root: `experience_docx/experiment_logs/haze4k_v4_7_dcfsb_candidate_validation_20260708/locked_test_once/`.

Constraints:

- Do not save prediction images.
- Do not run a second locked-test metric command for this candidate.
- Do not use the locked-test result to tune model structure, checkpoint, epoch, or thresholds.
- Record paired A0/candidate metrics in one pass to preserve metric alignment.

Confirmation gate:

- mean locked-test dPSNR > `0`
- locked-test positive ratio >= `0.50`
- locked-test p5 dPSNR >= `-0.50`
- mean dSSIM >= `-0.0001`
- saved prediction images == `false`
- metric-producing locked-test command count == `1`

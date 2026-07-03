# v2.17 R1 WLDB-A Postmortem Decision

Decision: `R1_CLOSE_WLDB_A_KEEP_NOPOST_LOWBAND_OPEN`

Interpretation:

- WLDB-A remains closed as a concrete form; do not expand seeds, epochs, hidden width, or locked-test use.
- The broader NoPost lowband direction remains open pending R2 capacity-ladder evidence.
- model_5 severe count: `67` / `480`.
- model_5 mean/hard/easy dPSNR: `0.081889` / `0.105887` / `0.020994`.
- model_10 mean dPSNR: `0.031877`.
- model_5 vs model_10 severe-set Jaccard: `0.329114`.
- training history action-budget term all zero: `True`.

Answers to the five R1 questions:

1. Tail concentration is recorded in `v217_r1_tail_case_manifest.csv` using hard/easy/strong flags.
2. Checkpoint severe overlap is recorded in `v217_r1_severe_overlap_by_checkpoint.csv`.
3. The pareto table shows the mean/hard gain shrinks as later checkpoints reduce severe losses.
4. The v2.16 action-budget hinge stayed inactive in the logged train objective.
5. Loss-vs-identity and action statistics are recorded from the available checkpoints; use them for R3 objective audit.

Locked Haze4K test remains untouched. This is an audit only, not training.

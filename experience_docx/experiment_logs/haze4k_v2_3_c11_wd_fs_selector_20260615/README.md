# Haze4K v2.3 C11 WD0375-FS050 Selector Evidence

Decision: `C11E_SEALED_SELECTOR_PASS_READY_FOR_LOCKED_ONE_SHOT_REVIEW`

This route uses only train-derived C8/C9 per-image tables. Locked Haze4K output is not read, and no distillation or new expert acquisition is performed.

## Key Metrics

- Best deployable selector: `nested_oof_selected`
- OOF mean/hard/easy: `2.812140` / `3.567257` / `1.868307`
- OOF positive/severe: `0.982222` / `8.00/600`
- Sealed selector config: `feature_set=residual_consensus;kind=pairwise;lambda=0.5;severe_penalty=0.5;threshold=-0.15`
- Sealed train-derived mean/hard/easy: `2.828078` / `3.548762` / `1.953362`
- Sealed train-derived positive/severe: `0.985000` / `6.00/600`
- Sealed action usage: WD0375 `0.486667`, FS050 `0.513333`, A0 `0`

## Output Map

- C11-0: provenance, no-locked, source manifest, metric parity.
- C11-A: WD0375/FS050/A0 oracle decomposition and selected-negative report.
- C11-B: nested OOF low-capacity selector screen and ablations.
- C11-C: group-min shifted validation.
- C11-D: formal 5x3 replay.
- C11-E: sealed selector package for future locked replay review.

Locked Haze4K was not run in C11/C11-E. Any future locked replay must use
`v23_c11e_sealed_selector.json` exactly and record the locked output as evidence
only.

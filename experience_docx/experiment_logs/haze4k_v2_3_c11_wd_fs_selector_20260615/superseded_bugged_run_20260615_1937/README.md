# Haze4K v2.3 C11 WD0375-FS050 Selector Evidence

Decision: `C11_FAIL_SELECTOR_NOT_READY_LOCKED_BLOCKED`

This route uses only train-derived C8/C9 per-image tables. Locked Haze4K output is not read, and no distillation or new expert acquisition is performed.

## Key Metrics

- Best deployable selector: `nested_utility`
- OOF mean/hard/easy: `2.833738` / `3.666722` / `1.956188`
- OOF positive/severe: `0.985207` / `8.88/600`

## Output Map

- C11-0: provenance, no-locked, source manifest, metric parity.
- C11-A: WD0375/FS050/A0 oracle decomposition and selected-negative report.
- C11-B: nested OOF low-capacity selector screen and ablations.
- C11-C: group-min shifted validation.
- C11-D: formal replay only if C11-C passes; otherwise explicit skipped decision.

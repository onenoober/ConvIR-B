# Haze4K v2.3 C11 WD0375-FS050 Selector Evidence

Decision: `LOCKED_C11_SELECTOR_ONE_SHOT_RECORDED_DO_NOT_PROMOTE_OVER_WD0375`

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

## Locked One-Shot Closeout

The sealed selector was replayed once on locked Haze4K after C11-E. The replay
used `v23_c11e_sealed_selector.json` exactly and is evidence only:

- selector locked mean/hard/easy: `+1.449078 / +1.558683 / +1.248566 dB`;
- selector locked positive/severe: `0.896000` / `48.60/600`;
- action usage: WD0375 `0.386`, FS050 `0.614`, A0 `0`.

Compared with the already locked-pass fixed WD0375 baseline
(`+1.442090 / +1.529767 / +1.182529 dB`, positive `0.938000`, severe
`25.80/600`), C11 improved mean/hard/easy only slightly but materially worsened
positive ratio and severe tail risk. Therefore C11 is a useful selector
feasibility and oracle-to-locked-gap result, but it should not replace WD0375 as
the default locked-pass strong baseline and should not be used as a distillation
teacher without a separate risk-repair route.

Locked output must not tune alpha, features, checkpoints, profiles, actions,
experts, or distillation targets.

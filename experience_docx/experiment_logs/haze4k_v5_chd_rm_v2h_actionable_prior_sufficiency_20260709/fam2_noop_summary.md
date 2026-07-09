# v2h-C/D Closeout

Status: `COMPLETED_WITH_D_PREFLIGHT_BLOCKED`

Decision label: `V2H_ABC_PASS_D_BLOCKED_CREATE_SEPARATE_NOOP_ARCH_BRANCH`

## C Result

v2h-C passed fold calibration stability:

- D7c calibrated action recall mean/min: `0.576335` / `0.556955`
- D7c low-adjacent recall mean: `0.170063`
- D7c negative false mean/max: `0.003403` / `0.003996`
- D7c selected coverage std: `0.010785`
- Density-matched negative false mean/max: `0.049636` / `0.063885`

## D Result

v2h-D did not reach numerical no-op equivalence because the branch preflight blocked the architecture variant:

`Official ConvIR-B anchor only supports fam_mode='original'. Create a route branch for architecture variants.`

This is an architecture-boundary blocker, not a negative numerical equivalence result. v2h remains a diagnostic prior route and should not be mutated to carry FAM2/RARM structure.

## Decision

D7c prior sufficiency is supported by A/B/C. The next step is a separate no-op architecture branch from `github/codex/haze4k-official-arch-anchor`, not RARM/training inside v2h. Locked test, D2/F5/v3, RARM connection/training, adapter training, canary expansion, and architecture mutation inside v2h remain blocked.

# FAM2 Architecture Patch Summary

Status: `PLANNED`

The patch starts from `github/codex/haze4k-official-arch-anchor` and changes
only the FAM module plumbing needed for a FAM2 no-op shell.

## Expected Code Change

- `FAM(mode='original')`: identical behavior to anchor.
- `FAM(mode='modres')`: after the original `merge`, applies
  `fused * (1 + gamma) + beta`.
- `gamma,beta`: produced by `FAM2.modulator`, a zero-initialized `1x1`
  convolution from the FAM2 SCM feature tensor.
- `FAM1`: forced to original mode when `fam_mode='fam2_modres'`.
- `build_net`: accepts only `original` and `fam2_modres` on this branch.

## Expected Parameter Delta

FAM2 has `64` channels at this insertion point. The modulator maps `64 -> 128`
with a `1x1` convolution:

```text
64 * 128 weights + 128 bias = 8320 parameters
```

No official ConvIR-B parameter tensor shape should change.

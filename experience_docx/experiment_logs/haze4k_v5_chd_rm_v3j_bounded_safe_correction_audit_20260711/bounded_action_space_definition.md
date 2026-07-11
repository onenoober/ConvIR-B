# Bounded Action-Space Definition

v3j-A defines a bounded output residual actuator:

```text
Y_projected = A0 + M_D7c_output * clip(P(Y_teacher - A0), -B, B)
```

Where:

- `A0` is the official Haze4K ConvIR-B baseline output.
- `Y_teacher` is replay output from a fixed privileged teacher.
- Primary teacher is `CC_MIN4_FROM_OPEN_TOP_0.5`.
- Ceiling teacher is `ALPHA_SECANT_Q3`.
- `B` is per-channel p99 absolute primary-teacher output residual on
  `v3j_controller_calib`.
- `P` is one of `full_clip`, `half_bilinear`, `quarter_bilinear`,
  `half_smooth3`.
- `M_D7c_output` is the D7c action mask nearest-upsized to output resolution.

Minimum pass: paired mean delta vs same-split hard D7c has CI95 low > 0, p10 is
not below hard D7c and not below the fixed v3i hard p10 line, and severe
regressions do not exceed hard D7c.

Strong pass: minimum pass plus mean > `+0.033065 dB`, fewer severe regressions
than hard D7c, and mean >= `+0.112512 dB`.

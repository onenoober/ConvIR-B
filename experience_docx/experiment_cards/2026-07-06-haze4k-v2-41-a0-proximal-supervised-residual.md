# Haze4K v2.41 A0-Proximal Supervised Residual

Date: 2026-07-06

Branch:
`codex/haze4k-v2-41-a0-proximal-supervised-residual`

Starting source:
`github/codex/haze4k-official-arch-anchor` at `3b4da354`.

Route identity: new model-structure route authorized only for Stage-0
design/preflight by v2.40 manual review.

## Primary Question

Can a zero-init bounded residual head inside ConvIR-B provide a compliant
starting point for later A0-proximal, GT-risk-controlled supervised improvement
without reopening teacher selector or no-selector alpha routes?

## Architecture Contract

Model form:

```text
Y = A0 + beta * tanh(R_theta(decoder_feature, A0, x))
```

Declared new prefixes:

```text
allowed_new_prefixes = ("A0PROX_",)
```

Initialization:

- official ConvIR-B keys must load with exact shape match;
- `A0PROX_head` first convolution uses Kaiming initialization;
- `A0PROX_head` final convolution has zero weights and zero bias;
- `A0PROX_beta` is a fixed buffer with value `0.05`;
- Stage-0 must prove exact no-op vs official ConvIR-B.

## Not Allowed

- No WDMamba/ConvIR-L alpha continuation.
- No target-only selector threshold/feature tuning.
- No M0 bridge/generator or P5 projection.
- No canary80.
- No locked test.
- No canary training until Stage-0 preflight passes and a canary gate is written.

## Stage-0 Metric Contract

P0 passes only if:

- checkpoint sha256 is
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`;
- official keys partial-load exactly;
- missing keys are only `A0PROX_*`;
- synthetic forward outputs are finite;
- max absolute difference vs official ConvIR-B is exactly `0.0`;
- forbidden postprocess symbol hits are `0`;
- locked test remains untouched.

## Result

Status: `PLANNED`.

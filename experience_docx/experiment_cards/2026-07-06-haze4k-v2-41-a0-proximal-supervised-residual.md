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

Status: `COMPLETED_GATE_FAIL`.

P0 Stage-0 preflight passed on `convir-4090` at route commit `7c27b93`.
The official checkpoint sha256 matched
`6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`,
strict partial load accepted only `A0PROX_*` missing keys, finite synthetic
forwards passed for `64/128/256`, forbidden postprocess symbol hits were `0`,
locked test was untouched, and max absolute difference vs official ConvIR-B
was exactly `0.0`.

Two earlier cloud preflight attempts were engineering-invalid, not scientific
failures: synthetic size `128` was invalid for the route's reflect-padding
preflight, and the first beta gate compared the float buffer without tolerance.
Both were archived on cloud before the final P0 rerun.

## Canary32 OOF Gate

P0 closeout authorizes only canary32. Canary80 and locked test remain blocked
unless the canary32 closeout explicitly passes and writes the next gate.

Canary32 uses only train-derived Haze4K images from
`/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K/train`, with five
deterministic folds. Each fold trains on `32` train-derived images and
evaluates on a disjoint `32` train-derived held-out images. It freezes all
official ConvIR-B parameters and trains only `A0PROX_*`; no teacher target,
selector, canary80, or locked test is used.

The canary32 gate passes only if:

- global mean delta is at least `+0.15 dB`;
- global hard bottom-25% delta is at least `+0.30 dB`;
- global easy top-25% delta is at least `+0.00 dB`;
- global p05 delta is at least `-0.01 dB`;
- global CVaR5 delta is at least `-0.02 dB`;
- severe regressions (`delta <= -0.20 dB`) are `0`;
- strong-reference regressions (`easy/top-25% delta < -0.01 dB`) are `0`;
- at least `4/5` folds pass the same quality/tail gate;
- easy residual energy is at most `0.50` of hard residual energy;
- hinge violation rate does not worsen from epoch `1` to final epoch.

Canary32 OOF completed on `convir-4090` at route commit `32e7791`. It was a
cloud-only train-derived run: `5` folds, `32` training images per fold,
disjoint `32` held-out train-derived images per fold, frozen official ConvIR-B
parameters, and `11843` trainable `A0PROX_*` parameters. Locked test remained
untouched.

Decision:
`CANARY32_OOF_GATE_FAIL_LOCK_CANARY80_LOCKED_TEST`.

Global canary32 metrics over `160` held-out train-derived images:

| Metric | Value | Gate |
| --- | ---: | ---: |
| mean delta | `-0.0277 dB` | `>= +0.15 dB` |
| hard bottom-25% delta | `+0.0742 dB` | `>= +0.30 dB` |
| easy top-25% delta | `-0.0724 dB` | `>= +0.00 dB` |
| p05 delta | `-0.3981 dB` | `>= -0.01 dB` |
| CVaR5 delta | `-0.5972 dB` | `>= -0.02 dB` |
| severe regressions | `27` | `0` |
| strong-reference regressions | `25` | `0` |
| fold pass count | `0/5` | `>= 4/5` |

Mechanism diagnostics were not the blocker: easy residual energy ratio was
`0.2871` against the `<=0.50` gate, and hinge violation rate decreased from
`0.6625` to `0.58125`. The route failed because the residual head did not
produce tail-safe quality improvement under the written OOF contract.

Canary80 and locked test remain blocked.

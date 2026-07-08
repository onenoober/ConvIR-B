# CHD-RM Route Scope

Date: 2026-07-08

## Fixed Scope

```text
连续雾浓度感知的区域自适应残差调制与低雾区域保护去雾方法研究
```

The route answers whether continuous haze-density response and regional
restoration-need response can safely drive adaptive residual restoration on
ConvIR-B for Haze4K.

## Included

- single-image dehazing;
- ConvIR-B backbone;
- Haze4K train-derived validation and final Haze4K confirmation;
- continuous haze-density response `H_density`;
- regional restoration-need response `R_need`;
- adaptive residual branch `R_adapt`;
- bounded modulation coefficient `gamma`;
- low-haze protection;
- multi-scale haze modulation only after the single-scale and low-haze gates.

## Excluded

- video dehazing;
- multi-image dehazing;
- NH-HAZE, Dense-Haze, or DNH-HAZE as main experiments;
- real unpaired/no-reference training;
- backbone replacement;
- diffusion models;
- multi-expert large-model routes;
- independent color, luminance, texture, or structure modeling;
- color-correction, texture-enhancement, or structure-preservation modules;
- Lab, luminance, gradient, or texture as core training targets.

## Non-Drift Rule

Later architecture changes may change module placement, parameterization,
initialization, or training schedule. They may not change the route objective
away from:

```text
H_density + R_need -> gamma -> adaptive residual modulation with low-haze protection.
```

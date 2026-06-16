# Haze4K v2.9 NH-HAZE Official-Test Alpha Grid Decision

Decision: `V29_NHHAZE_OFFICIAL_TEST_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`

v2.9 is the clean rerun requested after the v2.8 NH-HAZE reproduction audit. It
uses only official-style NH-HAZE test ids `51-55`, staged at
`/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE-official-test-51-55`, and rejects
the earlier all-55 mixed train/val/test aggregate as a usable evidence record.

## Protocol

- Runtime host: `convir-4090`.
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Pair ids: `51 52 53 54 55`.
- Pair count: `5` hazy/GT pairs.
- A0 checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/nhhaze-base.pkl`.
- A0 construction: `build_net("base", "NHR", "original")`.
- WDMamba checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/NH_20.83.pth`.
- WDMamba construction: `WaveMamba` with `DENet(3, 4)`.
- Alpha grid: `{0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.0}`.
- Primary inherited alpha: `0.375`.
- Haze4K locked test touched: `false`.
- NH-HAZE alpha tuning: `false`.

The evaluation script was reused from the v2.8 audit workspace, so the raw
stdout log retains a legacy `V28_NHHAZE_OFFICIAL_AGGREGATE_OK` marker. The
evidence directory normalizes this rerun as v2.9 through `v29_decision.md`,
`v29_final_audit.json`, and the v2.9 summary/manifest files.

## Absolute Reproduction

```text
A0_NH / ConvIR-B: 20.663593 PSNR, 0.796806 SSIM
WDMamba_NH:       20.830742 PSNR, 0.818217 SSIM
```

This resolves the previous anomaly. The official-test A0 result aligns with the
ConvIR-B README NH-HAZE base number `20.66/0.802`, and the WDMamba result aligns
with the NH checkpoint name `NH_20.83.pth`. The earlier all-55 A0 result near
`26.10/0.9296` was caused by evaluating train, validation, and test ids together.

## Inherited Alpha Diagnostic Row

- count: `5`
- alpha: `0.375`
- mean/hard/easy dPSNR: `+0.515796` / `+0.078772` / `+0.732107`
- dSSIM: `+0.02203434`
- positive/nonnegative: `1.000000` / `1.000000`
- severe: `0/5` (`0.00/600`)
- worst dPSNR: `+0.078772`

## WDMamba Endpoint

- alpha `1.0` mean/hard/easy dPSNR:
  `+0.167149` / `-1.103455` / `+1.017271`
- dSSIM: `+0.02141021`
- positive/nonnegative: `0.600000` / `0.600000`
- severe: `2/5` (`240.00/600`)
- worst dPSNR: `-1.103455`

## Interpretation

Clean official-test rerun supports the residual-shrinkage safety pattern on
NH-HAZE-specific weights: medium alpha improves mean PSNR and reduces tail risk
relative to the full WDMamba endpoint. However, this is not an NH-HAZE alpha
selection result. It is a five-image official-test diagnostic for the
pre-existing Haze4K inherited alpha, and any NH-HAZE-specific alpha claim needs
a separate validation or OOF protocol before test reporting.

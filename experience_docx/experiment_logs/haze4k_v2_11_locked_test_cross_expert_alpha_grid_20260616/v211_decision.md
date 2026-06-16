# Haze4K v2.11 Locked Cross-Expert Alpha Grid

Decision: `V211_HAZE4K_LOCKED_CROSS_EXPERT_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`

This route evaluates FSNet+UDP and MB-TaylorFormerV2-L residual-shrinkage alpha grids on the Haze4K locked test split (`1000` images). It is a diagnostic audit only and must not be used to select or retune alpha.

Metric protocol: alpha candidates use the v2.2 locked one-shot compatible A0 32-pad PSNR/SSIM convention, so the curves are directly comparable with the WDMamba v2.10 locked alpha grid. Expert endpoint/direct metrics are also recorded separately for official-standard reproduction context.

Loader provenance:

- FSNet+UDP: official UDPNet `Dehazing/ITS/models/FSNet_UDPNet.py`, with documented `num_heads=1 -> 2` builder patch needed to strict-load `FSNet_UDPNet_haze4k.ckpt` OCAB bias tables; depth input uses official-style `depth2l/*.png` files generated from DepthAnything V2 raw predictions by per-image min-max normalization, then read through PIL `L` mode like the official dataloader.
- MB-TaylorFormerV2-L: official `Dehazing/Options/MB-TaylorFormerV2-L.yml`, `HAZE4K-L.pth`, `strict=False` matching official `Dehazing/test.py`; factor-8 reflect padding for model inference.

## Endpoint Reproduction Metrics

| expert | A0 PSNR | A0 SSIM grid32 | endpoint PSNR | endpoint SSIM | grid32 SSIM | direct SSIM | official reference | mean dPSNR | hard dPSNR | easy dPSNR | positive | severe/600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| FSNet+UDP | 34.145502 | 0.989619 | 35.274720 | 0.990780 | 0.990983 | 0.989769 | 35.31 / 0.99 (UDPNet README Table 2) | +1.129218 | +1.569141 | +0.694019 | 0.692000 | 165.00 |
| MB-TaylorFormerV2-L | 34.145502 | 0.989619 | 34.932525 | 0.990711 | 0.990901 | 0.989741 | not clearly reported for Haze4K V2-L in checked repo files | +0.787023 | +3.049596 | -1.778787 | 0.580000 | 238.80 |

## Alpha Grid Absolute Metrics

| expert | alpha | label | PSNR | SSIM grid32 | mean dPSNR | hard dPSNR | easy dPSNR | dSSIM grid32 | positive | severe/600 | worst dPSNR |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FSNet+UDP | 0.000 | A0 / ConvIR-B | 34.145502 | 0.989619 | +0.000000 | +0.000000 | +0.000000 | +0.00000000 | 0.000000 | 0.00 | +0.000000 |
| FSNet+UDP | 0.125 | FSNet+UDP alpha=0.125 | 34.489578 | 0.990209 | +0.344076 | +0.267909 | +0.420864 | +0.00059040 | 0.892000 | 25.80 | -1.243401 |
| FSNet+UDP | 0.250 | FSNet+UDP alpha=0.250 | 34.796454 | 0.990688 | +0.650952 | +0.527762 | +0.776468 | +0.00106909 | 0.875000 | 45.60 | -2.565327 |
| FSNet+UDP | 0.375 | FSNet+UDP alpha=0.375 | 35.055160 | 0.991052 | +0.909658 | +0.774331 | +1.047503 | +0.00143315 | 0.858000 | 63.00 | -3.840307 |
| FSNet+UDP | 0.500 | FSNet+UDP alpha=0.500 | 35.253876 | 0.991298 | +1.108374 | +1.000030 | +1.215762 | +0.00167884 | 0.834000 | 79.80 | -5.023464 |
| FSNet+UDP | 0.750 | FSNet+UDP alpha=0.750 | 35.430463 | 0.991415 | +1.284961 | +1.364635 | +1.193786 | +0.00179556 | 0.777000 | 117.60 | -7.094852 |
| FSNet+UDP | 1.000 | FSNet+UDP full | 35.274720 | 0.990983 | +1.129218 | +1.569141 | +0.694019 | +0.00136369 | 0.692000 | 165.00 | -8.827869 |
| MB-TaylorFormerV2-L | 0.000 | A0 / ConvIR-B | 34.145502 | 0.989619 | +0.000000 | +0.000000 | +0.000000 | +0.00000000 | 0.000000 | 0.00 | +0.000000 |
| MB-TaylorFormerV2-L | 0.125 | MB-TaylorFormerV2-L alpha=0.125 | 34.550039 | 0.990348 | +0.404537 | +0.443969 | +0.335874 | +0.00072846 | 0.904000 | 24.60 | -1.996803 |
| MB-TaylorFormerV2-L | 0.250 | MB-TaylorFormerV2-L alpha=0.250 | 34.883635 | 0.990908 | +0.738134 | +0.894108 | +0.496053 | +0.00128884 | 0.871000 | 49.20 | -4.128674 |
| MB-TaylorFormerV2-L | 0.375 | MB-TaylorFormerV2-L alpha=0.375 | 35.133079 | 0.991304 | +0.987577 | +1.345382 | +0.465108 | +0.00168541 | 0.827000 | 80.40 | -6.032616 |
| MB-TaylorFormerV2-L | 0.500 | MB-TaylorFormerV2-L alpha=0.500 | 35.290567 | 0.991541 | +1.145065 | +1.787917 | +0.255119 | +0.00192160 | 0.776000 | 115.20 | -7.675056 |
| MB-TaylorFormerV2-L | 0.750 | MB-TaylorFormerV2-L alpha=0.750 | 35.311925 | 0.991539 | +1.166423 | +2.580944 | -0.591450 | +0.00192030 | 0.681000 | 172.80 | -10.334991 |
| MB-TaylorFormerV2-L | 1.000 | MB-TaylorFormerV2-L full | 34.932525 | 0.990901 | +0.787023 | +3.049596 | -1.778787 | +0.00128147 | 0.580000 | 238.80 | -12.412699 |

## Reliability Checks

- Preflight alignment pass: `True`; image count `1000`, missing GT `0`, missing raw depth `0`, missing depth2l `0`, size mismatch `0`.
- A0 checkpoint sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- FSNet+UDP checkpoint sha256: `25cc334f44c2fac979baad7f158526c9f8d751c21ea282974b0e4d9791fc0a27`.
- MB-TaylorFormerV2-L checkpoint sha256: `954229a6862cd7058c8769a9362a88f9ef2ef132664a1b05e7f7f204b617f2f9`.
- A preliminary FSNet+UDP attempt using raw `.npy` depth values and pad-32 inference was invalidated and deleted; final FSNet+UDP rows use official-style `depth2l` PNG input and pad-8 inference. See `v211_invalidated_fsudp_raw_npy_pad32_note.md`.
- Locked policy: diagnostic-only; no locked-grid alpha selection or retuning.

## Evidence Files

- `v211_haze4k_locked_cross_expert_alpha_grid_summary.json`
- `v211_haze4k_locked_cross_expert_alpha_grid_endpoint_reproduction_metrics.csv`
- `v211_haze4k_locked_cross_expert_alpha_grid_combined_alpha_grid_absolute_metrics.csv`
- `v211_haze4k_locked_cross_expert_alpha_grid_combined_alpha_grid_compact_metrics.csv`
- `v211_haze4k_locked_cross_expert_alpha_grid_fsudp_per_image.csv`
- `v211_haze4k_locked_cross_expert_alpha_grid_mbtaylor_per_image.csv`
- `v211_preflight.json`
- `v211_invalidated_fsudp_raw_npy_pad32_note.md`
- `commands/run_v211_locked_cross_expert_alpha_grid_parallel.sh`
- `commands/run_v211_repair_fsudp_official_depth2l.sh`
- `runtime_logs/`

# Haze4K v2.11 Locked Cross-Expert Alpha Grid

Date: 2026-06-16

Status: `V211_HAZE4K_LOCKED_CROSS_EXPERT_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`

## Purpose

Run the Haze4K locked test `1000`-image alpha grid for the two remaining strong
experts, FSNet+UDP and MB-TaylorFormerV2-L, under official loading/evaluation
contracts where available. This extends v2.10 WDMamba locked-grid evidence, but
it is diagnostic only and must not be used to select or retune alpha.

## Runtime

- host: `convir-4090`
- runtime workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router-locked`
- route evidence:
  `experience_docx/experiment_logs/haze4k_v2_11_locked_test_cross_expert_alpha_grid_20260616/`
- data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K/test`
- A0 checkpoint sha256:
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`
- FSNet+UDP checkpoint sha256:
  `25cc334f44c2fac979baad7f158526c9f8d751c21ea282974b0e4d9791fc0a27`
- MB-TaylorFormerV2-L checkpoint sha256:
  `954229a6862cd7058c8769a9362a88f9ef2ef132664a1b05e7f7f204b617f2f9`

## Protocol

Alpha candidates are:

```text
A0 + alpha * (Expert - A0)
alpha in {0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.0}
```

The alpha-grid metrics use the v2.2 locked one-shot compatible convention:
A0 and alpha candidates use the A0 factor-32 padded PSNR/SSIM protocol. Expert
endpoint metrics are recorded separately.

FSNet+UDP uses the official UDPNet `Dehazing/ITS/models/FSNet_UDPNet.py` source
with the documented `num_heads=1 -> 2` builder patch required to strict-load
the Haze4K checkpoint. Its final repaired run uses official-style
`test/depth2l/*.png` depth input and factor-8 inference padding; the endpoint
reproduces the UDPNet README Haze4K reference within rounding tolerance.

MB-TaylorFormerV2-L uses official `Dehazing/Options/MB-TaylorFormerV2-L.yml`,
`HAZE4K-L.pth`, factor-8 inference padding, and `strict=False` checkpoint load
matching the official `Dehazing/test.py`. The checked repo files do not expose a
clear Haze4K V2-L reference table row, so its endpoint is recorded as provenance
rather than a pass/fail reproduction gate.

## Results

Endpoint reproduction:

| Expert | Endpoint PSNR | Endpoint SSIM | Reference |
| --- | ---: | ---: | --- |
| FSNet+UDP | `35.274720` | `0.990780` | UDPNet README Haze4K `35.31 / 0.99` |
| MB-TaylorFormerV2-L | `34.932525` | `0.990711` | no clear checked Haze4K V2-L row |

Alpha-grid compact result:

| Expert | Alpha | PSNR | SSIM grid32 | mean dPSNR | hard dPSNR | easy dPSNR | positive | severe/600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FSNet+UDP | `0.125` | `34.489578` | `0.990209` | `+0.344076` | `+0.267909` | `+0.420864` | `0.892` | `25.80` |
| FSNet+UDP | `0.250` | `34.796454` | `0.990688` | `+0.650952` | `+0.527762` | `+0.776468` | `0.875` | `45.60` |
| FSNet+UDP | `0.375` | `35.055160` | `0.991052` | `+0.909658` | `+0.774331` | `+1.047503` | `0.858` | `63.00` |
| FSNet+UDP | `0.500` | `35.253876` | `0.991298` | `+1.108374` | `+1.000030` | `+1.215762` | `0.834` | `79.80` |
| FSNet+UDP | `0.750` | `35.430463` | `0.991415` | `+1.284961` | `+1.364635` | `+1.193786` | `0.777` | `117.60` |
| FSNet+UDP | `1.000` | `35.274720` | `0.990983` | `+1.129218` | `+1.569141` | `+0.694019` | `0.692` | `165.00` |
| MB-TaylorFormerV2-L | `0.125` | `34.550039` | `0.990348` | `+0.404537` | `+0.443969` | `+0.335874` | `0.904` | `24.60` |
| MB-TaylorFormerV2-L | `0.250` | `34.883635` | `0.990908` | `+0.738134` | `+0.894108` | `+0.496053` | `0.871` | `49.20` |
| MB-TaylorFormerV2-L | `0.375` | `35.133079` | `0.991304` | `+0.987577` | `+1.345382` | `+0.465108` | `0.827` | `80.40` |
| MB-TaylorFormerV2-L | `0.500` | `35.290567` | `0.991541` | `+1.145065` | `+1.787917` | `+0.255119` | `0.776` | `115.20` |
| MB-TaylorFormerV2-L | `0.750` | `35.311925` | `0.991539` | `+1.166423` | `+2.580944` | `-0.591450` | `0.681` | `172.80` |
| MB-TaylorFormerV2-L | `1.000` | `34.932525` | `0.990901` | `+0.787023` | `+3.049596` | `-1.778787` | `0.580` | `238.80` |

## Decision

```text
V211_HAZE4K_LOCKED_CROSS_EXPERT_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY
```

This confirms the Haze4K locked residual-shrinkage curve extends beyond
WDMamba: FSNet+UDP and MB-TaylorFormerV2-L both gain strongly at intermediate
alpha while full replacement increases tail risk and/or easy-case damage. The
result is locked diagnostic evidence only; no new alpha, expert, router,
threshold, checkpoint, or distillation target may be selected from this locked
grid.

## Reliability Note

An initial FSNet+UDP attempt used raw `.npy` depth values and factor-32 padding,
which failed official reproduction. Those preliminary CSV/log rows were deleted
and replaced by the repaired official-style `depth2l`/pad-8 run. See
`v211_invalidated_fsudp_raw_npy_pad32_note.md` in the evidence root.

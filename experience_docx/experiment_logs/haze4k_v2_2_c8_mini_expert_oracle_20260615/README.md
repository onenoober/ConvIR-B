# Haze4K v2.2 C8-Mini Multi-Expert Complementarity Evidence

Status: `C8_PASS_COMPLEMENTARITY_PROVEN`

Route card: `experience_docx/experiment_cards/2026-06-15-haze4k-v2-2-c8-mini-expert-oracle.md`

## Runtime Contract

- Host: `convir-4090` only.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Haze4K data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Split JSON: `experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json`.
- Locked test: untouched and forbidden for C8.

## Completed Phases

- C8-0: preregistration, expert manifest, script hashes, dependency/asset audit, no-locked status.
- C8-1: WDMamba single-expert audit completed and passed.
- C8-2: FSNet+UDP duplicate audit completed; not duplicate; S2 forward-selection passed.
- C8-3: MB-TaylorFormerV2-L fallback completed; S3 forward-selection passed.
- C8-Decision: C9 router design is scientifically justified; no router was trained here.

## Main Evidence

- `v22_c8_decision.md`
- `v22_c8_summary.json`
- `v22_c8_1_wdmamba_decision.md`
- `v22_c8_2_fsudp_duplicate_audit.md`
- `v22_c8_2_s2_decision.md`
- `v22_c8_3_s3_decision.md`
- `v22_c8_forward_selection_per_image.csv`

## Headline Metrics

- S1 WDMamba mean/hard gain over S0: `2.824226` / `4.453624` dB.
- S2 mean/hard gain over S0: `3.116570` / `4.473811` dB.
- S3 mean/hard gain over S0: `3.158518` / `4.559721` dB.
- S3 selected-oracle severe count: `0`.

- S3 group-min mean/hard gain over S0 across fixed bins: `+1.559336` / `+1.966238` dB.

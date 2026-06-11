# Haze4K DTA-v3 DAPC Fine-Tune Evidence

Date: 2026-06-11

Status: `PLANNED_FINE_TUNE_PREFLIGHT`

This directory stores text evidence for `codex/haze4k-dta-v3-dapc-finetune`.
Checkpoints, model weights, datasets, images, arrays, archives, and raw inference
outputs are not committed by default. Contact-sheet PNGs are generated on
`convir-5090` for visual judgment and recorded by path only unless explicitly
requested otherwise.

## Runtime Contract

- Host: `convir-5090`.
- Workspace: `/home/caozhiyang/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune`.
- Python: `/home/caozhiyang/ConvIR-B/envs/convir-cu128/bin/python`.
- Data: `/home/caozhiyang/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official A0 checkpoint: `/home/caozhiyang/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Depth cache: to be verified by setup; expected `/home/caozhiyang/ConvIR-B/depth_cache/depth_anything_v2_small_hf`.

## Execution Order

1. Setup remote workspace from GitHub branch and run cloud static `py_compile`.
2. Generate/verify train-derived OOF splits.
3. Run DTA-v3 preflight from A0 partial-load.
4. Run Phase A R0 zero-depth fine-tune and evaluate vs A0.
5. Run Phase B frozen-R0 depth fine-tune with invert/normal/zero/deterministic shuffle matrix.
6. Run output-refine-only, FiLM-only, trans-head-only, and phys-blend-only ablations.
7. Generate tail/win contact sheets on cloud for visual inspection.
8. Sync text evidence to GitHub and keep locked Haze4K test blocked unless internal gates pass.

## Planned Text Artifacts

- `setup_convir5090_dta_v3.sh`
- `run_dta_v3_preflight_convir5090.sh`
- `run_dta_v3_phase_a_r0_convir5090.sh`
- `run_dta_v3_phase_b_depth_matrix_convir5090.sh`
- `status.txt`
- `dta_v3_preflight.json/log`
- `depth_eval_pairing_audit.csv/json`
- `train_eval_depth_matrix.json/csv`
- `r0_vs_rdepth_attribution.csv`
- ablation JSON/CSV/log files
- contact-sheet generation logs and remote image paths

## Current Decision

`PLANNED_FINE_TUNE_PREFLIGHT`: code and scripts are being prepared. No cloud
runtime validation has been interpreted yet, and locked Haze4K test is blocked.

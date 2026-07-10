# FAM2 No-Op Protocol

Status: `PLANNED`

## Inputs

- Baseline model: `build_net("base", "Haze4K", "original")`.
- Candidate model: `build_net("base", "Haze4K", "fam2_modres")`.
- Checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Data:
  `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Split:
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json`.

## Checks

1. Strict-load A0 with no missing or unexpected keys.
2. Partial-load candidate with missing keys exactly:
   - `FAM2.modulator.weight`
   - `FAM2.modulator.bias`
3. Verify no unexpected keys and no shape mismatches.
4. Verify candidate parameter delta equals `8320`.
5. Verify FAM2 modulator weight/bias stats are all zero.
6. Verify random tensor no-op equivalence.
7. Verify real train batch no-op equivalence.
8. Verify internal val-inner 600 final output, PSNR, and SSIM equivalence.

## Gates

- output max abs diff: `<= 1e-7`
- PSNR/SSIM absolute delta: `<= 1e-10`
- training: none
- locked test: none
- RARM/D7c forward connection: none

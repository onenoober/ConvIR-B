# Haze4K v2.9 NH-HAZE Official-Test Alpha Grid Evidence

- Route card: `experience_docx/experiment_cards/2026-06-16-haze4k-v2-9-nhhaze-official-test-alpha-grid.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Decision: `V29_NHHAZE_OFFICIAL_TEST_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`
- Runtime host: `convir-4090`
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v28-nhhaze-official-weights`
- Staging dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE-official-test-51-55`
- Official split: NH-HAZE test ids `51 52 53 54 55` only
- Pair count: `5`
- Haze4K locked test touched: `false`
- NH-HAZE alpha tuning: `false`

This route replaces the deleted v2.8 all-55 mixed-split records and the v2.8b
post-hoc split audit with a clean rerun on the official-style NH-HAZE test ids
`51-55`. Both endpoints use NH-HAZE-specific checkpoints:

- A0 / ConvIR-B checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/nhhaze-base.pkl`
- A0 construction: `build_net("base", "NHR", "original")`
- A0 checkpoint sha256:
  `aab6a72613781900a23c3922ad2dd60f6b0d563018e33ae75162bcf3338f5bac`
- WDMamba checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/NH_20.83.pth`
- WDMamba construction: `WaveMamba` with `DENet(3, 4)` for strict NH checkpoint
  loading
- WDMamba checkpoint sha256:
  `e097524f466b24f32843867911f9cbd47be8d51e61e5e345f8a27c22c73d5c5a`

Absolute endpoint reproduction on official test `51-55`:

```text
A0_NH / ConvIR-B: 20.663593 PSNR, 0.796806 SSIM
WDMamba_NH:       20.830742 PSNR, 0.818217 SSIM
```

These align with the ConvIR-B README NH-HAZE base row `20.66/0.802` and the
WDMamba `NH_20.83.pth` checkpoint naming. The earlier inflated all-55 A0 result
was therefore a split-contamination artifact, not a valid official benchmark
number.

Alpha grid relative to A0_NH:

```text
alpha=0.125: mean +0.224743, hard +0.100395, easy +0.277756, dSSIM +0.00889715, positive 1.0, severe 0/5, worst +0.100395
alpha=0.250: mean +0.398761, hard +0.127104, easy +0.524015, dSSIM +0.01628561, positive 1.0, severe 0/5, worst +0.127104
alpha=0.375: mean +0.515796, hard +0.078772, easy +0.732107, dSSIM +0.02203434, positive 1.0, severe 0/5, worst +0.078772
alpha=0.500: mean +0.571267, hard -0.042160, easy +0.895655, dSSIM +0.02600487, positive 0.8, severe 0/5, worst -0.042160
alpha=0.750: mean +0.490403, hard -0.475842, easy +1.068354, dSSIM +0.02805873, positive 0.6, severe 1/5, worst -0.475842
alpha=1.000: mean +0.167149, hard -1.103455, easy +1.017271, dSSIM +0.02141021, positive 0.6, severe 2/5, worst -1.103455
```

Interpretation: inherited `alpha=0.375` is positive and tail-safer than the full
WDMamba endpoint on the five official-test images. This remains diagnostic only:
NH-HAZE has only five official test images, and this route must not be used to
select a new NH-HAZE alpha without a separate validation or OOF protocol.

Primary files:

- `v29_decision.md`
- `v29_final_audit.json`
- `v29_nhhaze_official_test_alpha_grid_summary.json`
- `v29_nhhaze_official_test_alpha_grid_manifest.json`
- `v29_nhhaze_official_test_alpha_grid_alpha_grid.csv`
- `v29_nhhaze_official_test_alpha_grid_compact_alpha_comparison.csv`
- `v29_nhhaze_official_test_alpha_grid_per_image.csv`
- `v29_nhhaze_official_test_alpha_grid_group_metrics.csv`
- `v29_nhhaze_official_test_alpha_grid_group_min.csv`
- `commands/run_v29_nhhaze_official_test_alpha_grid.sh`
- `runtime_logs/v29_nhhaze_official_test_alpha_grid.log`
- `status_v29_nhhaze_official_test_alpha_grid.txt`

Note: the runtime tool was reused from the v2.8 audit workspace and prints a
legacy `V28_NHHAZE_OFFICIAL_AGGREGATE_OK` marker in the raw log. The normalized
route, status, and decision for this rerun are the v2.9 files listed above and
`V29_NHHAZE_OFFICIAL_TEST_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`.

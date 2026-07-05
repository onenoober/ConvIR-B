# Haze4K v2.32 NoPost Bounded Internal Low-Frequency Correction Field Evidence

Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-32-nopost-bounded-internal-lowfreq-correction-field.md`

Status: `PLANNED`

Runtime server: `convir-4090`
Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-32-nopost-bounded-internal-lowfreq-correction-field`
Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Hard blocks:

- `p2b_selector_probe_launched: false`
- `locked_test_touched: false`
- `rgb_output_output_residual: false`
- `learned_rgb_post_output_correction: false`

## Current Plan

P0 validates the BILFCF architecture contract and zero-init identity. P1 and P2
then use train-derived canary screens only. If a gate fails, the route pauses as
a normal scientific/engineering screen result and does not continue to later
stages.

This directory is intended for compact text evidence only. It excludes
checkpoints, weights, datasets, images, arrays, archives, and raw feature tables
by default.

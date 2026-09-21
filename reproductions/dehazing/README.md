# AECR-Net and FFA-Net reproduction

This route uses pinned upstream implementations as separate source checkouts.
It does not modify the ConvIR-B architecture or reuse its checkpoint.

See `experience_docx/experiment_cards/2026-09-21-aecr-ffa-repro.md` for the
dataset, validation, metric, and source-compatibility contracts.

Pinned sources:

- `https://github.com/GlassyWu/AECR-Net.git` at
  `d046d0f5f849692ddb6eb6eb3d14f23c86ff5ca6`
- `https://github.com/zhilin007/FFA-Net.git` at
  `6651bcf693edd3ba2d097f728676942bd40e6e99`

Clone them as sibling `aecr-net-upstream-20260921` and
`ffa-net-upstream-20260921` directories under the cloud `repos/` root, then
run the recorded `run_cloud_smoke.sh` once in a fresh output directory. The
script refuses to overwrite checkpoint files. It reads Haze4K `train` only;
the resulting checkpoints contain one optimization step and are not pretrained
models. The AECR compatibility operators retain the official state-key names
but have not been numerically compared with the old DCNv2 extension.

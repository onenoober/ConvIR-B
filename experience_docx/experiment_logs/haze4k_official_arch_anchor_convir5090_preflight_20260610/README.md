# Haze4K Official Architecture Anchor convir-5090 Preflight

Date: 2026-06-10

Status: `OFFICIAL_ANCHOR_CONVIR5090_PREFLIGHT_OK`.

## Scope

This evidence records a migration smoke/preflight of the GitHub branch
`codex/haze4k-official-arch-anchor` on the backup cloud server `convir-5090`.
It verifies that the official ConvIR-B Haze4K architecture anchor can load the
trusted official Haze4K baseline checkpoint, run a synthetic forward pass, and
read one Haze4K training batch on GPU without using the locked test split for
model selection.

## Runtime Paths

- Workspace: `/data/caozhiyang/ConvIR-B/repos/ConvIR-B-official-arch-anchor`
- Python: `/data/caozhiyang/ConvIR-B/envs/convir-cu128/bin/python`
- Data: `/data/caozhiyang/ConvIR-B/datasets/Haze4K/Haze4K`
- Checkpoint: `/data/caozhiyang/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`

## Primary Evidence

- `run_official_anchor_preflight_convir5090.sh` - durable command script.
- `official_anchor_preflight_convir5090.log` - stdout/stderr log.
- `official_anchor_preflight_convir5090.json` - structured preflight result.
- `status.txt` - start/done markers and final OK marker.

## Key Result

- Branch commit: `2d529d4` from `github/codex/haze4k-official-arch-anchor`.
- Checkpoint sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Parameter count: `8630665`.
- Synthetic output shapes: `[1,3,64,64]`, `[1,3,128,128]`, `[1,3,256,256]`.
- One train-batch multiscale L1: `0.009163891896605492`.
- Final marker: `OFFICIAL_ANCHOR_CONVIR5090_PREFLIGHT_OK`.

## Notes

The convir-5090 environment uses PyTorch `2.11.0+cu128`; the script sets
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the trusted legacy checkpoint whose
sha256 is recorded above.

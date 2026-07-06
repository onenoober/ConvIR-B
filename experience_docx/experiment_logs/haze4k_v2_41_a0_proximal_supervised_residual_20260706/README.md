# Haze4K v2.41 A0-Proximal Supervised Residual Evidence

Status: `COMPLETED_GATE_FAIL`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-41-a0-proximal-supervised-residual.md`

Central index path:
`experience_docx/EXPERIMENT_INDEX.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-41-a0-proximal-supervised-residual`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked.

## Evidence Files

Compact sync candidates:

- `status.txt`
- `run_v241_p0_stage0_preflight.sh`
- `v241_p0_stage0_preflight.json`
- `v241_p0_closeout.json`
- `run_v241_canary32_oof.sh`
- `v241_canary32_oof_summary.json`
- `v241_canary32_oof_folds.csv`
- `v241_canary32_oof_epoch_history.csv`
- `v241_canary32_oof_per_image.csv`
- `v241_canary32_closeout.json`

## Metric Contract

P0 Stage-0 proves strict official-checkpoint partial load, zero-init/no-op
equivalence, finite forward, no forbidden postprocess symbols, and locked-test
blocked status. It does not train, run canary80, or touch locked test.

Canary32 OOF is authorized only after P0 pass. It trains only `A0PROX_*` on
five train-derived folds (`32` train images and disjoint `32` held-out images
per fold) and freezes all official ConvIR-B parameters. It does not use teacher
targets, selectors, canary80, or locked test.

Canary32 passes only if global mean/hard/easy/p05/CVaR5/tail gates pass,
`4/5` folds pass the same gate, easy residual energy is at most `0.50` of hard
residual energy, and hinge violation rate does not worsen across training.

## Result

P0 Stage-0 preflight passed at route commit `7c27b93`.

Key P0 facts:

- checkpoint sha256:
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`;
- strict partial load loaded `602` official keys;
- missing keys were only `A0PROX_beta` and `A0PROX_head.*`;
- synthetic output shapes were `64/128/256`;
- identity max absolute difference vs official ConvIR-B was `0.0`;
- forbidden postprocess symbol hits were `0`;
- locked test was untouched.

Canary32 OOF completed on `convir-4090` at route commit `32e7791`.

Decision:
`CANARY32_OOF_GATE_FAIL_LOCK_CANARY80_LOCKED_TEST`.

The run used only train-derived Haze4K images: `5` folds, `32` training images
per fold, and disjoint `32` held-out train-derived images per fold. It froze all
official ConvIR-B parameters and trained only `11843` `A0PROX_*` parameters.
No teacher target, selector, canary80, or locked test was used.

Global OOF metrics over `160` held-out train-derived images:

| Metric | Value | Gate |
| --- | ---: | ---: |
| mean delta | `-0.0277 dB` | `>= +0.15 dB` |
| hard bottom-25% delta | `+0.0742 dB` | `>= +0.30 dB` |
| easy top-25% delta | `-0.0724 dB` | `>= +0.00 dB` |
| p05 delta | `-0.3981 dB` | `>= -0.01 dB` |
| CVaR5 delta | `-0.5972 dB` | `>= -0.02 dB` |
| severe regressions | `27` | `0` |
| strong-reference regressions | `25` | `0` |
| fold pass count | `0/5` | `>= 4/5` |

Mechanism diagnostics passed but did not rescue the route:

- easy/hard residual-energy ratio: `0.2871` against gate `<=0.50`;
- hinge violation rate: `0.6625 -> 0.58125`.

Interpretation: the frozen-backbone A0-proximal residual head can concentrate
more residual energy on hard than easy images, but it does not deliver
tail-safe OOF quality improvement. Canary80 and locked test remain blocked.

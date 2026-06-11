# Haze4K ConvIR-B DTA-v3 DAPC Fine-Tune Route

Date: 2026-06-11

Status: `PREFLIGHT_COMPLETE_READY_PHASE_A_R0`

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: Innovation 1 / depth-guided transmission adapter.
- Route name: DTA-v3 DAPC, Depth-Attributed Preservation-Controlled Adapter.
- Branch: `codex/haze4k-dta-v3-dapc-finetune`.
- Anchor: `github/codex/haze4k-official-arch-anchor` at `2d529d4`; this route imports DTA data/prior plumbing needed by the mechanism but does not fine-tune from a DTA-v2 checkpoint.
- Diagnostic predecessor: `github/codex/haze4k-dta-v2-calibrated` at `9e95408`, used only for mechanism diagnosis and evidence targets.
- Runtime host: `convir-4090`.
- Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Expected depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf` unless setup audit finds a different existing cache.
- Evidence root: `experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/`.
- Local policy: local WSL is for editing and compile/static checks only; no local tests, smoke tests, training, eval, inference, demos, or image generation.

## Hypothesis

DTA-v2 CalGate was a positive diagnostic but not a depth-attributed candidate:
zero/shuffle controls retained most of the gain, `normal` slightly beat calibrated
`invert`, SSIM was slightly negative, and tail regressions remained high. DTA-v3
DAPC tests whether separating a generic zero-depth residual `R0` from a frozen-R0
depth-specific physical correction `R_depth * M_depth` can increase true-vs-zero
surplus while reducing SSIM/tail risk.

## Architecture Change

`--arch dta_v3` adds `DTA.*` modules only:

- stage-2/stage-3 calibrated prior FiLM with wider but bounded gates;
- supervised `transmission_head` for `t_pred`;
- `r0_refine` generic zero-depth residual branch;
- `depth_mask_head` risk/preservation mask;
- physical correction blend using `J_phys = (I - A * (1 - t_pred)) / clamp(t_pred)` and a bounded mask.

The final output is:

```text
A0 output + R0(x) + M_depth(x, d, t_pred, conf) * bounded(J_phys - (A0 output + R0))
```

## Partial-Load And Initialization Contract

- Fine-tuning starts from official Haze4K A0, never from DTA-v2 weights.
- Use `--init_model_partial --partial_new_prefixes DTA.` for Phase A from A0.
- Official ConvIR-B keys must load with strict matching shapes.
- Missing keys are allowed only under `DTA.`.
- Unexpected checkpoint keys are fatal.
- New module init:
  - `DTA.stage2/3`: identity by zero last projection; gate bias defaults to `-5.0`, gate limit `0.10`, gamma limit `0.16`, beta limit `0.08`.
  - `DTA.transmission_head`: zero last projection.
  - `DTA.r0_refine`: zero last projection, so Phase A starts as exact no-op.
  - `DTA.depth_mask_head`: conservative bias `-4.0`, bounded budgets `0.04` easy / `0.12` dense.
- Phase B initializes from the same-arch Phase A route checkpoint using full strict load with `--init_model_allow_full_route`; `DTA.r0_refine` is frozen by train scope.

## Fine-Tune Ladder

| Stage | Trainable scope | Command intent | Continue only if |
| --- | --- | --- | --- |
| Stage 0 preflight | none / one-batch gradient probe | strict A0 partial load, no-op/bounded diff, depth/trans availability, finite losses | partial load clean, no locked test touched, DTA gradients finite |
| Phase A R0 | `dta_r0_only`, `dta_phase=r0`, `dta_ablation=r0_only`, `depth_mode=zero` | learn the generic zero-depth residual baseline safely | R0 is positive vs A0, SSIM non-negative or near zero, tail no worse than current DTA-v2 zero |
| Phase B depth surplus | `dta_depth_only`, `dta_phase=depth`, init from Phase A | freeze R0, train FiLM/trans/head/mask physical depth surplus | `invert/true` beats zero/shuffle/wrong orientation, tail and SSIM gates pass |
| Ablations | route-specific train scopes/modes | locate source of gains | output-refine-only, FiLM-only, trans-head-only, and phys-blend-only evidence is written |
| OOF/multi-seed | same fixed settings | verify stability | promotion gate passes on train-derived OOF only |

## Default DTA-v3 Fine-Tune Settings

- Gate: `gate_bias=-5.0`, `gate_limit=0.10`, `gamma_limit=0.16`, `beta_limit=0.08`, confidence floor `0.30`.
- Mask: easy budget `0.04`, dense budget `0.12`, density threshold `0.35`, mask bias `-4.0`, physical `t_min=0.10`.
- Loss: original multiscale L1 + `0.1 * FFT`, plus supervised transmission/physics/preserve losses when Haze4K `trans/A` are available.
- Added protection: optional A0 reference preserve and tail guard losses using frozen official ConvIR-B output.
- Depth controls: `invert` is the primary calibrated mode from DTA-v2 audit; `normal` is wrong-orientation control; `zero` and deterministic eval shuffle are mechanism controls.

## Required Intermediate Artifacts

| Artifact | Required purpose |
| --- | --- |
| `dta_v3_preflight.json/log` | partial-load, no-op/bounded diff, real batch gradients, finite losses |
| `depth_eval_pairing_audit.csv/json` | prove eval shuffle is deterministic and not batch-size-1 no-op |
| `train_eval_depth_matrix.json/csv` | separate training regularization from inference depth use |
| `r0_vs_rdepth_attribution.csv` | per-image generic residual vs depth surplus decomposition |
| `output_refine_only_ablation.json` | measure generic residual head contribution |
| `film_only_ablation.json` | measure stage2/stage3 depth FiLM contribution |
| `trans_head_only_no_rgb_residual.json` | prove transmission head can learn without RGB action |
| `phys_blend_only.json` | measure physical correction tail behavior |
| `t_to_image_coupling.csv/json` | correlate t quality with image delta |
| `tail_regression_contact_sheet/` | cloud PNG contact sheets for best wins and worst regressions; not committed to Git by default |
| `risk_router_calibration.json` | mask/gain/loss calibration if Phase B reaches selector analysis |
| `ssim_tail_report.json` | SSIM and regression distribution summary |

## Internal Promotion Gate

Locked Haze4K test is blocked unless all train-derived gates pass:

```text
mean_dPSNR(true - A0) >= +0.08
mean_dPSNR(true - zero/R0) >= +0.03
hard_dPSNR(true - zero/R0) >= +0.04
true - shuffle_eval_fixed_perm >= +0.03
true - normal_wrong_orientation >= +0.02
SSIM(true - A0) >= 0 or CI not significantly negative
worst regressions(true) <= zero/R0
strong regressions(true) <= zero/R0
positive_ratio >= 0.65
```

If the route only beats A0 but does not beat zero/shuffle/wrong-orientation, it
is a generic adapter and must not be called depth-guided.

## Locked-Test Policy

Locked Haze4K test must not be used to select checkpoint, depth mode, gate,
loss, mask budget, train scope, or ablation. A locked test is allowed only once
for a fixed configuration that passes the internal OOF mechanism/preservation
gate. Current status: locked test blocked.

## Stop Rules

- Stop Phase A if R0 is not positive or worsens SSIM/tail beyond DTA-v2 zero.
- Stop Phase B if `invert` fails to beat zero/shuffle/wrong orientation by the written surplus gates.
- Stop ablations that are preservation-negative before OOF expansion.
- Do not continue `adapter_neighbors` from DTA-v2 unless a new Phase B gate passes first.

## 2026-06-11 Default Host Correction

The route default cloud host has been changed to `convir-4090` by user instruction.
The earlier `convir-5090` SSH blocker is superseded and is retained only in
`status.txt` as historical setup evidence. Runtime validation, training, eval,
and contact-sheet generation should now use `ssh convir-4090` and the
`/sda/home/wangyuxin/ConvIR-B/...` runtime paths.

## 2026-06-11 Convir-4090 Preflight

Stage 0 preflight completed on `convir-4090` with `DTA_V3_PREFLIGHT_OK`.

- R0 preflight: partial load `602` official keys, `29` missing keys all under `DTA.`, unexpected `[]`, synthetic no-op max diff `0.0`, real-batch DTA grad sum `0.06329656`.
- Depth-bounded preflight: partial load `602` official keys, `29` missing keys all under `DTA.`, unexpected `[]`, synthetic max diff `0.00024414` under the written bounded tolerance, real-batch DTA grad sum `0.19976439`.
- OOF split generation: five train-derived folds, `600` validation images per fold.
- Deterministic eval shuffle audit: `600` fold0 validation rows, `same_image_count=0`, `same_image_ratio=0.0`, density-bin match ratio `0.275`.
- Locked Haze4K test remains blocked. Next authorized step is Phase A `dta_r0_only` fine-tuning from official A0 on `convir-4090`.

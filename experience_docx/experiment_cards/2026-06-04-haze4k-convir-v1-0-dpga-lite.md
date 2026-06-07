# Haze4K ConvIR-Dehaze-v1.0-DPGA-Lite

Date: 2026-06-04

Status: root-cause pre-experiments completed; adapter-only DPGA-Lite stop20
training and A0 comparisons completed on `autodl-dehaze4`. Best checkpoint
passes the minimum positive gate; exact final/epoch20 is borderline and does
not cleanly clear the `+0.02 dB` mean-PSNR rule.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: Depth/Prior-Guided Adapter Lite for ConvIR-B.
- Dataset or task: Haze4K dehazing, strict comparison against official ConvIR-B
  A0.
- Primary objective: test whether a small in-network depth/prior-guided adapter
  can improve hard hazy cases without the APDR output-residual tail failures.
- Main metric: full-test PSNR/SSIM delta against official A0, hard/easy bucket
  behavior, and strong-reference regression count.
- Execution environment: `autodl-dehaze4`,
  `/root/miniconda3/envs/convir-cu128/bin/python`.
- Branch or isolated workspace: `codex/haze4k-convir-v1-0-dpga-lite`.

## Baseline Contract

- Baseline implementation: official frozen ConvIR-B Haze4K checkpoint.
- Training entrypoint: `Dehazing/ITS/main.py --arch dpga`.
- Evaluation entrypoint:
  `experience_docx/tools/eval_haze4k_checkpoint_compare.py --candidate_arch dpga`.
- Dataset and split: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Depth prior: DepthAnything V2 cache at
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf`.
- Checkpoint/export/resume contract: initialize ConvIR weights from official A0;
  train only `DPGA_*` modules for the first run.

## Route Boundary

- Do not train full ConvIR backbone in the first run.
- Do not add RGB output residual heads.
- Do not use APDR `M_safe` or `P_benefit` as an output gate.
- Do not add dense local correction, basis coefficient heads, color branches,
  hard FFT boost, teacher distillation, or token-wise low-rank routing.
- Insert only three conservative adapters:
  shallow encoder output, bottleneck output, and decoder skip-fusion
  pre-projection.

## Architecture

The adapter form is:

```text
F_l' = F_l + effective_scale_l * Adapter_l(F_l, prior_l)
```

where `Adapter_l` is:

```text
concat(F_l, prior_l)
1x1 conv reduce
depthwise 3x3 conv
GELU
1x1 conv restore
zero-init projection
```

The learnable adapter scale is initialized at zero and the projection is
zero-initialized, so initial DPGA outputs match ConvIR exactly. A tiny fixed
bootstrap scale is used only to avoid the mathematically dead double-zero start
while preserving exact zero-output equivalence.

## Input Priors

First-run priors:

- DepthAnything V2 depth map.
- Depth gradient.
- Dark channel.
- Bright channel or sky proxy.
- Saturation.
- Local contrast and luminance gradient.
- Input luminance.

APDR masks and benefit probabilities are analysis-only for this route.

## Pre-Experiment Dependencies

The following root-cause experiments must be read before launching DPGA training:

| Dependency | Required answer | Current status |
| --- | --- | --- |
| Exp2 ConvIR-B finetune baseline | stop5/stop20 gain versus official A0 | completed: stop5 `-11.1718 dB`, Best `-0.7189 dB`, Final `-3.4913 dB`; simple finetune does not stably improve A0 under this setting |
| Exp3 residual coefficient predictability | whether depth/physics/frequency priors materially improve coefficient correlation | completed: priors add deployable signal over current/global, but do not make output residual coefficients a safe action surface |
| Exp4 severe failure visual audit | whether failure modes point to depth, sky/airlight, color constancy, or residual unpredictability | completed: failures are concentrated in sky/white-wall/water/bright low-contrast regions and strong-anchor cases where residuals introduce broad low-frequency/color shifts |

## Root-Cause Readout

Evidence directory:

```text
experience_docx/experiment_logs/haze4k_rootcause_preexp_20260604/
```

Exp2 fixed A1-vs-A0 comparisons:

- Official A0 mean: `34.1466 dB`.
- A1 stop5 mean: `22.9748 dB`, delta `-11.1718 dB`,
  strong-reference regressions `250/250`.
- A1 Best mean: `33.4277 dB`, delta `-0.7189 dB`,
  strong-reference regressions `223/250`.
- A1 Final mean: `30.6553 dB`, delta `-3.4913 dB`,
  strong-reference regressions `249/250`.

Exp3 coefficient predictability:

- `current_global_stats`: best corr `0.6065`, R2 `0.3504`.
- `convir_spatial_features`: best corr `0.6768`, R2 `0.4435`.
- `depth_features`: best corr `0.6191`, R2 `0.3639`.
- `physics_proxy_features`: best corr `0.6094`, R2 `0.3518`.
- `frequency_amplitude_features`: best corr `0.6652`, R2 `0.4281`.
- `depth_physics_frequency`: best corr `0.6662`, R2 `0.4292`.
- `global_plus_convir_plus_priors`: best calibrated kNN corr `0.6822`,
  R2 `0.4508`; ridge can reach corr `0.6965` but with negative R2
  (`-0.2600`) and overlarge prediction scale.

Decision from Exp3: depth/physics/frequency priors carry useful information,
but the improvement over ConvIR spatial features is modest and does not rescue
the output-residual coefficient route. Use these priors inside ConvIR features,
not as another RGB output action/gate.

Exp4 severe/strong visual audit:

- v0.4D rendered `24` grids. Severe mean gain was `-1.2396 dB`, strong mean
  gain was `-1.1692 dB`, and mean residual correlation was about `-0.47`.
- v0.4E rendered `24` grids selected from `60,000` candidates. Severe mean
  gain was `-0.8159 dB`; strong-anchor mean gain was `-0.1184 dB` with anchors
  around `44.46 dB`.
- Repeated failure patterns: sky and white-wall/white-ground regions, water and
  glass/airlight gradients, bright low-contrast regions, mild near-foreground
  color/contrast shifts, and cases where the anchor is already very close to GT
  but the residual introduces a broad low-frequency field.

Decision from Exp4: the next prior should emphasize depth/transmission,
sky/airlight, and color constancy inside feature learning. Do not continue
output-end RGB residual coefficient mapping.

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| Zero-init equivalence | max absolute output difference from ConvIR A0 is `<=1e-7` before training | adapter insertion does not alter the anchor | required before any cloud training |
| Stop5 smoke | no catastrophic full-test drop; inspect hard/easy/strong buckets | adapter can learn without immediately dragging easy/strong-reference cases | continue to stop20 only if no obvious tail collapse |
| Stop20 pass | mean PSNR delta `>= +0.02 dB`, SSIM delta `>= 0`, hard bottom-25% improves, easy top-25% not below `-0.05 dB`, strong-reference regression count improves over failed APDR output-residual routes | depth/prior-guided in-network modulation is useful and safer than output residual | consider a second adapter-only refinement |
| Stop20 fail | mean/hard gain absent or easy/strong tail regresses | DPGA-Lite in this form is not enough; next change must revisit priors, loss, data, or backbone strength | do not add complex token router as a reflex |

## Current Decision

Decision label:

```text
DPGA_LITE_ADAPTER_ONLY_MIN_POSITIVE_BEST_BORDERLINE_FINAL
```

Local code checks completed:

- Python compile for DPGA model, data loaders, train/valid/eval, compare tool,
  and zero-init checker.
- Synthetic `256x256` zero-init equivalence:
  all three ConvIR output scales have max absolute difference `0`.
- Adapter projection gradient probe: all three DPGA projections receive nonzero
  first-step gradients under adapter-only training mode.
- Synthetic Haze4K-style data-loader smoke:
  train returns aligned `(image, label, depth)` `256x256` crops; valid/test
  return aligned full-size tensors plus filename.

Remote execution completed:

- DepthAnything V2 Haze4K test cache generated and verified at
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf/test`
  with `1000` `.npy` files.
- Adapter-only training ran on `autodl-dehaze4` in
  `/root/autodl-tmp/workspace/ConvIR-B-dpga-lite-826caaf`.
- Training status:
  `train_done ConvIR-Haze4K-DPGA-Lite-v1.0-adapter-only-stop20-seed3407-20260604 rc=0`.
- Trainable/frozen split at launch: `48,435` trainable DPGA parameters and
  `8,630,665` frozen ConvIR parameters.
- Checkpoints produced:
  `model_5.pkl`, `model_10.pkl`, `model_15.pkl`, `model_20.pkl`, `Best.pkl`,
  and `Final.pkl`.

## Adapter-Only Stop20 Results

Evidence directory:

```text
experience_docx/experiment_logs/haze4k_dpga_lite_20260604/eval_a0_compare
```

Comparison baseline:

- Official frozen ConvIR-B A0 checkpoint:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Full Haze4K test split: `1000` images.
- Evaluation command:
  `experience_docx/experiment_logs/haze4k_dpga_lite_20260604/run_dpga_lite_eval_a0_compare.sh`.

Full-test A0 comparisons:

| Checkpoint | Mean PSNR | Delta vs A0 | SSIM delta | Hard bottom-25% delta | Easy top-25% delta | Strong refs regressed <= -0.05 | Worst regressions <= -0.20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `model_5.pkl` | `34.1237` | `-0.0229` | `+0.000003` | `-0.0058` | `-0.0581` | `121/250` | `194/1000` |
| `model_20.pkl` | `34.1659` | `+0.0193` | `+0.000048` | `+0.0037` | `+0.0096` | `108/250` | `161/1000` |
| `Best.pkl` | `34.1778` | `+0.0312` | `+0.000088` | `+0.0146` | `+0.0209` | `105/250` | `163/1000` |
| `Final.pkl` | `34.1659` | `+0.0193` | `+0.000048` | `+0.0037` | `+0.0096` | `108/250` | `161/1000` |

Bucket details:

- `Best.pkl` has the clearest positive shape: hard bottom-25% mean delta
  `+0.0146 dB`, medium middle-50% `+0.0447 dB`, easy top-25% `+0.0209 dB`,
  and positive-delta ratios of `0.600`, `0.614`, and `0.532`.
- Exact stop20/final is positive but thin: mean delta `+0.0193 dB`, just below
  the preset `+0.02 dB` mean-PSNR gate; hard delta is only `+0.0037 dB`.
- Stop5 did not pass: mean delta `-0.0229 dB`, hard delta `-0.0058 dB`, and
  easy top-25% delta `-0.0581 dB`.

Gate readout:

| Gate | Result | Decision |
| --- | --- | --- |
| Zero-init equivalence | Passed exactly on all three output scales (`max_abs=0`) | route is implementation-safe |
| Stop5 smoke | No collapse, but negative mean/hard/easy deltas | continue only because stop20 was already requested |
| Stop20 exact final | Mean `+0.0193 dB`, SSIM positive, hard/easy positive | borderline fail against strict `+0.02 dB` mean rule |
| Best checkpoint | Mean `+0.0312 dB`, SSIM positive, hard/easy positive | minimum positive pass |

## Route Decision

DPGA-Lite adapter-only is not a dead route. It is the first recent ConvIR-B
Haze4K route in this thread that gives a small positive deployable full-test
gain without RGB output residuals, full-backbone training, FFT boost, teacher
distillation, or token-wise routing. The result supports the root-cause
hypothesis that depth/prior information is more useful inside ConvIR features
than as output-end residual coefficients.

The evidence is still not strong enough for promotion as a final model line.
The exact final/stop20 checkpoint misses the strict mean-PSNR gate by about
`0.0007 dB`, and strong-reference tail regressions remain high
(`105-108/250`). Treat this as a minimum positive proof-of-direction, not a
finished route.

Next action should be a second conservative adapter-only refinement, not a jump
to full token-wise low-rank routing. The next run should target tail control:
lighter adapter scale, validation-selected early stopping, sky/airlight/color
constancy loss or weighting, and a visual audit of the `Best.pkl` worst
regressions before changing architecture depth.

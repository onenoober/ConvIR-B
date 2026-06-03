# Haze4K APDR-v0.4A Low-Field-Only

Date: 2026-06-03

Status: failure-branch diagnostics completed; current deployable low-field forms failed Gate B, so Gate C and stop20 are blocked.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-v0.4A LF, cached-mask correctability-gated low-field ConvIR.
- Dataset or task: Haze4K train/test on `autodl-dehaze3`.
- Primary objective: train only a deployable low-frequency residual field while keeping the ConvIR-B anchor, APDR-v0.2RC `M_safe`, and train-calibrated correctability fixed.
- Main metric: hard-sample PSNR gain under easy/strong-reference preservation.
- Secondary metrics: weighted delta target correlation, opened/closed subset gain, cache max-diff audit, residual amplitude audit, SSIM delta, strong/severe regression counts.
- Execution environment: `autodl-dehaze3`, `/root/miniconda3/envs/convir-cu128/bin/python`.
- Artifact roots: `experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603/`, `experience_docx/experiment_logs/haze4k_apdr_v0_4a_lowfield_gate_ab_20260603/`, and `experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/`.
- Branch or isolated workspace: `codex/haze4k-apdr-v0-4a-low-field-only`.
- Review package location: text-only logs/JSON/CSV under `experience_docx/experiment_logs/`; no checkpoints, tensor caches, arrays, or image outputs are committed.

## Baseline Contract

- Baseline implementation: ConvIR-B official Haze4K checkpoint.
- Baseline checkpoint or initialization: `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Frozen selector source: APDR-v0.2RC full-image selector checkpoint from `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl`.
- Evaluation entrypoint: `Dehazing/ITS/main.py --mode test` plus `experience_docx/tools/eval_haze4k_checkpoint_compare.py`.
- Training entrypoint: new v0.4A-only training/preflight scripts in this branch; no crop-recomputed APDR route may be reused.
- Dataset and split: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Metric implementation: same PSNR/SSIM and bucket analysis used by prior Haze4K route cards.
- Reproduced baseline result: Haze4K pretrained ConvIR-B around PSNR `34.14`, SSIM `0.98971`.
- Reference entrypoints that must remain stable: ConvIR-B forward path, APDR-v0.2RC full-image selector behavior, v0.4 cache/scale diagnostic semantics.
- Checkpoint/export/resume contract: a v0.4A scout checkpoint is only meaningful after Gate C; pre-Gate artifacts are text diagnostics only.

## Most Valuable Attempt

- Why this is the highest-value next attempt: v0.4 diagnostics show that `M_safe`, lowpass targets, and train-calibrated correctability are useful assets, while full low+color v0.4C, crop recompute, toy residual heads, and color-only residuals are blocked.
- Target failure or opportunity: learn a low-frequency haze/illumination/color-cast field without letting easy or strong-reference images move.
- Cheap preflight evidence: cache roundtrip max diff `0.0`, sigma sweep with lowpass oracle gain, sigma-7 free-parameter low recovery/correlation, and train-calibrated correctability threshold.
- Earliest decisive gate: Gate A no-op/cache preflight, followed by Gate B overfit32.
- Expected cost or attempt-count saving: prevents another stop20 run unless the deployable low-field predictor actually learns the cached target.
- What success decides: authorize one stop20 low-field-only scout with frozen backbone, frozen selector, and frozen correctability.
- What failure decides: do not return to color or high-frequency branches; diagnose residual expression with ID-embedding, basis mixture, or physics-shaped veil residuals.
- Why a cheaper diagnostic is not enough: v0.4 proved oracle/application headroom, but a deployable image-to-field mapping still has not been shown.

## Hypothesis

Observed failure:

```text
v0.4C mixed low+color evidence is unsafe: color has weak target correlation and nonzero severe regressions.
Prior APDR toy residual heads also failed to learn useful hard-case gains.
```

Target mechanism:

```text
prior-guided, cache-safe, correctability-gated low-frequency field residual.
```

Mechanism sentence:

```text
If only LowFieldNet is trainable and the output is multiplied by cached full-image M_safe
and train-calibrated P_benefit, hard correctable samples should gain because low-frequency
haze veil residuals are learnable while closed easy samples remain effectively no-op.
```

## Change

- Code branch: `codex/haze4k-apdr-v0-4a-low-field-only`.
- Exact code/config change: introduce a dedicated low-field predictor and training protocol only after Gate A/B implementation checks; do not reuse v0.4C color or crop-recompute code.
- Enabled mechanisms: ConvIR-B anchor, frozen APDR-v0.2RC full-image `M_safe`, frozen train-calibrated `P_benefit`, cached full-image lowpass delta target, zero-init bounded low-frequency residual.
- Explicitly disabled mechanisms: v0.4C low+color stop20, standalone color branch, high-frequency/detail branch, crop-recomputed mask training, direct SHED, HardFreq loss, joint router/residual training.
- Parameter/runtime/memory impact expected: LowFieldNet target range `0.3M` to `0.8M` parameters; stop20 is not authorized until this is measured.
- Initialization or no-op behavior: final residual head must be zero-initialized; initial output max absolute diff vs ConvIR-B anchor must be `<= 1e-6`.
- Resume policy: no resume across preflight gates unless the text log states the exact checkpoint and schedule; stop20 schedule must remain matched to baseline.
- Defaults changed: lowpass sigma `7.0` is the first deployable target because free-parameter low and correctability are already aligned there.
- Defaults intentionally preserved: official ConvIR-B checkpoint, Haze4K split, train seed `3407`, APDR-v0.2RC selector checkpoint, text-only evidence policy.

## Sigma Boundary

| Target | Evidence state | Decision |
| --- | --- | --- |
| sigma `7.0` | free-parameter low and train-calibrated correctability are already aligned; low target is smoother and easier to learn | use as first v0.4A deployable target |
| sigma `3.0` | cache-scale oracle is strongest; parallel alignment diagnostics now exist for free-parameter low and correctability | eligible for Gate A/B LowFieldNet preflight; do not run stop20 until Gate C |
| sigma `5/11/15` | cache-scale sweep exists for oracle comparison | keep as table-only evidence unless a future card justifies them |

The sigma `3.0` alignment diagnostics are useful and can run in parallel with
training progress, but they must not silently authorize stop20.

Sigma `3.0` alignment result: free-parameter low has loss-drop `0.6891`,
recovery `1.0551`, corr `0.9309`, hard gain `+1.2639 dB`, easy gain
`+0.4701 dB`, and no strong/severe regressions; train-calibrated correctability
passes with test AUC `1.0`, Spearman `0.9701`, easy open `0.012`, false-open
`0.0`, and positive-hard recall `0.9600`.

## LowFieldNet-v1 Contract

Required input channels:

```text
hazy RGB x
anchor J0
x - J0
abs(x - J0)
dark channel
bright channel
saturation
gradient magnitude
M_safe
P_benefit broadcast map
```

Optional input:

```text
ConvIR decoder full-scale feature compressed to 8-16 channels
```

Architecture target:

```text
stem 3x3 conv, hidden 48 or 64
encoder downsample to 1/2 and 1/4
residual blocks with depthwise 7x7 or 11x11 kernels
dilated conv rates 1, 2, 4
simple pooled global context or gated MLP
decoder back to full resolution with shallow skips
3-channel zero-init residual head
Gaussian lowpass projection
residual_max * tanh bounding
```

## Preflight

| Gate | Pass line | Result |
| --- | --- | --- |
| Gate A no-op/cache | initial output max diff vs `J0 <= 1e-6`; cached mask crop max diff `<= 1e-8`; cached low-target crop max diff `<= 1e-8`; selector/backbone trainable params `0` | pass for sigma `3.0` and `7.0` |
| Gate B overfit32 | weighted delta L1 drop `>= 0.50`; pred-target corr `>= 0.50`; oracle recovery `>= 0.30`; hard train gain `>= +0.30 dB`; easy train gain `>= -0.010 dB`; strong/severe regressions `0` | fail for sigma `3.0` and `7.0`; learnability did not emerge |
| Gate C train128/mini-val | train128 hard gain `>= +0.20 dB`; easy gain `>= -0.010 dB`; weighted delta corr `>= 0.35`; opened positive-hard samples outperform closed samples | blocked by Gate B |
| stop20 authorization | Gate A/B/C all pass | blocked; do not run LowFieldNet-v1 stop20 |

## Training Protocol

Stage 1 asks only whether LowFieldNet can learn the cached lowpass delta target:

```text
W = stopgrad(M_safe * P_benefit)
Delta_target = cached Delta_low_star
L_delta = mean(W * SmoothL1(Delta_pred - Delta_target))
L_lowpass = mean(abs(Delta_pred - GaussianBlur(Delta_pred)))
L_tv = TV(Delta_pred)
L = L_delta + 0.05 * L_lowpass + 0.001 * L_tv
```

Stage 2 is allowed only after overfit32 and train128 pass:

```text
J = J0 + W * Delta_pred
L_output = L1(J, GT)
L_anchor_outside = mean((1 - W) * abs(J - J0))
L = L_delta + 0.2 * L_output + 0.05 * L_lowpass + 0.01 * L_anchor_outside
```

Hard FFT loss, SSIM loss, residual sparsity, and joint router training are not
part of the first v0.4A scout.

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| cached patch max diff | proves crop training uses full-image routing without recompute drift | Gate A random crop audit | `cache_usage_audit_*.csv` |
| trainable parameter counts | proves only LowFieldNet is trainable | Gate A | `preflight_noop_cache_*.json` |
| weighted delta L1 drop | asks whether the predictor learns the target | overfit32/train128 | `lowfield_overfit32_summary_*.json` |
| pred-target correlation | separates target learning from output PSNR noise | overfit32/train128 | `lowfield_overfit32_per_image_*.csv` |
| opened/closed gain | verifies correctability gating behavior | train128/test | `opened_closed_groups_*.csv` |
| residual amplitude audit | prevents hidden outside-mask changes | overfit32/train128 | `lowfield_amplitude_audit_*.json` |

## Required Intermediate Results

The next decision should retain these text artifacts:

1. sigma `3/5/7/11/15` summary table with mean, hard, easy, strong regressions, severe regressions, and opened-subset gain.
2. sigma `3.0` free-parameter low and correctability train-calibrated diagnostics.
3. low-field overfit32 per-image table with `anchor_psnr`, `M_safe_mean`, `P_benefit`, `target_abs_mean`, `pred_abs_mean`, `corr`, and `gain_delta`.
4. opened/closed group statistics for open hard, closed hard, open easy, and closed easy PSNR deltas.
5. low-field amplitude audit with residual mean, residual p95, lowpass consistency, and outside-mask residual max.
6. cache usage audit recording random batch patch max diff between cached patch and full tensor crop.

Only `.md`, `.txt`, `.log`, `.json`, `.csv`, and `.sh` artifacts should be
synced. Tensor caches, checkpoints, image outputs, NumPy arrays, and datasets
stay on ignored storage.

## Fair Run Contract

- Training or inference budget: Gate A no-op/cache, Gate B overfit32, Gate C train128/mini-val, then optional stop20 scout.
- Batch/sample policy: cache full images offline; training crops may only crop cached full-image tensors.
- Optimizer: AdamW for LowFieldNet, learning rate `1e-4` or `2e-4`.
- Schedule: stop20 scout only after Gate C.
- Loss weights: Stage 1 delta-only; Stage 2 may add light output loss after Gate B/C.
- Random seed policy: seed `3407` first; additional seeds only after a positive stop20.
- Evaluation cadence: every gate writes JSON/CSV before moving on.
- Hardware/runtime assumptions: AutoDL only; local work is compile/static verification.
- Allowed resume behavior: written checkpoint and schedule only; no silent horizon changes.
- Sample-size policy: overfit32 is diagnostic, train128 is pre-stop20, full test only for authorized scout.

## Stop20 Gate

If Gate C passes, the stop20 scout must satisfy:

| Metric | Pass line |
| --- | --- |
| mean PSNR delta | `>= +0.020 dB` |
| hard bottom-25% delta | `>= +0.080 dB` |
| easy top-25% delta | `>= -0.010 dB` |
| strong-reference regressions | `<= 30 / 250` |
| severe regressions | `<= 10 / 1000` |
| median PSNR delta | `>= 0` |
| SSIM delta | `>= 0` |
| true worst-10-image mean | `> -0.300 dB` |
| hard opened subset gain | `>= +0.150 dB` |
| closed easy subset absolute delta | `<= 0.005 dB` |

## Decision

- Decision label: `DO_NOT_RUN_STOP20_FROM_CURRENT_LOWFIELD_FORMS`.
- Image/global metric reason: v0.4 low-field diagnostics and ID-embedding evidence show target headroom, but deployable low-field forms do not learn enough under Gate B.
- Mechanism reason: cache exactness, correctability, and the delta loss are valid; the blocker is image-feature-to-lowfield mapping, not the target or gate.
- Preservation or regression reason: current forms keep strong/severe regressions at `0/0`, but their hard-sample gain and target recovery are far below the pass line.
- Cost/deployability reason: Gate C and stop20 are not cost-effective until a deployable residual expression passes overfit32 learnability.
- What this decides next: derive or initialize better bases from successful ID/free-parameter targets, then train image-to-basis weights plus a tiny local correction under the same Gate A/B protocol.

## Gate A/B Update

LowFieldNet-v1 passed Gate A but failed Gate B for both sigma `3.0` and sigma
`7.0`:

| Target | L1 drop | Corr | Recovery | Hard gain | Easy gain | Strong/severe |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sigma `3.0` | `0.0211` | `0.0072` | `0.0210` | `+0.0191 dB` | `-0.0037 dB` | `0/0` |
| sigma `7.0` | `0.0227` | `0.0107` | `0.0222` | `+0.0192 dB` | `-0.0042 dB` | `0/0` |

Do not run stop20 from LowFieldNet-v1. Since free-parameter low succeeds but
image-feature LowFieldNet fails, the next diagnostic should change residual
expression rather than return to color/detail: basis-mixture low-field or
physics-shaped veil residual.

## Failure-Branch Plan

Run the residual-form diagnostics under the same Gate A/B protocol:

| Branch | Continue condition | Decision use |
| --- | --- | --- |
| ID embedding | should pass if cache/gate/loss can express the target | if this fails, debug implementation/loss before any model change |
| Basis mixture | pass Gate B or clearly outperform LowFieldNet-v1 | promote basis-mixture to Gate C candidate |
| Basis + local residual | pass Gate B or clearly outperform pure basis | promote basis-local to Gate C candidate |
| Physics veil | pass Gate B or clearly outperform LowFieldNet-v1 with better structure | promote veil residual to Gate C candidate |

Do not return to color/detail branches unless these residual-form diagnostics
show no safe low-field expression path.

## Failure-Branch Results

| Branch | Verdict | L1 drop | Corr | Recovery | Hard gain | Easy gain | Strong/severe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ID embedding | pass | `0.7807` | `0.9712` | `0.9720` | `+1.2022 dB` | `+0.3935 dB` | `0/0` |
| Basis mixture | fail | `0.0717` | `0.3214` | `0.0870` | `+0.0834 dB` | `+0.0648 dB` | `0/0` |
| Basis + local | fail | `0.0791` | `0.3813` | `0.0979` | `+0.0920 dB` | `+0.0771 dB` | `0/0` |
| Physics veil | fail | `0.0437` | `0.2582` | `0.0375` | `+0.0490 dB` | `-0.0018 dB` | `0/0` |

The ID result proves the target and loss are learnable under the current
cache/gate protocol. The deployable residual forms do not pass Gate B, although
basis + local is the best non-ID signal. Do not run Gate C or stop20 yet. The
next design should derive bases from successful ID/free-parameter targets before
training image-to-basis weights.

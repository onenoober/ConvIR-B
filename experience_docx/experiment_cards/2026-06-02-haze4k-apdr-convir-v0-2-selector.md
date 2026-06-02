# Haze4K APDR-ConvIR v0.2 Selector

Date: 2026-06-02

Status: completed cloud selector-only preflight; failed selector gate, so
residual training was not launched.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-ConvIR, anchored on official ConvIR-B.
- Route name: APDR-v0.2 Global-Hard + Spatial-Risk Selector.
- Dataset or task: Haze4K dehazing.
- Primary objective: decide whether APDR can learn a deployable hard-case
  selector before any residual is trained.
- Execution environment: AutoDL `autodl-dehaze3`, `convir-cu128`.
- Artifact root: `experience_docx/experiment_logs/haze4k_apdr_v0_2_selector_20260602/`.
- Branch or isolated workspace: `codex/haze4k-apdr-convir-v0-2`.

## Baseline Contract

- Baseline implementation: original ConvIR-B Haze4K `--arch convir --version base`.
- Baseline checkpoint or initialization: official Haze4K ConvIR-B checkpoint
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Dataset and split: Haze4K train/test under
  `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Selector output contract: APDR output must remain exactly A0 during this
  stage; residual head is frozen and the model is built with force-zero output
  gating.
- Sample-size policy: full Haze4K test for selector gate.

## Most Valuable Attempt

- Why this is the highest-value next attempt: APDR-v0.1 fixed most
  preservation regressions but failed to create hard-case gains, showing that
  the next bottleneck is hard/easy selectivity rather than residual capacity.
- Target failure or opportunity: learn a deployable image-level hard selector
  and spatial risk selector from absolute A0 error thresholds.
- Cheap preflight evidence: architecture preflight plus selector-only training
  where output is forced equal to A0.
- Earliest decisive gate: selector-only full-test gate.
- What success decides: APDR-v0.2 may proceed to a residual stop20 stage using
  the learned selector mechanism.
- What failure decides: do not train APDR-v0.2 residuals; redesign the selector
  before spending residual training budget.

## Hypothesis

The v0.1 target normalized A0 error within each image, so even easy images could
receive high local risk. v0.2 instead trains:

```text
H_img = image-level hard selector from x, J0, decoder features, and RGB haze priors
S_pixel = spatial risk selector from x, J0, decoder features, and RGB haze priors
M = H_img * S_pixel
J = J0 during selector-only preflight
```

Targets:

```text
E0 = |J0 - GT|
hard_target = clamp((RMSE0 - q50_train) / (q90_train - q50_train), 0, 1)
spatial_target = sigmoid((E0 - q70_global_pixel_error) / tau)
```

## Change

- Code branch: `codex/haze4k-apdr-convir-v0-2`.
- Exact code/config change:
  - add APDR selector mode `v0_2` with separate `global_gate` and
    `spatial_gate`;
  - keep the legacy `v0` selector mode as default;
  - add a selector-only cloud preflight tool that trains only full-scale
    selector/context parameters and freezes the residual head;
  - fix APDR preflight naming by making `stage` configurable;
  - rename parameter stats from `apdr_v0` to `apdr_candidate`;
  - assert that `--apdr_loss_scales full_only` requires
    `--apdr_active_scales full`.
- Explicitly disabled mechanisms: residual training, depth prior,
  diffusion/teacher, hard FFT, PFD/RHFD, HSCM, PFFB, FAM modulation, and
  low-frequency veil.

## Gates

Selector-only gate:

| Rule | Required |
| --- | ---: |
| `mean(H_img | hard bottom-25%) / mean(H_img | easy top-25%)` | `>= 3.0` |
| `Spearman(H_img, A0_PSNR)` | `<= -0.45` |
| `AUC(hard vs easy by H_img)` | `>= 0.75` |
| `mean(H_img | strong-reference)` | `<= 0.05` |
| spatial-risk BCE on deterministic train subset | final `<` initial |
| zero-residual output max absolute diff vs A0 | `< 1e-6` |

Stop rule:

- If selector-only gate fails, stop v0.2 and do not train residual.
- If selector-only gate passes, write the residual-stage command and run the
  residual stop20 gate separately.

## Cloud Run Outcome

Cloud execution ran on AutoDL `autodl-dehaze3` in
`/root/autodl-tmp/workspace/ConvIR-B-apdr-convir-v0-2`.

Architecture preflight passed:

- Haze4K pair audit: `3000/3000` train and `1000/1000` test pairs.
- Official ConvIR-B checkpoint loaded exactly into A0; APDR only missed
  expected `APDR_*` keys.
- Zero-init equivalence: random and real-batch `max_abs_diff = 0.0`.
- APDR-v0.2 candidate parameters: `8,695,240`, delta `64,575`
  (`0.7482%`) vs official ConvIR-B.

Selector-only calibration:

- Train images: `3000`.
- Pixel samples: `6,144,000`.
- RMSE q50/q90: `0.0105008` / `0.0219539`.
- Pixel-error q70/q90: `0.0100390` / `0.0211587`.
- Spatial tau: `0.0111198`.

Selector-only full-test gate:

| Gate | Observed | Required | Result |
| --- | ---: | ---: | --- |
| hard/easy `H_img` ratio | `1.00245` | `>= 3.0` | fail |
| Spearman(`H_img`, A0 PSNR) | `-0.35373` | `<= -0.45` | fail |
| AUC hard vs easy by `H_img` | `0.768624` | `>= 0.75` | pass |
| strong-reference mean `H_img` | `0.0212881` | `<= 0.05` | pass |
| spatial BCE, deterministic subset | `2.06398 -> 0.729276` | final `<` initial | pass |
| zero-residual output max diff vs A0 | `0.0` | `< 1e-6` | pass |

Mechanism observations:

- Spatial-risk supervision learned a useful pixel-level target reduction.
- The image-level selector remained nearly flat in absolute magnitude:
  hard bottom-25% mean `H_img = 0.0213403`, easy top-25% mean
  `H_img = 0.0212881`.
- AUC barely passed, but the ratio and Spearman gates show that this is not a
  deployable hard/easy selector.
- Output remained exactly A0 throughout selector-only evaluation, so this is a
  clean selector diagnostic rather than a hidden residual run.

## Decision

- Decision label: `FAIL_STOP_APDR_V0_2_SELECTOR_ONLY`.
- Image/global metric reason: no residual output was trained or evaluated by
  design.
- Mechanism reason: `H_img` did not separate hard and easy images strongly
  enough; the hard/easy ratio was essentially `1.0`.
- Preservation reason: strong-reference mean `H_img` stayed low and
  zero-residual output matched A0 exactly.
- Cost/deployability reason: the selector is not strong enough to justify a
  residual stop20 run.
- What this decides next: do not launch APDR-v0.2 residual training. Future
  APDR work should redesign the image-level hard selector target or training
  objective before spending residual budget.

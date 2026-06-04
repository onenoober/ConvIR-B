# Haze4K APDR-ConvIR v0.2R Selector

Date: 2026-06-02

Status: completed cloud selector-only preflight; failed selector gate, so
residual training was not launched.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-ConvIR, anchored on official ConvIR-B.
- Route name: APDR-v0.2R Full-Image Calibrated Hard Router.
- Dataset or task: Haze4K dehazing.
- Primary objective: decide whether APDR can learn a deployable full-image
  hard-case budget before any residual is trained.
- Execution environment: AutoDL `autodl-dehaze3`, `convir-cu128`.
- Artifact root: `experience_docx/experiment_logs/haze4k_apdr_v0_2r_selector_20260602/`.
- Branch or isolated workspace: `codex/haze4k-apdr-convir-v0-2r-fullimage-router`.

## Baseline Contract

- Baseline implementation: original ConvIR-B Haze4K `--arch convir --version base`.
- Baseline checkpoint or initialization: official Haze4K ConvIR-B checkpoint
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Dataset and split: Haze4K train/test under
  `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Selector output contract: APDR output must remain exactly A0 during this
  stage; residual head is frozen and model output gating is force-zero.
- Sample-size policy: full Haze4K test for selector gate.

## Most Valuable Attempt

- Why this is the highest-value next attempt: APDR-v0.2 showed spatial BCE can
  learn, but `H_img` stayed nearly flat because the image-level selector was
  trained on random crop difficulty while the gate evaluated full-image A0
  difficulty.
- Target failure or opportunity: make the image-level hard budget learn
  full-image A0 risk before spending residual training budget.
- Cheap preflight evidence: architecture preflight plus selector-only Phase A/B
  training where residual output is forced equal to A0.
- Earliest decisive gate: selector-only full-test gate.
- What success decides: a residual stop20 stage may be justified using
  `J = J0 + B_img * S_pixel * bounded_residual`.
- What failure decides: do not train residuals; the hard-router mechanism is
  still not deployable.

## Hypothesis

v0.2 failed because its global selector target was effectively crop-level while
promotion gates are image-level. v0.2R separates the mechanisms:

```text
z_img = GlobalDensityRouter(full-image x, J0, RGB priors, |x-J0| stats, GAP(feature))
B_img = train-calibrated sigmoid((z_img - tau_train) / temperature_train)
S_pixel = SpatialRiskGate(local x, J0, feature, RGB priors)
M = B_img * S_pixel
J = J0 during selector-only preflight
```

Phase A trains only the global router on full-image A0 RMSE targets. Phase B
freezes the global router and trains only the spatial gate on global-thresholded
pixel error. Selector-only uses no gate sparsity penalty because preservation is
already enforced by zero residual output.

## Change

- Code branch: `codex/haze4k-apdr-convir-v0-2r-fullimage-router`.
- Exact code/config change:
  - add APDR selector mode `v0_2r`;
  - add `GlobalDensityRouter`, decoupled from the spatial context;
  - keep `v0` and `v0_2` selector modes available;
  - add a selector-only cloud preflight with Phase A global-router training,
    train-set budget calibration, Phase B spatial-gate training, and full-test
    gate JSON.
- Explicitly disabled mechanisms: residual training, depth prior,
  diffusion/teacher, hard FFT, PFD/RHFD, HSCM, PFFB, FAM modulation, and
  low-frequency veil.

## Gates

Selector-only gate:

| Rule | Required |
| --- | ---: |
| zero-residual output max absolute diff vs A0 | `< 1e-6` |
| deterministic full-image hard BCE | final `<= 0.55` and drop `>= 0.05` |
| AUC hard/easy by `z_img` | `>= 0.82` |
| Spearman(`z_img`, A0 PSNR) | `<= -0.50` |
| mean(`B_img` hard bottom-25%) | `>= 0.20` |
| mean(`B_img` easy top-25%) | `<= 0.05` |
| hard/easy `B_img` ratio | `>= 4.0` |
| spatial-risk BCE on deterministic train subset | final `<` initial |
| full-test mean spatial BCE | `<= 0.80` |

Stop rule:

- If selector-only gate fails, stop v0.2R and do not train residual.
- If selector-only gate passes, write and run a separate residual stop20 gate.

## Cloud Run Outcome

Cloud execution ran on AutoDL `autodl-dehaze3` in
`/root/autodl-tmp/workspace/ConvIR-B-apdr-convir-v0-2r-fullimage-router`.

Architecture preflight passed:

- Haze4K pair audit: `3000/3000` train and `1000/1000` test pairs.
- Official ConvIR-B checkpoint loaded exactly into A0; APDR only missed
  expected `APDR_*` keys.
- Zero-init equivalence: random and real-batch `max_abs_diff = 0.0`.
- APDR-v0.2R candidate parameters: `8,715,784`, delta `85,119`
  (`0.9862%`) vs official ConvIR-B.

Full-image calibration matched the previous A0 risk distribution:

- Train images: `3000`.
- Pixel samples: `6,144,000`.
- RMSE q25/q50/q75/q90: `0.0073429` / `0.0105008` / `0.0155625` /
  `0.0219539`.
- Pixel-error q70/q90: `0.0100390` / `0.0211587`.
- Spatial tau: `0.0111198`.

Selector-only full-test gate:

| Gate | Observed | Required | Result |
| --- | ---: | ---: | --- |
| zero-residual max diff vs A0 | `0.0` | `< 1e-6` | pass |
| deterministic full-image hard BCE | `1.70804 -> 0.630292` | final `<= 0.55`, drop `>= 0.05` | fail |
| AUC hard vs easy by `z_img` | `0.97664` | `>= 0.82` | pass |
| Spearman(`z_img`, A0 PSNR) | `-0.74664` | `<= -0.50` | pass |
| mean `B_img` hard bottom-25% | `0.782352` | `>= 0.20` | pass |
| mean `B_img` easy top-25% | `0.146157` | `<= 0.05` | fail |
| hard/easy `B_img` ratio | `5.35281` | `>= 4.0` | pass |
| spatial BCE, deterministic subset | `2.06208 -> 0.733602` | final `<` initial | pass |
| full-test mean spatial BCE | `0.757404` | `<= 0.80` | pass |

Mechanism observations:

- The full-image hard router fixed the v0.2 flat-selector failure: test AUC
  rose from `0.7686` to `0.9766`, and Spearman improved from `-0.3537` to
  `-0.7466`.
- Train-calibrated `B_img` separated hard/easy images by ratio, but the easy
  budget remained too open at `0.146`, above the `0.05` preservation bound.
- Spatial-risk supervision remained effective after decoupling, with full-test
  mean spatial BCE `0.7574`.
- Output stayed exactly equal to A0, so this is a clean selector diagnostic.

## Decision

- Decision label: `FAIL_STOP_APDR_V0_2R_SELECTOR_ONLY`.
- Main success: full-image global router learned a strong hard/easy ranking.
- Main failure: budget calibration is not conservative enough for
  strong-reference preservation, and deterministic hard BCE remains above the
  predeclared ceiling.
- Cost/deployability reason: do not launch residual training while easy
  `B_img` remains too high.
- What this decides next: APDR is still alive as a selector direction, but the
  next attempt must tighten budget calibration or train the router with an
  explicit low-easy-budget constraint before residual training.

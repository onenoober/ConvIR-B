# Haze4K B1r Decoder RHFD Preservation Rescue

Date: 2026-06-02

Status: preflight-ready rescue route; opened after B1 feature-delta RHFD failed preservation.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: ConvIR-B with PFD wrapper.
- Dataset or task: Haze4K dehazing.
- Primary objective: test a preservation-safe decoder-side RHFD adapter.
- Main metric: PSNR delta vs official ConvIR-B Haze4K checkpoint.
- Secondary metrics: SSIM delta, hard/easy bucket deltas, strong-reference regressions, severe regressions.
- Execution environment: AutoDL Haze4K workspace when available.
- Artifact root: `experience_docx/experiment_logs/haze4k_b1r_decoder_rhfd_preserve_20260602/`.
- Branch or isolated workspace: `codex/haze4k-b1r-decoder-rhfd-preserve`.
- Review package location: sync text evidence to `main` after run completion.

## Baseline Contract

- Baseline implementation: original ConvIR-B Haze4K `--arch convir --version base`.
- Baseline checkpoint or initialization: official Haze4K ConvIR-B checkpoint.
- Evaluation entrypoint: `experience_docx/tools/eval_haze4k_checkpoint_compare.py`.
- Training entrypoint: `Dehazing/ITS/main.py --mode train`.
- Dataset and split: Haze4K train/test under the audited dataset root.
- Preprocessing and decoding: repository Haze4K loader, factor-32 reflect padding during eval.
- Metric implementation: PyTorch PSNR, `pytorch_msssim.ssim`, per-image CSV.
- Reproduced baseline result: A0 official checkpoint was previously recorded near `34.14 dB / 0.98971`.
- Known reproduction gap: A1 stop20 full fine-tune fell to `33.4260 / 0.988607`.
- Checkpoint/export/resume contract: `--init_model` initializes weights; `--resume` is optimizer-state resume only.

## Most Valuable Attempt

- Why this is the highest-value next attempt: B1 moved hard cases but damaged easy and strong-reference images; adding HSCM/PFFB would compound an unresolved preservation failure.
- Target failure or opportunity: convert RHFD from encoder-side feature delta into true decoder-side residual feedback.
- Cheap preflight evidence: zero-init equivalence against official ConvIR-B, checkpoint load, trainable adapter count, finite one-batch backward.
- Earliest decisive gate: adapter-only stop10 preservation screen, then stop20 B1r preservation gate.
- Expected cost or attempt-count saving: one isolated rescue route instead of B2/B3 module stacking.
- What success decides: decoder RHFD-Lite is worth promotion or teacher-preserve follow-up.
- What failure decides: RHFD direction is diagnostic under the current evidence budget; move to thesis negative-evidence synthesis rather than stacking modules.
- Why a cheaper diagnostic is not enough: B1 already proved hard cases can move; the unresolved question is whether adapter-only decoder feedback can preserve A0 quality during training.

## Hypothesis

- Observed failure: B1 feature-side RHFD improved hard bottom-25% by `+0.3838 dB` but produced global `-0.0885 dB`, easy top-25% `-0.3345 dB`, strong-reference regressions `137/250`, and severe regressions `434/1000`.
- Target mechanism: decoder-side residual haze feedback from low-scale restored outputs to the next decoder stage.
- Primary variable: `--pfd_decoder_rhfd 1` with adapter-only training.

Mechanism sentence:

```text
If residual haze feedback is injected only into decoder features with detached low-scale outputs and zero-init feedback projections, hard-case residual correction should improve while preserving official ConvIR-B strong/easy cases.
```

## Change

- Code branch: `codex/haze4k-b1r-decoder-rhfd-preserve`.
- Exact code/config change:
  - add `DecoderResidualHazeFeedback` in `Dehazing/ITS/models/pfd_modules.py`;
  - add `--pfd_decoder_rhfd` and fixed `--pfd_decoder_rhfd_scale 0.1`;
  - feed `1/4` output residual to the `1/2` decoder feature;
  - feed `1/2` output residual to the full-resolution decoder feature;
  - use `.detach()` on restored output and residual inputs to the feedback modules;
  - add `--pfd_adapter_only` so only enabled PFD adapter modules are trainable.
- Enabled mechanisms: decoder RHFD-Lite only.
- Explicitly disabled mechanisms: feature-side `--pfd_rhfd`, HSCM, PFFB, PFFB-high, teacher preservation, hard-frequency loss, haze-prior auxiliary loss.
- Parameter/runtime/memory impact expected: small adapter overhead; measured in preflight.
- Initialization or no-op behavior: final feedback projection is zero-init; output equivalence must match ConvIR-B at FP32 thresholds.
- Resume policy: first runs start from official checkpoint through `--init_model`; no optimizer resume unless continuing the same adapter-only model.
- Defaults changed: none for original `--arch convir`; B1r is opt-in.
- Defaults intentionally preserved: original L1 + `0.1` FFT loss, Haze4K crop/loader, validation cadence, official checkpoint baseline.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| pair audit | no fatal missing GT or duplicate target mapping | pending |
| checkpoint load | original keys load, only `PFD_` keys missing in candidate | pending |
| zero-init equivalence | random and real-batch max diff `< 1e-6` | pending |
| finite forward/backward | one-batch adapter-only backward finite | pending |
| trainable parameters | only `PFD_DECODER_RHFD*` trainable under `--pfd_adapter_only` | pending |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| hard bottom-25% PSNR delta vs A0 | direct target group | full Haze4K test | bucket JSON |
| easy top-25% PSNR delta vs A0 | preservation target | full Haze4K test | bucket JSON |
| strong-reference regression count | protects already-good images | A0 top-25% PSNR | compare JSON |
| severe regression count | catches collapse | full Haze4K test | compare JSON |
| mean SSIM delta | guards structural quality | full Haze4K test | compare JSON |

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| A0 official checkpoint | avoid the A1 stop20 degradation baseline | B1r must compare directly against A0 |
| zero-init equivalence | proves adapter starts as no-op | max diff `< 1e-6` |
| adapter-only freeze | prevents full-backbone drift | only `PFD_DECODER_RHFD*` trainable |

## Fair Run Contract

- Training or inference budget: preflight, A0 eval through compare tool, B1r adapter-only stop10, B1r adapter-only stop20 only if stop10 is not catastrophic.
- Batch/sample policy: batch size 8 unless hardware forces a recorded change.
- Optimizer: Adam, LR `1e-4`, repository scheduler.
- Schedule: `num_epoch 1000`, `stop_epoch 10` or `20`, so LR remains comparable to earlier stop20 scouts.
- Loss weights: original L1 + `0.1` FFT.
- Random seed policy: first seed `3407`.
- Evaluation cadence: every epoch during training; full compare after stop10 and stop20.
- Checkpoint cadence: save every 5 epochs and keep `Best.pkl`.
- Hardware/runtime assumptions: AutoDL CUDA environment from previous Haze4K runs.
- Allowed resume behavior: no cross-route resume; optional same-model resume only.
- Sample-size policy: full Haze4K test for gate.

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| preflight | zero-init equivalence and checkpoint load pass | trainable adapter count finite | stop on any failure |
| stop10 screen | record A0-vs-B1r per-image deltas | look for no catastrophic preservation loss | continue to stop20 only if review remains plausible |
| stop20 B1r gate | global PSNR delta `>= -0.05`, SSIM delta `>= -0.001` | hard bottom-25% delta `>= +0.15` | pass only if easy delta `>= -0.05`, strong regressions `<= 50/250`, severe regressions `<= 150/1000` |
| promotion | only after stop20 gate pass | decoder feedback has clear hard-case gain without strong/easy collapse | then consider 80 epoch or 2-seed confirmation |

## Analysis Plan

- Per-sample or subgroup analysis: hard/easy by A0 PSNR, worst regressions, best gains.
- Visual or qualitative analysis: only after stop20 gate produces plausible candidate.
- Complexity analysis: parameter delta and trainable adapter count in preflight.
- Robustness or held-out analysis: not before stop20 pass.
- Regression analysis: strong-reference and severe regression counts.
- Required docs to update: route card, experiment index, run README, status text.
- Required artifacts to retain: `.md`, `.sh`, `.log`, `.txt`, `.json`, `.csv`.
- Required artifacts to delete or keep external: checkpoints, images, datasets, arrays.
- Evidence package contents: run script, preflight JSON, train logs, compare/bucket/gate JSON, per-image CSV, status.
- Evidence package audit: local/remote text-only parity after sync to `main`.

## Decision

- Decision label: pending.
- Image/global metric reason: pending.
- Mechanism reason: pending.
- Preservation or regression reason: pending.
- Cost/deployability reason: pending.
- What this decides next: B1r pass leads to promotion confirmation or teacher-preserve; B1r fail stops PFD expansion and supports negative-evidence thesis synthesis.

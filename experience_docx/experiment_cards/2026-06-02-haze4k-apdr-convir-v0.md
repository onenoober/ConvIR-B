# Haze4K APDR-ConvIR v0

Date: 2026-06-02

Status: completed cloud stop20 scout; failed promotion gate.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-ConvIR, anchored on official ConvIR-B.
- Dataset or task: Haze4K dehazing.
- Primary objective: test whether an output-level, bounded, gated residual
  adapter can improve hard samples without repeating the easy/strong-reference
  regressions seen in FAM, hard-frequency, haze-prior SCM, PFD, and B1r.
- Main metric: PSNR delta vs official ConvIR-B Haze4K checkpoint.
- Secondary metrics: SSIM delta, hard/easy bucket deltas, median delta,
  strong-reference regressions, severe regressions, APDR gate/residual stats.
- Execution environment: AutoDL `autodl-dehaze3`, `convir-cu128`.
- Artifact root: `experience_docx/experiment_logs/haze4k_apdr_v0_20260602/`.
- Branch or isolated workspace: `codex/haze4k-apdr-convir-v0`.
- Review package location: text evidence under
  `experience_docx/experiment_logs/haze4k_apdr_v0_20260602/`; sync to `main`
  during evidence consolidation.

## Baseline Contract

- Baseline implementation: original ConvIR-B Haze4K `--arch convir --version base`.
- Baseline checkpoint or initialization: official Haze4K ConvIR-B checkpoint
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Evaluation entrypoint: `experience_docx/tools/eval_haze4k_checkpoint_compare.py`.
- Training entrypoint: `Dehazing/ITS/main.py --mode train`.
- Dataset and split: Haze4K train/test under
  `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Preprocessing and decoding: repository Haze4K loader, factor-32 reflect padding during eval.
- Metric implementation: PyTorch PSNR, `pytorch_msssim.ssim`, per-image CSV.
- Reproduced baseline result: A0 official checkpoint previously recorded near
  `34.14 dB / 0.98971`.
- Checkpoint/export/resume contract: `--init_model` initializes weights;
  `--resume` is optimizer-state resume only.

## Most Valuable Attempt

- Why this is the highest-value next attempt: existing internal feature changes
  repeatedly moved hard samples but damaged easy/strong-reference cases.
  APDR keeps `J0` as an explicit output anchor and moves the new variable to a
  bounded, local, gated residual path.
- Target failure or opportunity: hard-sample under-correction without global
  feature drift.
- Cheap preflight evidence: official checkpoint load, pair audit, zero-init
  equivalence, trainable APDR parameter count, finite adapter-only backward.
- Earliest decisive gate: APDR-v0 adapter-only stop20.
- Expected cost or attempt-count saving: one preflight plus one stop20 run
  decides whether output-level anchor residuals are worth 80/full budget.
- What success decides: APDR-v0 deserves an 80-epoch promotion run and possible
  low-frequency veil v1.
- What failure decides: output-level APDR-v0 is diagnostic only; do not add
  depth, diffusion, or low-frequency veil until the failure mode is understood.
- Why a cheaper diagnostic is not enough: the key question is training behavior
  under official-checkpoint warm start and full Haze4K test gates.

## Hypothesis

- Observed failure: prior routes found hard-case signal but harmed preservation:
  FAM2-only hard `+0.8159 dB` with easy `-0.2860 dB`, hard-frequency hard
  `+0.5999 dB` with easy `-1.2363 dB`, haze-prior SCM hard `+0.3501 dB` with
  easy `-1.6511 dB`, SafeRHFD-v2 hard only `+0.04890 dB` with strong
  regressions `70/250`.
- Target mechanism: preserve official ConvIR-B output as `J0` and learn only a
  bounded local residual where RGB haze priors and decoder features indicate
  under-correction.
- Primary variable: `--arch apdr --apdr_train_scope apdr_only`.

Mechanism sentence:

```text
If ConvIR-B output is preserved as J0 and a zero-init APDR branch predicts only
bounded, gated output residuals from decoder features plus RGB haze priors, hard
Haze4K samples should improve while easy and strong-reference images stay near
the official checkpoint.
```

## Change

- Code branch: `codex/haze4k-apdr-convir-v0`.
- Exact code/config change:
  - add `Dehazing/ITS/models/APDRConvIR.py`;
  - add `Dehazing/ITS/models/apdr_modules.py`;
  - add `--arch apdr`, `--init_model`, and APDR flags in `Dehazing/ITS/main.py`;
  - add `--apdr_train_scope apdr_only` plus optional anchor/gate/residual
    regularizers in `Dehazing/ITS/train.py`.
- Enabled mechanisms: RGB haze prior encoder, decoder-feature context,
  zero-init output residual heads, spatial confidence gates, bounded residual.
- Explicitly disabled mechanisms: depth prior, diffusion/teacher, hard FFT
  loss, HSCM, PFD/RHFD, PFFB, FAM modulation.
- Initialization or no-op behavior: APDR residual heads are zero-init; output
  must match ConvIR-B at `max_abs_diff < 1e-6`.
- Defaults changed: none for original `--arch convir`; APDR is opt-in.
- Defaults intentionally preserved: original L1 + `0.1` FFT loss, Haze4K
  loader/crop/eval, official checkpoint baseline.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| pair audit | no fatal missing GT or duplicate target mapping | pass: `3000/3000` train and `1000/1000` test pairs |
| checkpoint load | original keys load, only `APDR_` keys missing in candidate | pass: no original missing/unexpected keys; APDR missing keys all under `APDR_` |
| zero-init equivalence | random and real-batch max diff `< 1e-6` | pass: random and real-batch `max_abs_diff = 0.0` |
| finite backward | one-batch APDR-only backward finite | pass: finite loss `0.0203865`, no nonfinite gradients |
| trainable parameters | only `APDR_*` trainable under `--apdr_train_scope apdr_only` | pass: `62,700` trainable APDR params, `8,630,665` frozen backbone params |

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| preflight | checkpoint load, data audit, zero-init, finite backward pass | trainable APDR count finite and nonzero | stop on any failure |
| stop20 | mean PSNR delta `>= +0.020`, SSIM delta `>= 0`, median delta `>= 0` | hard bottom-25% delta `>= +0.080` | pass only if easy top-25% `>= -0.010`, strong regressions `<= 30/250`, severe regressions `<= 10/1000`, worst10 mean `> -0.300` |
| promotion | only after stop20 pass | APDR gate/residual stats active and non-collapsed | then consider 80 epoch or v1 low-frequency veil |

## Fair Run Contract

- Training budget: APDR-v0 adapter-only stop20, seed `3407`.
- Batch/sample policy: batch size `8`, full Haze4K train/test.
- Optimizer: Adam, LR `1e-4` for APDR adapter-only.
- Schedule: `num_epoch 1000`, `stop_epoch 20`, preserving prior stop20 LR horizon.
- Loss weights: original L1 + `0.1` FFT; APDR regularizers default to zero in v0.
- Evaluation cadence: every epoch during training; full compare after stop20.
- Checkpoint cadence: save every 5 epochs and keep `Best.pkl`.
- Hardware/runtime assumptions: AutoDL RTX 5090, `convir-cu128`.
- Allowed resume behavior: no cross-route resume.
- Sample-size policy: full Haze4K test for gate.

## Cloud Run Outcome

Cloud execution ran on AutoDL `autodl-dehaze3` in
`/root/autodl-tmp/workspace/ConvIR-B-apdr-convir-v0`.

Run notes:

- First stop20 launch passed preflight, then crashed after epoch 1 validation
  inside `collect_apdr_stats` because the stats path forwarded unpadded
  validation images into ConvIR internals. Training/eval padding was unchanged.
- The fix was limited to padding APDR stats samples to factor 32 with reflect
  padding before `collect_apdr_stats`; the model, loss, schedule, and gate
  contract were not changed.
- The crashed log and result directory were archived with suffix
  `crash_apdr_stats_padding_20260602_1747`.
- The corrected run completed stop20 and produced full A0-vs-APDR test
  compare, per-image CSV, bucket analysis, and gate JSON.

Stop20 Best vs A0:

| Metric | Observed | Gate | Result |
| --- | ---: | --- | --- |
| A0 mean PSNR | `34.1406455 dB` | baseline | reference |
| APDR mean PSNR | `34.1339912 dB` | compare | lower |
| mean PSNR delta | `-0.0066543 dB` | `>= +0.020` | fail |
| mean SSIM delta | `+0.0000197` | `>= 0` | pass |
| median PSNR delta | `-0.0066500 dB` | `>= 0` | fail |
| hard bottom-25% delta | `-0.0009749 dB` | `>= +0.080` | fail |
| easy top-25% delta | `-0.0150923 dB` | `>= -0.010` | fail |
| strong-reference regressions | `100/250` | `<= 30/250` | fail |
| severe regressions | `24/1000` | `<= 10/1000` | fail |
| worst10 mean delta | `-0.1759460 dB` | `> -0.300` | pass |

Mechanism observations:

- APDR zero-init and official anchor loading worked exactly in preflight.
- APDR stats were active during training; by epoch 20, gate means were about
  `0.0339` full, `0.0307` half, and `0.0514` quarter scale.
- Residual magnitudes stayed bounded, with epoch-20 residual absolute means
  about `0.00056`, `0.00049`, and `0.00080` across full/half/quarter scales.
- Activity was not enough to improve hard samples, while easy and
  strong-reference preservation still regressed beyond the stop gate.

## Decision

- Decision label: `FAIL_STOP_APDR_V0_ADAPTER_ONLY`.
- Image/global metric reason: mean PSNR delta was negative
  (`-0.0066543 dB`) and median delta was negative, so APDR-v0 did not improve
  the official ConvIR-B anchor.
- Mechanism reason: the anchor-preserved residual mechanism was technically
  valid and active, but the learned residual did not produce the required hard
  bottom-25% gain (`-0.0009749 dB` observed vs `+0.080 dB` required).
- Preservation or regression reason: easy top-25% delta failed
  (`-0.0150923 dB`), strong-reference regressions were too high (`100/250`),
  and severe regressions exceeded the limit (`24/1000`).
- Cost/deployability reason: APDR adds `62,700` parameters and extra inference
  cost without passing any key quality/preservation gate; do not spend 80/full
  budget on this exact v0 route.
- What this decides next: keep APDR-v0 as diagnostic evidence only. Do not add
  depth, diffusion, low-frequency veil, or hard-frequency components on top of
  this failed v0 without a separate mechanism change that first addresses
  strong-reference preservation and hard-gain selectivity.

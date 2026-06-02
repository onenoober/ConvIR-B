# Haze4K APDR-ConvIR v0.1

Date: 2026-06-02

Status: completed cloud stop20 scout; failed promotion gate.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-ConvIR, anchored on official ConvIR-B.
- Dataset or task: Haze4K dehazing.
- Primary objective: test whether adding training-time A0-risk/no-degrade
  constraints makes APDR selective enough to protect easy and strong-reference
  images while still allowing hard-case correction.
- Main metric: PSNR delta vs official ConvIR-B Haze4K checkpoint.
- Secondary metrics: SSIM delta, hard/easy bucket deltas, median delta,
  strong-reference regressions, severe regressions, true worst-10-image mean
  delta, APDR gate/residual stats.
- Execution environment: AutoDL `autodl-dehaze3`, `convir-cu128`.
- Artifact root: `experience_docx/experiment_logs/haze4k_apdr_v0_1_20260602/`.
- Branch or isolated workspace: `codex/haze4k-apdr-convir-v0-1`.

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
- Checkpoint/export/resume contract: `--init_model` initializes weights;
  `--resume` is optimizer-state resume only.

## Most Valuable Attempt

- Why this is the highest-value next attempt: APDR-v0 proved zero-init
  anchoring, adapter-only training, and bounded residual plumbing work, but it
  failed because the training objective did not keep the candidate close to A0
  in low-risk regions and the gate had no direct selectivity signal.
- Target failure or opportunity: distinguish hard under-correction from easy
  over-modification before adding any larger prior or residual mechanism.
- Cheap preflight evidence: official checkpoint load, pair audit, zero-init
  equivalence, APDR full-branch trainable parameter count, finite backward.
- Earliest decisive gate: APDR-v0.1 anchor-risk stop20.
- Expected cost or attempt-count saving: one preflight plus one stop20 run
  decides whether APDR should continue as an anchor-risk route or be closed.
- What success decides: A0-risk supervised APDR deserves a second, tuned
  preservation scout or an 80-epoch promotion candidate.
- What failure decides: APDR remains diagnostic-only; do not add depth,
  diffusion, low-frequency veil, or hard-frequency components until a stronger
  deployable selector is designed.

## Hypothesis

- Observed v0 failure: APDR-v0 kept an explicit output anchor at initialization
  but trained with original reconstruction loss only. Mean PSNR fell
  `-0.0066543 dB`, hard bottom-25% moved `-0.0009749 dB`, easy top-25% moved
  `-0.0150923 dB`, strong-reference regressions were `100/250`, and severe
  regressions were `24/1000`.
- Target mechanism: use A0 output error against GT as a training-time risk map.
  Low A0-error pixels receive an anchor/no-degrade penalty, while the gate is
  directly supervised toward high A0-error pixels.
- Primary variable: `--arch apdr --apdr_active_scales full
  --apdr_loss_scales full_only` plus nonzero anchor/gate/residual penalties.

Mechanism sentence:

```text
If the APDR residual is trained only on the full output and receives an A0-risk
map during training, then the gate should close in A0-safe regions and open only
where the official checkpoint is measurably under-corrected, reducing
easy/strong-reference regressions relative to APDR-v0.
```

## Change

- Code branch: `codex/haze4k-apdr-convir-v0-1`.
- Exact code/config change:
  - add `--apdr_active_scales` so APDR can train only the full-scale branch;
  - add `--apdr_loss_scales full_only` so quarter/half outputs do not drive
    the v0.1 objective;
  - add `--apdr_gate_supervision_lambda` and `--apdr_risk_temperature`;
  - compute A0-risk anchor loss from `|J0 - GT|` with
    `w_safe = exp(-k * normalize(|J0 - GT|))`;
  - compute direct gate supervision against normalized A0 error;
  - rename evaluation tail metrics into `worst10pct` and `worst10img` fields;
  - make the stop20 gate use the true worst-10-image mean from bucket JSON.
- Enabled mechanisms: RGB haze prior encoder, decoder-feature context,
  zero-init full-output residual head, spatial confidence gate, bounded
  residual, A0-risk anchor/no-degrade loss, A0-risk gate supervision.
- Explicitly disabled mechanisms: depth prior, diffusion/teacher, hard FFT
  loss, HSCM, PFD/RHFD, PFFB, FAM modulation, low-frequency veil.
- Initialization or no-op behavior: APDR residual heads are zero-init; output
  must match ConvIR-B at `max_abs_diff < 1e-6`.
- Defaults changed: none for original `--arch convir`; APDR remains opt-in.

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| preflight | checkpoint load, data audit, zero-init, finite backward pass | only full APDR branch trainable and finite | stop on any failure |
| stop20 | mean PSNR delta `>= +0.020`, SSIM delta `>= 0`, median delta `>= 0` | hard bottom-25% delta `>= +0.080` and gate/residual stats active | pass only if easy top-25% `>= -0.010`, strong regressions `<= 30/250`, severe regressions `<= 10/1000`, true worst-10-image mean `> -0.300` |
| promotion | only after stop20 pass | preservation and selectivity both improve over v0 | then consider longer budget |

## Fair Run Contract

- Training budget: APDR-v0.1 anchor-risk stop20, seed `3407`.
- Batch/sample policy: batch size `8`, full Haze4K train/test.
- Optimizer: Adam, LR `1e-4` for APDR full-branch adapter-only.
- Schedule: `num_epoch 1000`, `stop_epoch 20`, preserving prior stop20 LR horizon.
- Loss weights: full-output L1 + `0.1` FFT plus
  `apdr_anchor_lambda=0.10`, `apdr_gate_supervision_lambda=0.02`,
  `apdr_gate_lambda=0.002`, `apdr_residual_lambda=0.02`.
- Gate init: `apdr_gate_init=0.01`.
- A0-risk temperature: `apdr_risk_temperature=5.0`.
- Evaluation cadence: every epoch during training; full compare after stop20.
- Checkpoint cadence: save every 5 epochs and keep `Best.pkl`.
- Hardware/runtime assumptions: AutoDL RTX 5090, `convir-cu128`.
- Allowed resume behavior: no cross-route resume.
- Sample-size policy: full Haze4K test for gate.

## Cloud Run Outcome

Cloud execution ran on AutoDL `autodl-dehaze3` in
`/root/autodl-tmp/workspace/ConvIR-B-apdr-convir-v0-1`.

Stop20 Best vs A0:

| Metric | Observed | Gate | Result |
| --- | ---: | --- | --- |
| mean PSNR delta | `+0.0001148 dB` | `>= +0.020` | fail |
| mean SSIM delta | `+0.00000335` | `>= 0` | pass |
| median PSNR delta | `+0.0003586 dB` | `>= 0` | pass |
| hard bottom-25% delta | `+0.0006680 dB` | `>= +0.080` | fail |
| easy top-25% delta | `-0.0010661 dB` | `>= -0.010` | pass |
| strong-reference regressions | `1/250` | `<= 30/250` | pass |
| severe regressions | `0/1000` | `<= 10/1000` | pass |
| true worst-10-image mean delta | `-0.0449356 dB` | `> -0.300` | pass |

Mechanism observations:

- The A0-risk/no-degrade objective changed the failure mode relative to v0:
  easy, strong-reference, severe-regression, median, and true worst-10-image
  gates all passed.
- The hard bottom-25% delta was only `+0.0006680 dB`, far below the required
  `+0.080 dB`, so v0.1 did not create a useful hard-case correction mechanism.
- The active full-scale APDR branch stayed bounded. By epoch 20, full gate mean
  was in the low `0.04` range and quarter/half scales remained inactive.
- Preflight confirmed direct gate supervision gives the full gate head
  nonzero gradient at initialization; however, that was not enough to learn a
  deployable hard selector from the current A0-error target.

## Decision

- Decision label: `FAIL_STOP_APDR_V0_1_ANCHOR_RISK_HARD_GAIN`.
- Image/global metric reason: mean PSNR delta was essentially flat
  (`+0.0001148 dB`) and below the required `+0.020 dB`.
- Mechanism reason: A0-risk supervision improved preservation but did not
  produce the required hard bottom-25% gain.
- Preservation reason: v0.1 fixed the major v0 preservation failure, with
  easy top-25% `-0.0010661 dB`, strong-reference regressions `1/250`, severe
  regressions `0/1000`, and true worst-10-image mean `-0.0449356 dB`.
- Cost/deployability reason: full-branch APDR adds active parameters and
  inference cost without passing hard-gain or mean-improvement gates.
- What this decides next: do not promote v0.1 to 80/full. Do not simply raise
  residual or gate strength; any next APDR route needs a stronger hard-case
  selector than normalized A0-error alone.

## Decision Boundary

- Promote only if the predeclared stop20 gate passes.
- If global mean improves but strong-reference or severe regressions still
  fail, keep v0.1 diagnostic-only.
- If preservation improves but hard bottom-25% gain remains absent, do not add
  larger residual modules; first decide whether APDR has a usable hard selector.

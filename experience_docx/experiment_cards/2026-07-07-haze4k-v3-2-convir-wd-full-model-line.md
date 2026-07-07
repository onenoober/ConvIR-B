# Haze4K v3.2 ConvIR-WD Full Model Line

Purpose: start a full-model dehazing line after v3.1 showed real standalone
full-model headroom and v3.0 closed A0-anchored safe-upgrade rescue.

Route identity: new model-structure route from
`github/codex/haze4k-official-arch-anchor`, not a v3.0 continuation.

Fact sources:
- GitHub `main` at `7d179ef` for v3.1 full-model bakeoff evidence.
- GitHub `main` `family_summaries/full_model_line_family_summary.md`.
- `Haze4K_ARCH_FINETUNE_WORKFLOW.md`.
- Cloud runtime state on `convir-4090`.

Hypothesis: an end-to-end ConvIR model with internal wavelet/low-frequency haze
state modulation can become a better full-model baseline than ConvIR-B without
requiring per-image A0 output dominance.

Material difference from stopped routes:
- not A0 output residual;
- not A0+teacher alpha;
- not selector/no-op deployment;
- not bridge/generator;
- not v3.0 partial-unfreeze rescue;
- primary gate is model-line success, not strict A0 per-image dominance.

Architecture:
- `--arch convir_wd_lite`;
- official ConvIR-B backbone path remains available through
  `--arch official_convir` / `--arch convir`;
- two-level differentiable Haar/DWT-style state from the input;
- WD state encoder at H/4 resolution;
- neutral-init feature modulation in the bottleneck and decoder fusion path;
- no final-output postprocess and no A0 residual head.

New parameter prefixes:
- `WD_state_encoder.*`;
- `WD_bottleneck_mod.*`;
- `WD_decoder2_mod.*`;
- `WD_decoder1_mod.*`.

Partial-load rule:
- official ConvIR-B keys from
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
  must load by exact key and shape;
- missing keys are allowed only when they start with `WD_`;
- unexpected checkpoint keys are fatal;
- official shape mismatches are fatal.

Initialization:
- WD state encoder uses normal ConvIR `BasicConv` initialization;
- every WD modulation projection is zero-initialized;
- Stage-0 expects exact no-op output equivalence to A0 after partial-load.

Frozen/trainable scopes:
- `wd_only`: train only `WD_*` parameters.
- `wd_decoder`: train `WD_*`, `Decoder.*`, `Convs.*`, `ConvsOut.*`, and
  decoder-side `feat_extract.3/4/5`.
- `all`: reserved for later full fine-tuning after internal validation.

Stage ladder:
- P0 architecture/preflight: strict partial-load, finite synthetic and one
  train-batch forward, no-op vs A0, trainable manifest, no locked test.
- P1 mini-overfit sanity: 8 train-derived center crops, `wd_decoder`, fixed
  seed, finite outputs, loss ratio `<= 0.95`, WD activity increases; no quality
  claim.
- P2 train-derived validation: only after P1 pass and written authorization;
  use a real train-derived split such as 480/120 or 5x larger train folds;
  checkpoint selection only on this split.
- P3 fixed internal confirmation: only after P2 pass with fixed candidate.
- P4 locked test: one-shot confirmation only after fixed candidate selection.

Metric contract:
- Baseline for future quality claims: official ConvIR-B A0 same split/context.
- P0 metric: max absolute no-op difference vs A0 must be `0.0`.
- P1 metric: numerical/trainability only; no PSNR/SSIM promotion claim.
- P2 model-line gate target: mean delta `>= +0.30 dB`, hard delta
  `>= +0.50 dB`, easy delta `>= -0.05 dB`, p05 `>= -0.30 dB`, CVaR5
  `>= -0.50 dB`, no catastrophic visual failures, and Pareto-competitive
  absolute PSNR/SSIM versus v3.1 standalone candidates.

Forbidden:
- no locked test in P0/P1/P2;
- no canary80 as a shortcut;
- no threshold/checkpoint selection from locked test;
- no A0 residual, selector, alpha, bridge, or generator;
- no v3.0 rescue by more decoder unfreezing, samples, folds, or loss tuning.

Evidence root:
`experience_docx/experiment_logs/haze4k_v3_2_convir_wd_full_model_line_20260707/`.

P0/P1 result:
- P0 passed from route commit `478ac83`: partial-load loaded `602` official
  keys, allowed `24` WD new keys, no-op max abs vs A0 was `0.0`, one train
  batch was finite, and locked test was untouched.
- P1 passed from route commit `35758db`: `8` train-derived center crops at
  crop size `256`, `wd_decoder` scope, loss `0.01778930053114891 ->
  0.012172756716609001`, loss ratio `0.6842740497466221`, WD activity delta
  `0.007103331430698745`, finite outputs, and locked test untouched.

Current status:
`COMPLETED_P0_P1_GATE_PASS_P2_DESIGN_OPEN_LOCKED_TEST_BLOCKED`.

Next action: write the P2 train-derived validation design before any larger
training. P1 is not quality evidence, and locked test remains blocked.

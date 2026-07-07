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
  claim. The corrected pass criterion is P1b aggregate: initial/final loss,
  output finite check, and WD activity must be measured over all loaded crops,
  not just the first mini-batch.
- P2 train-derived validation: authorized only after P1b aggregate pass and a
  written design. The fixed P2 screen uses the v3.1 600-image train-derived
  table: `fold_id=0` as the 120-image validation split and `fold_id=1..4` as
  the 480-image training split. Checkpoint selection is only by this
  train-derived validation split.
- P3 fixed internal confirmation: only after P2 pass with fixed candidate.
- P4 locked test: one-shot confirmation only after fixed candidate selection.

Metric contract:
- Baseline for future quality claims: official ConvIR-B A0 same split/context.
- P0 metric: max absolute no-op difference vs A0 must be `0.0`.
- P1/P1b metric: numerical/trainability only; no PSNR/SSIM promotion claim.
  P1b aggregate over all loaded train-derived crops supersedes the original
  first-mini-batch P1 gate for deciding whether P1 is sufficient.
- P2 model-line gate target: mean delta `>= +0.30 dB`, hard delta
  `>= +0.50 dB`, easy delta `>= -0.05 dB`, p05 `>= -0.30 dB`, CVaR5
  `>= -0.50 dB`, no catastrophic visual failures, and Pareto-competitive
  absolute PSNR/SSIM versus v3.1 standalone candidates.
- P2 fixed screen hyperparameters: `wd_decoder`, seed `3407`, 20 epochs,
  batch size `4`, WD LR `2e-4`, decoder LR `1e-5`, grad clip `0.01`,
  validation/save every 5 epochs. Primary checkpoint is `Best.pkl` selected by
  P2 validation PSNR; `Final.pkl` is supportive only.

Forbidden:
- no locked test in P0/P1/P1b/P2;
- no canary80 as a shortcut;
- no threshold/checkpoint selection from locked test;
- no A0 residual, selector, alpha, bridge, or generator;
- no v3.0 rescue by more decoder unfreezing, samples, folds, or loss tuning.

Evidence root:
`experience_docx/experiment_logs/haze4k_v3_2_convir_wd_full_model_line_20260707/`.

P0/P1/P1b result:
- P0 passed from route commit `478ac83`: partial-load loaded `602` official
  keys, allowed `24` WD new keys, no-op max abs vs A0 was `0.0`, one train
  batch was finite, and locked test was untouched.
- Original P1 from route commit `35758db` trained over `8` train-derived center
  crops at crop size `256`, but its initial/final gate and WD activity were
  measured only on `inputs[:batch_size]`. It is retained as a historical
  trainability sanity, not the final aggregate gate.
- P1b aggregate passed from route commit `31fbb01`: `8` train-derived center
  crops at crop size `256`, `wd_decoder` scope, all-sample aggregate loss
  `0.01766193099319935 -> 0.013848769944161177`, loss ratio
  `0.7841028225902132`, WD activity delta `0.005941152640540774`, finite
  outputs, and locked test untouched.
- Historical P1 measured on the first mini-batch reported:
  crop size `256`, `wd_decoder` scope, loss `0.01778930053114891 ->
  0.012172756716609001`, loss ratio `0.6842740497466221`, WD activity delta
  `0.007103331430698745`, finite outputs, and locked test untouched.

Current status:
`COMPLETED_P0_P1B_AGGREGATE_GATE_PASS_P2_DESIGN_OPEN_LOCKED_TEST_BLOCKED`.

Next action: write the P2 train-derived validation design before any larger
training. P1b is not quality evidence, and locked test remains blocked.

P2 design:
- design file:
  `experience_docx/experiment_logs/haze4k_v3_2_convir_wd_full_model_line_20260707/v32_p2_train_derived_validation_design.md`;
- split source: cloud-only v3.1 per-image table
  `experience_docx/experiment_logs/haze4k_v3_1_full_model_candidate_bakeoff_20260707/v31_candidate_per_image_cloud_only.csv`;
- locked Haze4K test remains blocked throughout P2;
- initial P2 launch at `8d7a9f4` failed in the auxiliary modulation-stat logging
  path due missing full-image padding and is engineering-invalid, not a quality
  result;
- corrected P2R1 run id:
  `ConvIR-Haze4K-v32-p2r1-wddecoder-seed3407-20260707`;
- continue to P3 only if `Best.pkl` passes the fixed P2 gate and is not
  Pareto-dominated by the v3.1 standalone candidates on the same 120-image
  validation names.

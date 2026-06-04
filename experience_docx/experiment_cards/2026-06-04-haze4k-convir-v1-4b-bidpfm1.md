# ConvIR-Dehaze-v1.4B-BiDPFM1

Date: 2026-06-04

Status: v1.4B adapter-only completed; internal regular+hard gate failed; locked Haze4K test blocked.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: DPGA/UDP-Lite depth-prior fusion inside ConvIR features.
- Dataset or task: Haze4K train-derived internal splits first (`train_inner`, `val_regular`, `val_hard`).
- Primary objective: test whether full-resolution DPFM1-focused bidirectional depth-RGB fusion can improve hard recovery without v1.4A multi-scale tail risk.
- Main metric: PSNR delta versus official ConvIR-B A0 on internal validation.
- Secondary metrics: SSIM delta, positive ratio, strong-reference regression ratio, worst `<= -0.20 dB` count, latency/memory, fusion activity, component matrix.
- Execution environment: cloud server `dehaze1`; local WSL checkout is for editing and compile/syntax-only checks.
- Artifact root: `experience_docx/experiment_logs/haze4k_udp_lite_v14b_bidpfm1_20260604/`.
- Branch or isolated workspace: `codex/haze4k-convir-v1-4b-bidirectional-dpfm1`.

## Baseline Contract

- Baseline implementation: official ConvIR-B Haze4K A0 checkpoint and unchanged `Dehazing/ITS/models/ConvIR.py`.
- Candidate initialization: load A0 into DPGA wrapper; all extra `DPGA_*` keys must be neutral at init.
- Evaluation entrypoint: `experience_docx/tools/eval_haze4k_checkpoint_compare.py` with `--candidate_arch dpga --candidate_dpga_fusion_mode udp_bi`.
- Training entrypoint: `Dehazing/ITS/main.py --arch dpga --dpga_fusion_mode udp_bi --mode train`.
- Dataset and split: v1.3/v1.4 train-derived split JSON with `train_inner`, `val_regular`, and `val_hard`.
- Preprocessing and decoding: existing Haze4K loader plus cached DepthAnything V2 depth prior.
- Metric implementation: same PSNR/SSIM implementation as v1.3/v1.4 compare tooling.
- Reference entrypoints that must remain stable: plain `--arch convir` remains unchanged; legacy DPGA remains `--dpga_fusion_mode legacy`; v1.4A UDP-Lite remains `--dpga_fusion_mode udp_lite`.
- Checkpoint/export/resume contract: candidate checkpoints save a standard `model` state dict; A0 load must be strict except for `DPGA_*` keys.

## Most Valuable Attempt

- v1.4A failed as full `DPFM1+2+4`, not as all depth fusion: DPFM1-only was the only clean positive contributor and DPFM2-only was negative.
- The highest-value next variable is fusion form at the proven scale: replace single-direction DPFM1 attention with bidirectional full-resolution depth-RGB fusion.
- The route avoids mixing in direct partial unfreeze until adapter-only BiDPFM1 proves whether DPFM1 can independently increase hard gain.
- The cheap no-training prerequisite is a v1.4A runtime component matrix to decide whether DPFM4 has enough synergy with DPFM1 to justify a later weak-DPFM4 variant.

## Hypothesis

If full-resolution DPFM1 uses bidirectional local-window depth-RGB cross-attention
plus zero-init depth-guided channel fusion, then hard-bottom PSNR should improve
over v1.4A DPFM1-only because RGB tokens can attend to local depth evidence while
depth tokens also receive RGB context before being projected back into ConvIR's
full-resolution feature stream.

## Change

- Code branch: `codex/haze4k-convir-v1-4b-bidirectional-dpfm1`.
- Exact code/config change:
  - add `--dpga_fusion_mode udp_bi`;
  - instantiate bidirectional DPFM modules when `udp_bi` is selected;
  - run v1.4B with `--dpga_active_adapters dpfm1`;
  - use `--dpga_train_scope active_adapter_only` to keep ConvIR-B frozen and train only active DPGA path parameters.
- Enabled mechanisms: `DPGA_prior_encoder.stem`, `DPGA_dpfm1.channel`, `DPGA_dpfm1.cross_rgb_from_depth`, `DPGA_dpfm1.cross_depth_from_rgb`, and `DPGA_dpfm1.scale`.
- Explicitly disabled mechanisms: `DPGA_dpfm2`, `DPGA_dpfm4`, `DPGA_agf1`, `DPGA_agf2`, legacy shallow/bottleneck/skip DPGA, hard gate, APDR/output residual, direct ConvIR-B finetune, locked Haze4K test.
- Parameter/runtime/memory impact expected: higher than v1.4A DPFM1-only due two local-window cross-attention directions, lower than active full DPFM123 because only full-resolution DPFM1 is used.
- Initialization or no-op behavior: all output projections are zero-init; preflight output must match A0 within `<= 1e-6` max absolute difference.
- Resume policy: resume only same config/split; any changed active modules/components are a new diagnostic.
- Defaults changed: none for plain ConvIR, legacy DPGA, or v1.4A `udp_lite`.
- Defaults intentionally preserved: `--arch convir` and `--dpga_fusion_mode legacy` remain backward-compatible.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| Python compile | edited model/train/tool files compile locally | pass |
| v1.4A component matrix | `v14b_runtime_component_matrix.csv` exists for `val_regular` and `val_hard` | pass; `DPFM1+4` improves mean on both splits but raises regular worst count/strong regressions, so default v1.4B remains BiDPFM1-only |
| zero-init equivalence | `max_abs_diff(output, A0) <= 1e-6` on cloud | pass; `max_abs_diff=0.0` in `preflight/v14b_zero_init_equivalence.json` |
| projection-gradient liveness | at least one active BiDPFM1 zero-init projection has nonzero grad on cloud | pass; BiDPFM1 channel and both cross directions have nonzero grad in `preflight/v14b_zero_init_equivalence.json` |
| active train scope audit | train log shows only active DPGA prefixes plus ConvIR frozen | pass; `trainable=11081`, `frozen=8929171`, active prefixes `DPGA_dpfm1,DPGA_prior_encoder.stem` |
| no locked test | all commands use train-derived internal splits before gate | pass; no locked Haze4K test was run |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| v1.4A runtime component matrix | resolves DPFM1 channel/cross and DPFM1+4 synergy before training | `val_regular`, `val_hard` | `v14b_runtime_component_matrix.csv` |
| zero-init A0 equivalence | proves bidirectional path preserves anchor before training | one internal or synthetic sample | `v14b_zero_init_equivalence.json` |
| BiDPFM1 projection gradients | confirms both added directions can learn from restoration loss | preflight sample | `v14b_zero_init_equivalence.json` |
| fusion delta norm | checks branch activity and prevents uncontrolled correction | training logs | train log `DPGA_TC ... fusion_delta` |
| regular+hard gate | prevents regular-only or easy-safe selection from running locked test | Best/Final on both splits | `v14b_gate_eval_regular_and_hard.json` |

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| A0 with `--arch convir` | reference baseline | unchanged from prior accepted baseline |
| `udp_lite` v1.4A checkpoint matrix | separate old scale/component attribution from new BiDPFM1 training | no-training audit completes before interpreting v1.4B |
| BiDPFM1 zero-init | no-op anchor preservation | `max_abs_diff <= 1e-6` |
| channel-only vs cross-only | distinguish DGCA from bidirectional local-window attention | both matrix rows written where applicable |

## Runtime Component Matrix Result

- `DPFM1` action is mostly cross-attention: on `val_hard`, channel-only mean is `+0.004168 dB`, cross-only mean is `+0.022658 dB`, and all DPFM1 mean is `+0.026774 dB`.
- `DPFM1+4` is the strongest no-training mean row: `val_regular` mean `+0.035215 dB`, `val_hard` mean `+0.031952 dB`, and `val_hard` worst count `0`.
- `DPFM1+4` is not clean enough to make v1.4B a two-scale route because `val_regular` worst count rises to `8` and strong regression ratio rises to `0.24`.
- `DPFM2` remains blocked: `DPFM1+2` is worse than DPFM1 on mean/tail, and full `DPFM1+2+4` reproduces the v1.4A tail failure.

## v1.4B Adapter-Only Result

- Gate result: `continue_allowed=false`, `locked_test_allowed=false` in `v14b_eval_regular_hard/v14b_gate_eval_regular_and_hard.json`.
- Best `val_regular`: mean delta `+0.028624 dB`, positive ratio `0.536667`, worst count `17`, strong regression ratio `0.280000`, SSIM delta `-0.000007778`.
- Best `val_hard`: mean delta `+0.023429 dB`, hard bottom-25 `+0.020760 dB`, positive ratio `0.570000`, worst count `8`, SSIM delta `-0.000012043`.
- Final checkpoint matched Best at the split-summary level but also failed the same continue and locked-test lines.

### Interpretation

BiDPFM1 is engineering-valid and still directionally mean-positive, but it does
not improve over the clean v1.4A `DPFM1-only` runtime signal and it worsens the
regular/hard tail profile. The failure is therefore not a zero-init or gradient
liveness bug; it is a mechanism/capacity result for adapter-only bidirectional
DPFM1 under the current frozen ConvIR-B wrapper.

## Fair Run Contract

- Training budget: adapter-only internal run first; no locked test.
- Batch/sample policy: hard/medium/easy balanced sampler from v1.3/v1.4.
- Optimizer: Adam; v1.4B trains `active_adapter_only`.
- Loss weights: Charbonnier reconstruction, FFT `0.05`, anchor `0.05`, chroma `0.02`, fusion delta norm `0.0001`.
- Random seed policy: seed `3407` for first v1.4B diagnostic.
- Evaluation cadence: Best/Final evaluated on `val_regular` and `val_hard`.
- Hardware/runtime assumptions: cloud `dehaze1`; local WSL compile/static checks only.
- Locked evaluation policy: locked Haze4K test only once after v1.4B internal regular+hard locked-test gate passes.
- Exception budget: if v1.4B reaches continue line but not locked-test line, do not run locked test; move only to v1.4C full-resolution fusion-neighbor adapter.

## Gates

### Continue To v1.4C Line

- `val_regular` mean delta `>= +0.035 dB`.
- `val_regular` positive ratio `>= 0.62`.
- `val_regular` worst `<= -0.20 dB` count `<= 8/300`.
- `val_regular` strong regression ratio `<= 0.18`.
- `val_regular` SSIM delta `>= 0`.
- `val_hard` mean delta `>= +0.030 dB`.
- `val_hard` hard bottom-25 delta `>= +0.035 dB`.
- `val_hard` worst `<= -0.20 dB` count `<= 4/300`.
- `val_hard` SSIM delta `>= 0`.

### Allow One Locked Haze4K Test Line

- `val_regular` mean delta `>= +0.040 dB`.
- `val_regular` positive ratio `>= 0.62`.
- `val_regular` strong regression ratio `<= 0.16`.
- `val_regular` worst `<= -0.20 dB` count `<= 12/300`.
- `val_regular` SSIM delta `>= 0`.
- `val_hard` mean delta `>= +0.030 dB`.
- `val_hard` hard bottom-25 delta `>= +0.050 dB`.
- `val_hard` SSIM delta `>= 0`.

## Analysis Plan

- Run the v1.4A no-training component matrix before or in parallel with BiDPFM1 preflight/training.
- Run BiDPFM1 zero-init equivalence and gradient liveness before training.
- Train v1.4B adapter-only on `train_inner` only after preflight passes.
- Evaluate Best/Final on `val_regular` and `val_hard`.
- If only the continue line passes, write a separate v1.4C route card for full-resolution fusion-neighbor adapters.
- Sync completed text evidence back into `experience_docx/`, update the index/family summary/README, then commit and push text evidence to GitHub.

## Decision

- Decision label: `FAIL_STOP_V14B_BIDPFM1_ADAPTER_ONLY`.
- Image/global metric reason: Best `val_regular` mean `+0.028624 dB` and Best `val_hard` mean `+0.023429 dB` are below the continue line.
- Mechanism reason: BiDPFM1 passed A0-equivalence and gradient liveness, but training did not amplify the DPFM1 signal beyond the v1.4A DPFM1-only runtime evidence.
- Preservation or regression reason: `val_regular` worst count `17`, strong regression ratio `0.28`, and `val_hard` worst count `8` fail the tail gates.
- Cost/deployability reason: bidirectional full-resolution attention roughly doubles CUDA peak memory versus A0 on internal eval (`~1147 MiB` vs `~535 MiB`) without passing gates.
- Evidence strength label: completed internal diagnostic with preflight, component matrix, adapter-only training, regular/hard eval, and gate.
- Reopen condition, if stopped: a separate full-resolution neighbor-adapter or official UDPNet reproduction route must introduce a new mechanism and explicitly protect the tail; do not rerun BiDPFM1-only as scale/gate tuning.
- What this decides next: do not run locked Haze4K test; stop adapter-only BiDPFM1 as-is.

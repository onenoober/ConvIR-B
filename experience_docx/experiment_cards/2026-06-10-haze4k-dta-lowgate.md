# Haze4K ConvIR-B DTA Low-Gate Adapter

Date: 2026-06-10

Status: `COMPLETED_GATE_PASS_SCOUT5`; `gate20` full diagnostic run pending

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: architecture-change / external-prior adapter.
- Dataset or task: Haze4K paired dehazing.
- Primary objective: implement Innovation 1, a depth-guided transmission adapter
  (DTA), as a lightweight ConvIR-B fine-tuning route.
- Main metric: Haze4K PSNR vs the official ConvIR-B A0 checkpoint.
- Secondary metrics: SSIM, per-image PSNR delta, hard/easy bucket deltas,
  strong-reference regressions, DTA gate activity, transmission-rank loss,
  latency and peak memory when available.
- Execution environment: `convir-4090` only.
- Artifact root: `experience_docx/experiment_logs/haze4k_dta_lowgate_20260610/`.
- Branch or isolated workspace: local branch `codex/haze4k-dta-lowgate`;
  cloud workspace `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-lowgate`.
- Review package location: this route card plus the evidence root above.

## Baseline Contract

- Baseline implementation: official ConvIR-B Haze4K anchor from
  `github/codex/haze4k-official-arch-anchor`.
- Anchor commit: `2d529d4`.
- Baseline checkpoint path:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
  after `dehaze1` sync.
- Baseline checkpoint sha256:
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Evaluation entrypoint: `Dehazing/ITS/main.py --mode test --version base
  --arch official_convir --data Haze4K`.
- Training entrypoint: `Dehazing/ITS/main.py --mode train --version base
  --arch dta --init_model <A0> --init_model_partial --train_scope adapter_only`.
- Dataset and split:
  `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`, train/test directories
  using the repository Haze4K loader. The user will upload the dataset.
- Depth prior: offline `.npy` relative-depth cache from Depth Anything V2 when
  available; DTA runtime requires this cache for formal runs.
- Metric implementation: repository PSNR/SSIM plus
  `experience_docx/tools/eval_haze4k_checkpoint_compare.py`.
- Reproduced baseline result: anchor preflight previously verified strict
  checkpoint load, synthetic forward, and one train batch on `convir-5090`.
  `convir-4090` setup/preflight remains pending until SSH is authorized.
- Known reproduction gap: none for checkpoint load/preflight; full metric gap is
  recorded by the comparison tool for each DTA run.
- Reference entrypoints that must remain stable: official `--arch
  official_convir` and `--fam_mode original` paths.
- Checkpoint/export/resume contract: DTA checkpoints save normal `model` state;
  official checkpoint reuse is partial-load only.

## Hypothesis

Observed failure: ConvIR-B has no explicit geometry/transmission anchor, so
depth-correlated haze thickness can be ambiguous in far regions and depth-edge
transitions.

Target mechanism: use cached relative depth to construct transmission proxies
and generate bounded FiLM/gate corrections at ConvIR-B stage-2/stage-3 features,
while keeping the official backbone trusted at first.

Mechanism sentence:

```text
If we add a low-gate depth-guided transmission adapter to ConvIR-B's middle and
deep feature path, Haze4K dehazing should gain or at least expose useful
hard/far-scene signal because the network receives an explicit depth-to-
transmission prior rather than inferring transmission only from RGB.
```

## Change

- Code branch: `codex/haze4k-dta-lowgate`.
- Exact code/config change:
  - add `--arch dta` builder path in `Dehazing/ITS/models/ConvIR.py`;
  - add `DepthTransmissionAdapter` under the new prefix `DTA.*`;
  - add optional depth-cache loading to the Haze4K dataloader and synchronized
    crop/flip augmentation;
  - add explicit partial-load, adapter-only freeze, DTA auxiliary rank/TV losses,
    and DTA-aware validation/evaluation paths;
  - add preflight and compare tooling for DTA.
- Enabled mechanisms:
  - depth pyramid from cached teacher depth;
  - `t_proxy = exp(-alpha * normalized_depth)`;
  - bounded FiLM modulation at channels 64 and 128;
  - low gate limit and negative gate bias;
  - transmission head for rank/TV supervision and interpretability.
- Explicitly disabled mechanisms: SCR, event/temporal branch, UDPNet expert,
  APDR output residuals, locked-test threshold tuning.
- Parameter/runtime impact expected: small adapter only, expected below +5%
  parameters and low inference overhead when depth is cached.
- Initialization or no-op behavior:
  - official ConvIR-B modules load from A0 checkpoint;
  - allowed new prefix: `DTA.`;
  - DTA prior projection and first transmission projection: Kaiming init;
  - DTA FiLM heads: zero weights, zero gamma/beta, gate bias `-7.0` in run
    scripts, gate limit `0.03`;
  - transmission final head: zero weights/bias, so `t_pred = 0.5` at init;
  - `log_alpha2/log_alpha3 = log(1.0)`;
  - no-op equivalence must pass before training.
- Resume policy: no resume for preflight/smoke; if a later run resumes, it must
  preserve the same run id and checkpoint policy.
- Defaults changed: only when `--arch dta` is selected.
- Defaults intentionally preserved: official `--arch official_convir` behavior,
  three-scale outputs, padding, checkpoint save format, Haze4K data split.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| branch/anchor | branch starts at anchor commit `2d529d4` | passed on `codex/haze4k-dta-lowgate` |
| partial load | missing keys all start with `DTA.`, unexpected keys empty | passed: `602` loaded, `18` `DTA.` missing, unexpected `[]` |
| no-op equivalence | synthetic max abs diff vs A0 <= `1e-7` | passed: max abs diff `0.0` |
| real batch | one train batch with depth cache has finite loss and DTA gradients | passed: DTA grad sum `0.02175214` |
| compile/static | local py_compile succeeds only; no local runtime | passed |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| DTA stage2/stage3 gate mean/max | confirms the low-gate adapter remains bounded but active | preflight, smoke, later stop gates | train log / preflight JSON |
| DTA delta absolute mean | checks whether the branch is a no-op, exploding, or useful | preflight and validation stats | train log |
| transmission rank loss | checks depth-transmission monotonic pressure | train batches | train log |
| transmission TV loss | checks smooth but edge-aware transmission behavior | train batches | train log |
| per-image PSNR delta vs A0 | catches average wins with regressions | eval subset/full test when allowed | compare CSV/JSON |
| strong-reference regression count | protects already-good A0 cases | compare split | compare JSON/CSV |

## Fair Run Contract

- Training or inference budget: first cloud validation is preflight plus a short
  adapter-only smoke; 5/20 epoch runs are scripts/next actions, not silently
  launched from this card unless the smoke succeeds.
- Batch/sample policy: Haze4K train/test loader with cached depth; first smoke
  may use small batch size for runtime sanity.
- Optimizer: Adam over trainable parameters only.
- Schedule: original warmup+cosine code with `T_max >= 1` guard for short smoke.
- Loss weights: original L1 + `0.1 * FFT`; DTA low auxiliary defaults:
  `rank_weight <= 0.003`, `tv_weight <= 0.0003`, `proxy_weight = 0`.
- Random seed policy: first smoke seed `3407`.
- Evaluation cadence: smoke can validate every epoch; later stop20 uses written
  cadence and checkpoint comparison.
- Hardware/runtime assumptions: `convir-4090`, GPU 0, explicit Python
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Allowed resume behavior: none for first preflight/smoke.
- Noise floor or minimum effect size for this route: smoke gates are deliberately
  lenient; no promotion claim below the documented Haze4K noise floor.
- Locked evaluation policy: locked Haze4K test must not be used for selection;
  any full test comparison here is diagnostic unless a later card fixes
  checkpoint/threshold selection before one-shot confirmation.
- Exception budget: one low-cost engineering smoke rerun is allowed only for
  infra/import/path failures, not for changing model scope or loss weights.

## Low-Gate / Lenient Gates

The user requested normal training/testing content and gates that are much less
strict than prior stop routes. This card therefore uses low DTA intervention
gates and lenient continuation gates:

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| preflight | no metric requirement | partial-load/no-op/finite-grad pass | stop on load/shape/NaN failure |
| smoke | finite validation PSNR/SSIM and checkpoint saved | DTA losses finite; gate mean remains bounded by `gate_limit` | continue if training and eval run normally |
| 5 epoch scout | PSNR not catastrophically below A0 smoke reference (`>-1.0 dB`) | DTA branch active or transmission loss decreases | continue to 20 only if no severe regressions are obvious |
| 20 epoch gate | mean PSNR within `0.50 dB` of A0 or any clear hard/far-scene gain | depth/transmission metrics not contradictory | diagnostic pass only; not promotion |
| promotion | requires a new, stricter card | mechanism and preservation both support the claim | no locked-test tuning in this card |

## Analysis Plan

- Per-sample or subgroup analysis: A0 vs DTA per-image PSNR/SSIM, hard bottom
  25%, easy top 25%, strong-reference regressions, worst regressions.
- Visual or qualitative analysis: optional later; not required for first smoke.
- Complexity analysis: parameter count and peak CUDA memory from preflight/eval.
- Robustness or held-out analysis: later 5/20 epoch card if smoke is healthy.
- Required docs to update: this card, evidence README, central index, DPGA/depth
  family summary or new DTA summary after cloud evidence exists.
- Required artifacts to retain: scripts, logs, status, JSON/CSV summaries.
- Required artifacts to keep external: checkpoints, images, depth arrays.
- Evidence package contents: text logs, command scripts, status, preflight JSON,
  train/eval summaries.
- Evidence package audit: `git diff --check`; no weights/datasets/images/arrays.

## Decision

- Decision label: `COMPLETED_GATE_PASS_SCOUT5_CONTINUE_GATE20`.
- Image/global metric reason: smoke 32-image diagnostic was neutral
  (`+0.002904 dB` mean PSNR delta); scout5 128-image diagnostic was mildly
  negative (`-0.036217 dB`) but comfortably above the lenient `-1.0 dB`
  continuation stop line with zero `<= -0.20 dB` worst regressions.
- Mechanism reason: DTA rank loss decreased from `0.6472` to `0.5469` across
  the 5-epoch scout, while low gates stayed bounded at about `2.8e-5`.
- Preservation or regression reason: scout5 has strong-reference regressions
  (`15/32` strong samples on the 128-image diagnostic), so this is not a
  promotion result; it is only enough to continue to the predeclared gate20 run.
- Cost/deployability reason: adapter-only trainable parameters are `17,765` of
  `8,648,430`; peak memory in the smoke/scout comparisons increased only from
  about `529.6 MiB` to `531.9 MiB`.
- Evidence strength label: convir-4090 preflight, smoke, and 5-epoch scout
  complete; full diagnostic gate20 pending.
- Reopen condition, if any: if gate20 fails scientifically, keep this route as
  diagnostic evidence for depth/transmission priors and do not tune on the
  locked Haze4K test.
- What this decides next: run the predeclared `gate20` adapter-only training and
  full diagnostic A0 comparison without changing DTA scope or low-gate settings.

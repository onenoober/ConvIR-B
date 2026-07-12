# Haze4K v5 CHD-RM v3u Render-Only Activation Diagnostic

Date: 2026-07-13

Status: `PLANNED`

## Scope

- Project: ConvIR-B Haze4K.
- Route type: fresh anchor-based architecture diagnostic, not a v3s/v3t resume.
- Question: does the v3s/v3t minimal-repair penalty itself cause the zero lock
  when all output-side, frozen-operator conditions are held fixed?
- Hypothesis: optimizing only real rendered `.25` candidate MSE activates the
  zero-init bounded output-side `Delta u` field; this test makes no safety or
  deployment claim.
- Dataset: first 32 fixed names of the train-derived, clean-reference-grouped
  v3j OOF list; no canary or locked test.
- GitHub rules commit: `github/main@67cf24a2b2aa24d20dfa3a966595241722a6c919`.
- Source anchor: `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Reused diagnostic implementation source: `github/codex/haze4k-v5-v3t-zero-lock-context-diagnostic-20260713@5895d161bf69938a938a7627b289befd6b87db41`.
- Route branch: `codex/haze4k-v5-v3u-render-only-activation-diagnostic-20260713`.
- Local WSL: `/home/ubuntu/workspace/ConvIR-B-v3u-render-only-activation-diagnostic-20260713`.
- Cloud `REMOTE_REPO`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3u-render-only-activation-diagnostic-20260713`.
- Cloud `RUN_ROOT`: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3u_render_only_activation_diagnostic_20260713`.
- Cloud `EVID_STAGE`: `$REMOTE_REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3u_render_only_activation_diagnostic_20260713`.
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Parent Evidence And Single Variable

- v3r establishes direction-line repair headroom: worst-operator LCB95 is
  `+0.280496 dB` over old `.25`, while scale and channel-scale fail.
- v3s exact no-op passed but its fixed-32 output-side learned field was
  inactive (`|Delta u|=1.252e-7 < 1e-6`) under the full safe objective.
- v3t exact no-op passed for output-side and frozen-context forms. Its output
  utility control removed anchor/harm/CVaR but retained the repair penalty and
  remained inactive (`|Delta u|=2.40e-7`, rendered-loss reduction `0.00011%`).
- v3u preserves v3t's output-side head, zero initialization, fixed operator,
  support, bounds, 32 names, seed, 16 epochs, LR `5e-4`, weight decay `1e-5`,
  risk window four, and gradient clip `0.1`. The sole scientific change is
  `rendered .25 MSE + 0.02 * repair` to `rendered .25 MSE`.

## Architecture And Static Contract

- New route prefix: `DIRT_*` in `Dehazing/ITS/models/direction_repair_context.py`.
- Active form: `DIRT_OutputDeltaU([I_hazy, y0_base, u_old])`, constrained as
  `support * delta_bound * tanh(raw)` under the inherited old hard support.
- The head output convolution is exactly zero initialized. The S0 contract
  requires exact zero `Delta u` and prediction difference before activation.
- Frozen context is used only to reconstruct the already-frozen `u_old` step;
  it is not an input to the v3u `DIRT_*` head.
- All base ConvIR-B, FAM2/D7c control, gate, density, `D_ref`, and `D_rep`
  components remain frozen in eval mode. Only `DIRT_*` is trainable; the
  output-side head has 2,883 trainable parameters.
- Official ConvIR-B keys retain exact shapes. The official Haze4K checkpoint
  path is `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
  with SHA-256 `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
  New `DIRT_*` parameters are zero-init modules, not checkpoint reuse targets.
- The runner retains the v3s/v3p frozen-source commits and all asset SHA-256
  checks from v3t; its source manifest records the names, bounds, assets, and
  objective weight `repair_weight=0.0` for each stage.

## Execution Profile And Gates

Profile: static contract -> S0 exact no-op -> S1 fixed-32 activation diagnostic.

| Stage | Scope | Pass rule | Pass authorizes | Fail consequence |
| --- | --- | --- | --- | --- |
| S0 | 32 names x two frozen operators, zero-init output-side head | exact zero `Delta u` and prediction difference; old `.125` replay difference <= `1e-6 dB` | S1 only | stop v3u |
| S1 | same 32 names, 16 epochs, seed 3407 | final mean `|Delta u| >= 1e-6` and independently measured rendered `.25` loss reduction >= `0.1%` | safety-curriculum training-contract design only | stop this direct output-side rendered activation form; redesign target or parameterization |

S1 uses only the mean real rendered `.25` candidate MSE for gradient updates.
Anchor, block-margin, harm, CVaR, and repair magnitude are not added to the
loss. The runner records the latter quantities only as diagnostics and does not
use them for optimization or the activity decision.

## Forbidden Continuations

- Do not resume v3s or v3t, lower the activity threshold, widen the context,
  tune the S1 budget/LR/bounds, or add a second loss term after seeing S1.
- Do not run formal candidate training, safety optimization, selector/scorer
  work, calibration, policy replay, canary, deployment, or locked test from
  either S0 or S1.
- An S1 pass is evidence of activation only. It does not establish safety,
  quality utility, low-haze protection, or a promotable candidate.

## Evidence Boundary

Cloud-only: checkpoints, raw outputs, images, full logs, and per-image tables.
Compact route evidence: tracked runner, source manifests, S0/S1 typed closeouts,
S1 history and summary, evidence README, and this route card. Terminal evidence
sync updates `EXPERIMENT_INDEX.md` and `CHD_RM_EXPERIMENT_INDEX.md` only.

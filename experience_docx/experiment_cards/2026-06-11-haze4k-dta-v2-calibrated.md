# Haze4K ConvIR-B DTA-v2 Calibrated Confidence-Gated Adapter

Date: 2026-06-11

Status: `IN_PROGRESS_CLOUD_QUEUE_PENDING`

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: Innovation 1 / depth-guided transmission adapter.
- Branch: `codex/haze4k-dta-v2-calibrated`, starting from completed DTA low-gate evidence commit `04c356c`.
- Cloud runtime only: `convir-4090` under `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated`.
- Local policy: local WSL is restricted to code editing and compile/static checks; no local runtime tests, smoke, training, eval, inference, or demos.
- Evidence root: `experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611/`.

## Reason To Reopen DTA

The completed low-gate DTA route was engineering-valid but not promotion-ready:
full gate20 was A0-level yet slightly negative and had no hard/far gain. This
route does not retune that same low-gate adapter. It implements the revised
analysis conclusion: first audit depth-vs-transmission calibration, then train a
confidence-gated, supervised-transmission DTA-v2 with mechanism controls.

## DTA-v2 Change

- Adds `--arch dta_v2` while preserving `--arch official_convir`, `--arch convir`, and the old `--arch dta` path.
- Extends the DTA prior to `[depth, calibrated t_proxy, -log(t_proxy), depth_grad_x, depth_grad_y, confidence]`.
- Adds confidence-gated bounded FiLM at stage-2/stage-3 and a zero-init decoder-side output residual refinement.
- Adds optional Haze4K `trans` loading with synchronized crop/flip and filename-derived airlight.
- Adds supervised transmission, physical hazy reconstruction, and easy/clear preservation auxiliary losses.
- Adds depth controls: `normal`, `invert`, `zero`, and `shuffle`.
- Adds DTA-specific gradient clipping and a gate ramp (`0.01 -> 0.03 -> 0.06`) for v2 runs.

## Initialization / Partial Load Contract

- Start from official Haze4K A0 checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Use `--init_model_partial --partial_new_prefixes DTA.` for all DTA-v2 fine-tuning.
- Official ConvIR-B backbone, FAM, SCM, decoder, and convolution modules load from A0.
- New `DTA.*` parameters are the only missing partial-load keys.
- FiLM/output residual heads are zero-initialized so synthetic no-op equivalence must pass before training.

## Required Evidence Queue

| Stage | Required artifact | Purpose |
| --- | --- | --- |
| depth-transmission audit | `dta_depth_transmission_audit_summary.json/csv` | Verify depth orientation, alpha calibration, proxy error, low-texture/bright/dense-region risk. |
| OOF split generation | `dta_v2_haze4k_oof_splits_seed3407.json` | Keep route selection on train-derived splits, not locked test. |
| preflight | `dta_v2_preflight.json/log` | Partial load, no-op equivalence, real-batch finite loss, DTA gradients, trans/physics losses. |
| adapter-only DTA-v2 | train/eval logs and compare JSON/CSV | Test calibrated/confidence DTA with true depth. |
| depth controls | zero, shuffle, invert compare JSON/CSV | Verify any gain is depth-mechanism dependent. |
| adapter-neighbors | train/eval logs and compare JSON/CSV | Test whether neighboring FAM/Convs can consume the prior. |
| final confirmation | one fixed locked-test compare only if internal gates pass | Avoid repeated locked-test selection. |

## Default V2 Run Settings

- Train scope first: `adapter_only`; second: `adapter_neighbors` only after adapter-only/control evidence exists.
- Loss: original multiscale L1 + `0.1 * FFT` plus `rank=0.001`, `tv=0.0001`, `trans=0.02`, `phys=0.005`, `preserve=0.02`.
- Gate: `gate_bias=-6.0`, final `gate_limit=0.06`, `gamma_limit=0.12`, `beta_limit=0.06`, confidence floor `0.25`.
- Clip: DTA `0.05`, neighbors `0.005`, fallback global `0.001`.
- Depth controls: true/normal, zero, shuffle, invert.
- Seeds: start with `3407`; expand to multi-seed OOF when the first full mechanism pass is healthy.

## Locked-Test Policy

Locked Haze4K test must not be used to select checkpoint, gate, loss, depth mode,
or train scope. The queue starts with audit/preflight and train-derived OOF or
small diagnostic runs. A locked full test is allowed only once for a fixed config
that has already passed internal mechanism and preservation gates.

## Current State

- Code implementation and scripts are in progress locally.
- No DTA-v2 cloud training/evaluation result exists yet.
- First cloud commands after push: convir-4090 setup, depth-transmission audit,
  OOF split generation, and DTA-v2 preflight.

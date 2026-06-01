# ConvIR-B Haze4K Experiment Index

Date: 2026-06-01

Status: evidence index for `codex/main-experiment-evidence-sync`.

## Purpose

This file is the main branch reading map for the Haze4K experiment sequence.
It centralizes route cards, text logs, result tables, and decision labels while
leaving experimental code on the route branches.

Use this index first when asking what happened, which route is still relevant,
and where the evidence lives. Use the listed source branch or commit only when
you need the exact runnable code snapshot.

## Branch Cleanup

Remote branch cleanup was done before this evidence sync. The deleted refs were
not unique heads: each was already an ancestor of one or both retained leaf
branches, so their commits remain reachable through the retained branches.

| Deleted remote ref | Reason |
| --- | --- |
| `codex/haze4k-repro` | Contained by all later Haze4K route branches. |
| `codex/haze4k-fam2-only` | Contained by later FAM2, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-bounded` | Contained by later confidence-gate, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-confidence-gate` | Contained by later selectivity, hardfreq, and haze-prior branches. |
| `codex/haze4k-stop20-noise-floor` | Contained by later selectivity, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-selectivity-or-kill` | Contained by both retained leaf branches. |

Retained remote refs:

- `github/main`: stable entry point plus consolidated text evidence.
- `github/codex/haze4k-hardfreq-loss`: leaf route containing hard frequency
  loss evidence and prior route history.
- `github/codex/haze4k-haze-prior-scm`: leaf route containing haze-prior SCM
  evidence, a GitHub-readable text package, and prior route history.

## Reading Order

1. Read the summary table below.
2. Open the route card for the route you care about.
3. Open the evidence root for JSON/CSV/log detail.
4. Use the retained source branch only when you need runnable code; do not infer
   that diagnostic or failed route code belongs in `main`.

## Route Summary

| Route | Status | Main result | Decision | Card | Evidence root | Source after cleanup |
| --- | --- | --- | --- | --- | --- | --- |
| FAM `modres` 5-epoch scout | Completed diagnostic | Mean PSNR `+0.0953 dB`, but median delta negative and strong-reference regressions `142/250`. | Do not promote unchanged `modres`; mechanism is active but preservation fails. | [card](experiment_cards/2026-05-31-haze4k-fam-feature-modulation.md) | [logs](experiment_logs/haze4k_fam_modres_scout_stop5_20260531/) | `github/main` |
| FAM2-only 20-epoch scout | Completed diagnostic | Mean PSNR `+0.1739 dB`; hard bottom 25% `+0.8159 dB`; easy top 25% `-0.2860 dB`; strong-reference regressions `138/250`. | Keep as diagnostic; preservation gate fails. | [card](experiment_cards/2026-05-31-haze4k-fam2-only-modulation.md) | [logs](experiment_logs/haze4k_fam2_modres_stop20_20260531/) | retained leaf branches |
| FAM2 bounded gamma | Completed diagnostic | Mean PSNR `-0.0271 dB`; hard `+0.8054 dB`; easy `-1.2740 dB`; strong-reference regressions `181/250`. | Bounded gamma does not solve preservation; do not promote. | [card](experiment_cards/2026-06-01-haze4k-fam2-bounded-modulation.md) | [logs](experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/) | retained leaf branches |
| FAM2 confidence-gated gamma | Completed diagnostic | Mean PSNR `+0.4523 dB`; hard `+0.9380 dB`; easy `-0.0700 dB`; strong-reference regressions `121/250`. | Positive quality signal, but preservation/selectivity still not decision-grade. | [card](experiment_cards/2026-06-01-haze4k-fam2-confidence-gate.md) | [logs](experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/) | retained leaf branches |
| Stop20 original noise floor | Completed baseline audit | Seed mean PSNR std `0.2206 dB`; hard bucket std `0.4551 dB`; single-seed route claims need caution. | Use as the noise floor for stop20 route decisions. | [card](experiment_cards/2026-06-01-haze4k-stop20-noise-floor.md) | [logs](experiment_logs/haze4k_stop20_noise_floor_20260601/) | retained leaf branches |
| FAM2 selectivity-or-kill | Completed no-training meta-analysis | Deployable selectors passing gate: `0`; best positive-gain AUC `0.5874`; best feasible threshold-gate mean delta `+0.1333 dB`. | `FAIL_STOP_FAM_ROUTE`; no deployable FAM selector is strong enough. | [card](experiment_cards/2026-06-01-haze4k-fam2-selectivity-or-kill.md) | [logs](experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/) | retained leaf branches |
| Hard-aware frequency loss | Completed diagnostic | Best mean PSNR `-0.2127 dB`; hard `+0.5999 dB`; easy `-1.2363 dB`; strong-reference regressions `188/250`; Best-vs-Last `-0.6922 dB`. | `FAIL_STOP_HARDFFT_LAMBDA_002`; do not repeat or promote `hard_fft_lambda=0.02` as-is. | [card](experiment_cards/2026-06-01-haze4k-hardfreq-loss.md) | [logs](experiment_logs/haze4k_hardfreq_loss_stop20_20260601/) | `github/codex/haze4k-hardfreq-loss` |
| Haze-prior SCM + hard auxiliary | Completed diagnostic | Best mean PSNR `-0.3789 dB`; hard `+0.3501 dB`; easy `-1.6511 dB`; strong-reference regressions `185/250`. | `NO_PROMOTE_STOP20_HAZE_PRIOR_SCM_HARDAUX`; do not promote this exact route. | [card](experiment_cards/2026-06-01-haze4k-haze-prior-scm.md) | [logs](experiment_logs/haze4k_haze_prior_scm_20260601/) | `github/codex/haze4k-haze-prior-scm` |

## Evidence Inventory

| Evidence root | Files | Main contents |
| --- | ---: | --- |
| `experiment_logs/haze4k_fam_modres_preflight_20260531/` | 2 | FAM preflight and one-batch train probe logs. |
| `experiment_logs/haze4k_fam_modres_scout_stop5_20260531/` | 8 | Stop5 train logs, compare JSON, per-image CSV, run script, README. |
| `experiment_logs/haze4k_fam2_modres_preflight_20260531/` | 2 | FAM2 equivalence and real-batch preflight JSON. |
| `experiment_logs/haze4k_fam2_modres_stop20_20260531/` | 7 | Matched original/FAM2 train logs and stop20 compare JSON/CSV. |
| `experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/` | 18 | Bounded gamma train log, modulation analysis, compare JSON/CSV, run script. |
| `experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/` | 20 | Confidence-gate train log, proxy separability, modulation analysis, compare JSON/CSV. |
| `experiment_logs/haze4k_stop20_noise_floor_20260601/` | 8 | Original multi-seed train logs, seed-noise JSON/CSV, tmux text output. |
| `experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/` | 3 | Selector meta-analysis JSON/CSV and per-image table. |
| `experiment_logs/haze4k_hardfreq_loss_stop20_20260601/` | 13 | Hard-frequency preflight, train log, Best/Last compare JSON/CSV, run script. |
| `experiment_logs/haze4k_haze_prior_scm_20260601/` | 10 | Haze-prior preflights, Best/Last compare JSON/CSV, run script, status. |
| `../docs/ai_text_packages/2026-06-01-haze4k-haze-prior-scm/` | 12 | GitHub-readable compact package for the haze-prior SCM route. |

## Current Route Verdict

The active conclusion is conservative:

- FAM2 found a real hard-sample improvement direction, but the deployable
  selector route failed.
- Hard-frequency weighting and haze-prior SCM also moved hard cases but harmed
  global/easy preservation too much.
- No current route is promotion-ready.
- `main` should carry the evidence and index, while runnable experimental code
  stays on the retained leaf branches.

## Artifact Boundary

This sync intentionally includes text evidence only:

- route cards: `.md`;
- logs and command transcripts: `.log`, `.txt`, `.out`;
- result tables: `.csv`;
- structured summaries: `.json`;
- reproducibility commands: `.sh`.

It intentionally excludes checkpoints, model weights, image outputs, datasets,
NumPy arrays, and raw inference artifacts.

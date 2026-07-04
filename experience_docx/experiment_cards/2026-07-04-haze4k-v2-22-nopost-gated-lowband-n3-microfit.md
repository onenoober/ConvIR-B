# Haze4K v2.22 NoPost Gated Lowband N3 Microfit

Date: 2026-07-04

Status: completed review-only

## Scope

- Project: ConvIR-B Haze4K NoPost lowband policy.
- Model family: NoPost feature-lowband action policy.
- Route branch: `codex/haze4k-v2-22-nopost-gated-lowband-n3-microfit`.
- Anchor: `github/codex/haze4k-official-arch-anchor` at `2d529d4`.
- Runtime: `convir-4090` only.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Evidence root: `experience_docx/experiment_logs/haze4k_v2_22_nopost_gated_lowband_n3_microfit_20260704/`.
- Locked Haze4K test: blocked.

## Precedent

v2.21 replay selected `V221_risk_temperature_gamma0p50` and passed fixed OOF safety replay gates:

```text
mean +2.2270 dB
hard +4.3031 dB
easy +0.7403 dB
p05 -0.0025 dB
CVaR5 -0.2089 dB
severe 1.79%
strong-reference regression 4.83%
fold tail pass 5/5
```

v2.21 explicitly authorized only a separate N3 microfit route-card review. It did not launch training and did not authorize locked-test use.

## Hypothesis

The v2.21 safety-temperature controller can be converted into a trainable internal mechanism:

```text
ConvIR internal decoder mid/final feature
-> Haar LL action grid
-> unsafe/no-op scalar gate from internal context
-> delta_LL *= (1 - p_unsafe)^0.5
-> IWT back into feature path
-> original ConvIR output path
```

N3 tests only whether this mechanism can train stably on small train-derived microfit stages while preserving zero-init identity and avoiding immediate tail/preserve collapse.

## Architecture Contract

- New file: `Dehazing/ITS/models/NoPostGatedLowbandConvIR.py`.
- New parameter prefix: `nopost_gated_lowband_policy.*`.
- Forward signature remains `forward(self, x)`.
- Official ConvIR-B keys are strict shape-match loaded from A0.
- Missing keys are allowed only under `nopost_gated_lowband_policy.*`.
- New action projections are zero-initialized.
- Unsafe/no-op heads have zero last-layer weights and conservative bias `-1.5`.
- No A0, WD0375, WDMamba, teacher, expert output, or RGB correction is used as forward input.
- No output-output delta is used.

## Stage Ladder

| Stage | Samples | Trainable scope | Continue rule |
| --- | ---: | --- | --- |
| P0 | 1 batch | none | strict partial-load, zero-init max abs vs A0 <= 1e-6, finite forward |
| N3-16 | 16 train-derived crops | `nopost_gated_lowband_policy.*` only | finite loss, nonzero action, nondegenerate gate, no severe explosion |
| N3-64 | 64 train-derived crops | same | same |
| N3-256 | 256 train-derived crops | same | same |

Each stage restarts from A0 plus zero-init route modules; stages are not cumulative training.

## Gates

N3 is review-only. Passing all stages means:

```text
V222_N3_MICROFIT_PASS_REVIEW_ONLY_NO_LOCKED_TEST
```

It authorizes only a later route review for an OOF train-derived training design. It does not authorize locked Haze4K test.

Normal pause if any stage fails:

```text
V222_N3_MICROFIT_NORMAL_GATE_PAUSE_NO_LOCKED_TEST
```

Minimum stage checks:

- finite mean dPSNR versus A0;
- severe rate <= 15% on the stage eval crop set;
- p05 dPSNR >= -1.0 dB;
- mid/final action RMS nonzero;
- mid/final unsafe probability in `[0.02, 0.98]`;
- locked test untouched.

These are microfit stability checks, not promotion metrics.

## Evidence

Expected compact outputs:

- `v222_n3_preflight.json`
- `v222_n3_stage_summary.csv`
- `v222_n3_closeout.json`
- `microfit*/v222_*_train_history.csv`
- `microfit*/v222_*_eval_summary.json`
- `microfit*/v222_*_gate.json`
- `README.md`
- `status.txt`

Do not commit checkpoints, raw images, arrays, or large per-image output tables by default.

## Closeout

Decision:

```text
V222_N3_MICROFIT_PASS_REVIEW_ONLY_NO_LOCKED_TEST
```

Runtime source:

- cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-22-nopost-gated-lowband-n3-microfit`
- branch: `codex/haze4k-v2-22-nopost-gated-lowband-n3-microfit`
- run commit: `9e2f548`
- evidence root: `experience_docx/experiment_logs/haze4k_v2_22_nopost_gated_lowband_n3_microfit_20260704/`

Results:

- P0 passed after filtering non-image files from Haze4K train directories.
- Strict partial load reused `602` official ConvIR-B state keys.
- Zero-init identity remained within the route tolerance.
- All microfit stages completed with adapter-only training and nonzero but bounded mid/final lowband actions.
- `microfit16`: mean `+0.0275 dB`, hard bottom25 `-0.0002 dB`, p05 `-0.1414 dB`, severe `0`, mean mid/final unsafe probability `0.1814/0.1815`.
- `microfit64`: mean `-0.0023 dB`, hard bottom25 `+0.0017 dB`, p05 `-0.1184 dB`, severe `0`, mean mid/final unsafe probability `0.1796/0.1805`.
- `microfit256`: mean `-0.0029 dB`, hard bottom25 `+0.0107 dB`, p05 `-0.2146 dB`, severe `5.47%`, mean mid/final unsafe probability `0.1817/0.1817`.

Interpretation:

v2.22 is a successful N3 stability check, not a model-quality win. The internal gated lowband route can train without immediate collapse, but the output remains near A0 and strong-reference regressions remain high (`26.56%` to `32.81%` across stages). Passing v2.22 authorizes only a separate OOF train-derived route review. Locked Haze4K test remains blocked.

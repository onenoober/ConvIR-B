# Haze4K v2.23 NoPost OOF Gated Lowband Train Screen

Date: 2026-07-04

Status: completed normal gate pause

## Scope

- Project: ConvIR-B Haze4K NoPost lowband policy.
- Model family: NoPost feature-lowband action policy.
- Route branch: `codex/haze4k-v2-23-nopost-oof-gated-lowband-train`.
- Anchor: `github/codex/haze4k-official-arch-anchor` at `2d529d4`.
- Runtime: `convir-4090` only.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Split source: v2.16/v2.17 train-derived CSV with `oof_fold`.
- Evidence root: `experience_docx/experiment_logs/haze4k_v2_23_nopost_oof_gated_lowband_train_20260704/`.
- Locked Haze4K test: blocked.

## Precedent

v2.21 fixed OOF safety replay selected `V221_risk_temperature_gamma0p50`:

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

v2.22 converted that controller into a trainable internal gated-lowband module
and passed N3 microfit stability. The result remained near A0, so v2.22 was not
a quality win; it authorized only this separate OOF train-derived review.

## Hypothesis

The trainable gated-lowband module can retain the v2.21 safety/no-op behavior
under real training and produce non-near-identity OOF movement without reopening
the v2.20 tail failure.

## Architecture Contract

- Reuse `Dehazing/ITS/models/NoPostGatedLowbandConvIR.py`.
- New parameter prefix: `nopost_gated_lowband_policy.*`.
- Forward signature remains `forward(self, x)`.
- Official ConvIR-B keys are strict shape-match loaded from A0.
- Missing keys are allowed only under `nopost_gated_lowband_policy.*`.
- Trainable scope is adapter-only: `nopost_gated_lowband_policy.*`.
- No A0, WD0375, WDMamba, teacher, expert output, or RGB correction is used as forward input.
- No output-output delta is used.

## Stages And Gates

| Stage | Purpose | Continue rule |
| --- | --- | --- |
| P0 | source/contract, official checkpoint partial load, zero-init identity | pass before OOF |
| OOF screen | train `k` fold-heldout adapter-only models on train-derived folds and evaluate heldout folds | pass only if OOF safety and non-near-identity gates both pass |

Default screen is intentionally small:

```text
folds: 0,1,2
train samples per fold: 384
heldout eval samples per fold: 160
epochs: 2
crop: 256
```

## OOF Gate

This is a screen gate, not promotion:

```text
mean dPSNR >= +0.05
hard bottom25 >= +0.05
easy top25 >= -0.05
positive_ratio >= 0.55
p05 >= -0.50
CVaR5 >= -0.80
severe_rate <= 0.08
strong-reference regression rate <= 0.20
fold tail pass >= 2/3
mean action RMS >= 1e-5
unsafe probability remains in [0.02, 0.98]
locked test untouched
```

If the OOF screen fails, pause normally. Do not increase epochs, samples, loss
weights, or fold selection under the same run id. If the screen passes, it only
authorizes a larger OOF route-card review; locked test remains blocked.

## Evidence

Expected compact outputs:

- `v223_oof_preflight.json`
- `v223_oof_fold_summary.csv`
- `v223_oof_summary.json`
- `v223_oof_gate.json`
- `v223_oof_closeout.json`
- `fold*/v223_fold*_train_history.csv`
- `fold*/v223_fold*_eval_summary.json`
- `README.md`
- `status.txt`

Do not commit checkpoints, raw images, arrays, archives, or large per-image
output tables by default.

## Closeout

Decision:

```text
V223_OOF_SCREEN_GATE_FAIL_NORMAL_PAUSE_NO_LOCKED_TEST
```

Runtime source:

- cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-23-nopost-oof-gated-lowband-train`
- branch: `codex/haze4k-v2-23-nopost-oof-gated-lowband-train`
- git checkout before rsync: `246bdce`
- runtime source patch: local commit `73bab6e` rsynced after GitHub HTTPS fetch on cloud timed out
- evidence root: `experience_docx/experiment_logs/haze4k_v2_23_nopost_oof_gated_lowband_train_20260704/`

Results:

- Initial launch failed preflight as an engineering issue because the route model
  remained in train mode during identity preflight; BatchNorm saw batch `1` and
  spatial `1x1`.
- The preflight was fixed by setting the route model to eval mode for P0 only.
- P0 then passed with official checkpoint partial load and zero-init identity.
- OOF screen completed folds `0,1,2` with `384` train samples and `160`
  heldout eval samples per fold.
- Aggregate OOF mean was `+0.0367 dB`, hard bottom25 `+0.0227 dB`, easy top25
  `+0.0165 dB`, p05 `-0.3528 dB`, CVaR5 `-0.4782 dB`, positive ratio `0.5625`,
  severe rate `14.79%`, strong-reference regression rate `32.08%`, and fold tail
  pass `1/3`.
- Gate passed only CVaR, easy preservation, p05, positive ratio, non-near-identity,
  nondegenerate gate, and locked-test untouched checks.
- Gate failed mean, hard, severe, strong-reference, and fold-tail checks.

Interpretation:

v2.23 is a useful negative screen. The v2.22 trainable gated-lowband module can
move nontrivially and keep gate probabilities nondegenerate, but the first
train-derived OOF screen does not preserve tail/strong-reference safety and does
not deliver enough mean or hard gain. Pause normally. Do not expand epochs,
samples, loss weights, or folds under this run id. Locked Haze4K test remains
blocked.

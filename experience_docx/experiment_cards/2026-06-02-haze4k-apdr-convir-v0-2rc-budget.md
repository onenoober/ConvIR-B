# Haze4K APDR-ConvIR v0.2RC Conservative Budget

Date: 2026-06-02

Status: planned cloud conservative-budget replay and oracle ceiling preflight.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-ConvIR, anchored on official ConvIR-B.
- Route name: APDR-v0.2RC Conservative Budget Router.
- Dataset or task: Haze4K dehazing.
- Primary objective: convert the strong v0.2R hard/easy ranking into a
  conservative action budget before residual training.
- Execution environment: AutoDL `autodl-dehaze3`, `convir-cu128`.
- Artifact root: `experience_docx/experiment_logs/haze4k_apdr_v0_2rc_budget_20260602/`.
- Branch or isolated workspace: `codex/haze4k-apdr-convir-v0-2rc-conservative-budget`.

## Baseline Contract

- Baseline implementation: original ConvIR-B Haze4K `--arch convir --version base`.
- Baseline checkpoint or initialization: official Haze4K ConvIR-B checkpoint
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Dataset and split: Haze4K train/test under
  `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Selector output contract: APDR output must remain exactly A0 during replay;
  residual output is force-zero.
- Calibration boundary: choose budget mapping using train scores only; test is
  read once for gate evaluation.

## Most Valuable Attempt

- Why this is the highest-value next attempt: v0.2R fixed the flat hard-router
  problem with AUC `0.97664` and Spearman `-0.74664`, but easy top-25% mean
  `B_img` stayed too high at `0.146157`.
- Target failure or opportunity: sharpen the action budget so easy images close
  while hard images retain enough budget.
- Cheap preflight evidence: no new residual training; rerun selector-only on
  AutoDL, save train/test scores, run train-only conservative budget grid, and
  optionally compute oracle residual ceiling.
- Earliest decisive gate: conservative budget replay gate.
- What success decides: if budget replay passes, oracle residual ceiling may
  decide whether a residual stop20 stage is worth launching.
- What failure decides: do not train residual; move to explicit SafeReferenceVeto
  or another easy-close mechanism.

## Hypothesis

v0.2R learned a useful ranking score:

```text
z_img = GlobalDensityRouter(...)
```

but its calibrated budget:

```text
B_img = sigmoid((z_img - tau_train) / temperature_train)
```

behaved like a ranking probability, not a conservative action budget. v0.2RC
keeps the APDR-v0.2R selector and searches train-only conservative maps:

```text
B_cons = B_raw ^ gamma
B_cons = relu((B_raw - b0) / (1 - b0)) ^ gamma
B_cons = sigmoid((z_img - tau2) / T2) ^ gamma
```

The replay must not use test metrics to choose `gamma`, `b0`, `tau2`, or `T2`.

## Gates

Budget replay gate:

| Rule | Required |
| --- | ---: |
| zero-residual output max diff vs A0 | `< 1e-6` |
| AUC by `z_img` | `>= 0.95` |
| Spearman(`z_img`, A0 PSNR) | `<= -0.70` |
| hard bottom-25% mean `B_cons` | `>= 0.35` |
| easy top-25% mean `B_cons` | `<= 0.03` |
| strong-reference mean `B_cons` | `<= 0.03` |
| hard/easy `B_cons` ratio | `>= 10` |
| hard BCE after calibration | `<= 0.55` |

Oracle residual ceiling gate, only evaluated as a decision gate if replay passes:

| Rule | Required |
| --- | ---: |
| oracle mean PSNR delta | `>= +0.050 dB` |
| oracle hard bottom-25% delta | `>= +0.150 dB` |
| oracle easy top-25% delta | `>= -0.005 dB` |
| oracle strong-reference regressions | `<= 5 / 250` |
| oracle severe regressions | `== 0 / 1000` |

Stop rule:

- If replay gate fails, stop v0.2RC and do not compute residual training.
- If replay passes but oracle fails, stop before residual training.
- If both replay and oracle pass, write a separate residual stop20 plan.

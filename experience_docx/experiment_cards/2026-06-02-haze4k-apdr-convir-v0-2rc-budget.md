# Haze4K APDR-ConvIR v0.2RC Conservative Budget

Date: 2026-06-02

Status: completed cloud conservative-budget replay; replay gate failed.

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

## Result Summary

- Cloud run: AutoDL `autodl-dehaze3`, tmux session
  `apdr_v0_2rc_budget_20260602`, completed at `2026-06-03T00:00:00+08:00`.
- Source snapshot on cloud: local archive of GitHub commit
  `6f7bf1dd2badaf6bc14aa14b1c4e091e08ff1f02`.
- Architecture preflight: pass; residual output remained zero during replay.
- Candidate grid: `489` budget maps; `33` passed train constraints; `0`
  passed the full held-out replay gate.
- Selected train-only candidate: `platt_tau-1.4839_t1_g4`.
- Train selected metrics: hard mean `0.373883`, easy mean `0.008297`,
  hard/easy ratio `45.0604`, calibration BCE `0.540913`; train constraints
  passed.
- Held-out replay metrics: AUC `0.97664`, Spearman `-0.74664`, hard mean
  `0.378346`, easy/strong-reference mean `0.002531`, hard/easy ratio
  `149.481`, zero-output diff `0.0`.
- Held-out failure: calibration BCE `1.619142` versus required `<= 0.55`.
- Oracle residual ceiling was not evaluated because the replay gate failed.

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
| selected candidate train constraints | pass |
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

## Decision

`FAIL_STOP_APDR_V0_2RC_BUDGET_CALIBRATION`.

The conservative budget map solves the easy/strong-reference over-open problem:
held-out easy mean budget fell from v0.2R's `0.146157` to `0.002531`.
However, no candidate simultaneously satisfied held-out hard/easy budget
constraints and calibration BCE. The selected train-valid candidate preserved
ranking and closed easy images, but held-out calibration BCE rose to `1.619142`.

Do not launch residual training from v0.2RC. The next APDR route should add an
explicit easy/strong-reference veto or otherwise decouple hard detection from
safe-reference closure, rather than making the single global budget sharper.

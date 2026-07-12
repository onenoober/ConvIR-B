# Haze4K v5 CHD-RM v3s Zero-Init Delta-u Direction Repair

Date: 2026-07-13

Status: `PLANNED`

## Scope

- Project: ConvIR-B Haze4K.
- Model family: frozen v3l direct correction operator plus one low-capacity direction-repair branch.
- Dataset/task: the fixed 1,200 train-derived, clean-reference-grouped OOF Haze4K names from `v3j_controller_train`; no canary or locked test access.
- Primary objective: repair the signed direction of the frozen `.125 -> .25` residual step while preserving the old `.125` anchor and low-action images.
- Main metric: dual-operator grouped-bootstrap LCB95 of fixed `.25` PSNR lift over the frozen old `.25` operator.
- Secondary metrics: `.125` anchor preservation, severe/lower-tail regressions, low-action preservation, float64 block G1, wrong-direction repair fraction, and repair norm.
- Execution environment: `convir-4090` only for runtime.
- GitHub rules commit: `github/main@f999591b8d6d09beef3079d51db6eac53d1ae302`.
- Local WSL: `/home/ubuntu/workspace/ConvIR-B-v3s-delta-u-direction-repair-20260713`.
- Source anchor: `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Route branch: `codex/haze4k-v5-v3s-delta-u-direction-repair-20260713`.
- Cloud `REMOTE_REPO`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3s-delta-u-direction-repair-20260713`.
- Cloud `RUN_ROOT`: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3s_delta_u_direction_repair_20260713`.
- Cloud `EVID_STAGE`: `$REMOTE_REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3s_delta_u_direction_repair_20260713`.
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Baseline And Source Contract

- Baseline: frozen v3p reconstruction of the canonical v3l `D_ref` and `D_rep` context direct operators, not a new scorer.
- Official base checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`, SHA-256 `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Frozen control checkpoint: `ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl`, SHA-256 `08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2`.
- Frozen source checkout: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3p-canonical-signed-gain-20260712@555fd008e29f02128564f2fad41d0095ee44f5ea`; it supplies only exact frozen operator code/assets. Current rules remain `github/main@f999591b`.
- Preprocessing/metric: frozen v3p load, padding, direct-head context, hard output gate, float32 add-and-clamp path. Formal G1 is regenerated from new renders with float64 RGB SSE and exact block16 coverage.
- The generic `main.py` optimizer is not used. The route runner trains only the new `DIRR_*` module and saves fixed final per-fold states in cloud `RUN_ROOT`; no resume or checkpoint selection exists.

## Most Valuable Attempt

v3r found worst-operator LCB95 lift over old `.25` of only `+0.000220 dB` for scale and `+0.004571 dB` for RGB channel scale, below the preregistered `+0.005 dB` SESOI. Its bounded direction-line ceiling was `+0.280496 dB`, fixed more than `99.99%` of active blocks, and had near-zero harmful SSE. Roughly 36.5% of active blocks are wrong-direction, with median required rotation around 81 degrees. This route therefore changes direction only; more scorer, threshold, calibration, or amplitude experiments are forbidden.

## Hypothesis And Change

```text
If a bounded Delta-u field is learned after the frozen correction step, fixed
alpha=.25 should gain over the old alpha=.25 operator because the field can
correct signed direction that scale and channel-scale cannot change, while the
old alpha=.125 anchor and old support preserve low-action images.
```

- Module: `Dehazing/ITS/models/direction_repair.py::DIRR_DeltaU`.
- Input: full-resolution `[I_hazy, y0_base, u_old]` (nine channels).
- Form: `3x3 -> depthwise 3x3 -> 1x1 -> zero-init 1x1`, width 24.
- Output: `Delta u = support_old * (2B * tanh(raw))`, with frozen v3j RGB action bound `B`. The repair cannot open new support.
- Candidate: `clip(y0 + alpha * (u_old + Delta u), 0, 1)` at exactly `.125` and `.25`.
- Initialization: final `DIRR_head` weights and bias are all zero; `Delta u == 0` and the initial candidate exactly equals the old operator.
- Strict load: base, FAM2/D7c control, and each direct-head artifact load strictly with their pinned SHA. No `strict=False`, broad missing-key allowlist, or official layer shape change is allowed.
- Frozen: base ConvIR-B, control path, density/D7c gate, `D_ref`, and `D_rep`, all in `eval()` with `requires_grad=False`.
- Trainable scope: `adapter_only`, `DIRR_*` exactly.
- Disabled: confidence/value/selector heads, threshold/coverage search, action-ladder expansion, teacher-residual regression, backbone/control/gate/direct-head tuning, physics reopen, policy replay, canary, locked test, and checkpoint selection.

## Training Contract

- Formal training: five fixed OOF fits. Each trains on exactly 960 images from four clean-reference folds for six epochs, then evaluates its final state only on the held-out 240 images.
- Scout: first 32 frozen names for eight fixed epochs, an activity check only.
- Optimizer: AdamW, LR `1e-4`, weight decay `1e-5`, gradient clip `0.1`; risk window four images and both frozen operators per update.
- Loss uses actual add-and-clamp renders only:
  `MSE(new .25,J)` + `30 * ReLU(MSE(new .125,J)-MSE(old .125,J))` + `5 * active_block_mean(ReLU(-G1_new))` + `20 * ReLU(MSE(new .25,J)-MSE(old .125,J))` + `40 * CVaR25` of image harm in the risk window + `0.02 * mean(abs(Delta u/(2B)))`.
- This combines continuous haze conditioning via `I_hazy`, regional adaptive residual modulation via `Delta u`, and low-haze protection via frozen support plus `.125` anchor/harm/CVaR penalties.
- Profile: static contract -> exact no-op smoke -> fixed-32 scout -> five-fold fixed-budget train -> canonical OOF decision.

## Static Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| route source | clean new branch from immutable anchor `3b4da354` | complete |
| model syntax | module and runner parse locally | complete |
| source identity | all checkpoint, artifact, manifest, and bound hashes match pinned values | pending cloud dynamic check |
| partial load | all frozen states strict; only `DIRR_*` are new | pending S0 |
| no-op | zero branch replays frozen old `.125` within `1e-6 dB` | pending S0 |
| data protection | exact train OOF names/folds; no test-like path | pending S0 |

Pinned identities: split `c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7`; operator manifest `1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84`; fixed reference rows `b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1`; RGB bounds `485ea12ff14c33b87105a50b6d118a9937c7e7f1b113062fe03d91eef3c9cc21`; density `1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f`; D7c `09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361`.

## Gates

| Stage | Question | Scope | Predeclared gate | `PASS` authorizes |
| --- | --- | --- | --- | --- |
| S0 | Does zero-init reproduce the frozen old operator? | 32 names x two operators | structural: exact zero `Delta u`/prediction difference and old reference replay <= `1e-6 dB` | S1 only |
| S1 | Is the branch numerically active under real rendered loss? | 32 names, 8 epochs | finite loss/gradients, mean abs `Delta u > 1e-6`, final render loss below initial | S2 only |
| S2 | Can all fixed folds train under the written contract? | 5 x 960 images, 6 epochs | exact fold count/size and final checkpoint hash manifests | S3 only |
| S3 | Does the fixed learned operator add utility, correct signed direction, and preserve anchors? | 1,200 held-out OOF images/operator | worst-operator LCB95 vs old `.25` >= `+.020 dB`; LCB95 vs old `.125` and low-action old `.125` >= `-.005 dB`; zero severe <= `-.2 dB`; >=20% old wrong-direction blocks repaired; positive mean G1 change; float64 aggregation <= `1e-8` | new confirmation-contract design only |

`+.020 dB` is twice v3p A2's fixed-action utility floor over uniform `.25`. The 20% repair line is a material fraction of the independently observed 36.5% wrong-direction population. The `-.005 dB` floor is v3r's direction-repair SESOI as a preservation budget. The low-action slice is the bottom frozen-step-energy quartile over all 1,200 names and does not use clean targets or new outcomes.

## Analysis And Decision

- Analysis unit: clean-reference image group. `D_ref/D_rep` are paired robustness environments, never independent evidence.
- Formal labels: `epsilon_G = 2 * (1e-12 + 1e-12 * max(abs(L_.125), abs(L_.25)))`; old v3p/v3r labels are not reused.
- Uncertainty: 4,000 seeded grouped bootstrap resamples, with the worst operator deciding utility.
- Cloud-only: checkpoints, raw image/block rows, outputs, and logs. Compact evidence: source manifest, closeouts, fixed histories, operator summary, canonical contract, and README.
- Initial label: `V3S_START_ZERO_INIT_DELTA_U_DIRECTION_REPAIR_ONLY`.
- A pass authorizes only a separately preregistered fixed-operator confirmation design. It does not authorize confidence training, threshold search, policy replay, canary, locked test, or deployment.
- A failure stops this low-capacity `Delta u` representation/loss contract; structural failure is not scientific negative evidence.

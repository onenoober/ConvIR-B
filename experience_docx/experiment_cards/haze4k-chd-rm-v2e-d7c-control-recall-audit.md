# Haze4K CHD-RM v2e D7c Control Recall Audit

Status: `PAUSE_V2E_D7C_RP_NO_SAFE_RECALL_PROTECTED_POINT_NO_V3`

Evidence root:

`experience_docx/experiment_logs/haze4k_v5_chd_rm_v2e_d7c_control_recall_audit_20260709/`

Runtime source:

- Host: `convir-4090`
- Branch: `codex/haze4k-v5-v2e-chd-rm-d7c-control-recall-audit`
- Base: v2d D7c route commit `ca2ca626c22b407c020025d3ff5b16dacd96bb7d`
- Locked Haze4K test usage: none

Route identity:

v2e is a gated control and recall-protection audit after v2d. It keeps ConvIR-B frozen, keeps D3 density frozen, keeps RARM disconnected, does not run D2, and does not enter v3. It tests whether the D7c top-k `R_need` map is real signal rather than density/context proxy, and whether low-density high-need regions can be recalled safely.

Main audit result:

- D7c top-k reproduces v2d: Spearman `0.5175`, AUROC `0.8456`, AUPRC `0.6442`, coverage `0.3027`, precision `0.6313`, recall `0.4493`.
- Safety is strong at the frozen threshold: false-global `0.0030`, false-p90 `0.0246`, false-p95 `0.0476`.
- Fixed image-level permutation is clean: empirical p `0.0099`, original Spearman `0.5165` vs p99 null `0.0447`.
- D7c beats density-only matched threshold by Spearman `0.2178`, AUROC `0.1481`, AUPRC `0.1046`, precision `0.0888`.
- LDHN support is real: pixel coverage `0.0899`, image coverage `0.9983`.
- Frozen D7c top-k LDHN recall is only `0.0370`, below the `0.10` protection line.

D7c-RP follow-up:

D7c-RP was run because controls were clean but LDHN recall failed. Five recall-protected heads were initialized from D7c top-k and trained with ConvIR-B/D3 frozen. The sweep found no safe operating point:

| variant | pass | false-p90 | false-p95 | LDHN recall |
| --- | ---: | ---: | ---: | ---: |
| d7c_rp_lam05_r3_lr2e4 | False | 0.0326 | 0.0741 | 0.0491 |
| d7c_rp_lam10_r3_lr2e4 | False | 0.0599 | 0.2069 | 0.1096 |
| d7c_rp_lam15_r3_lr15e4 | False | 0.0714 | 0.2941 | 0.1428 |
| d7c_rp_lam20_r5_lr15e4 | False | 0.1081 | 0.3545 | 0.1667 |
| d7c_rp_lam30_r5_lr1e4 | False | 0.1684 | 0.5348 | 0.1822 |

Decision:

`PAUSE_V2E_D7C_RP_NO_SAFE_RECALL_PROTECTED_POINT_NO_V3`

The current D7c family has real `R_need` signal, but the safe top-k candidate under-recovers LDHN regions and RP recovery breaks false-tail safety. Do not run v3, D2, RARM connection/training, or locked Haze4K test from this state.

Next allowed work:

- Diagnose the D7c safety/recall tradeoff or revise the `R_need` target/head inside a new gated v2-family route.
- Keep any new follow-up frozen-side-head only until it passes controls and LDHN safety together.

Forbidden:

- D2.
- RARM connection/training.
- v3 expansion or no-op RARM audit from this failed RP state.
- Locked Haze4K test.

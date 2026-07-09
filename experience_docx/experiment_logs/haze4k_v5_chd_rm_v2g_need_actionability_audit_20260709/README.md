# CHD-RM v2g Need Actionability Audit

Status: `PAUSED_AFTER_G4B`

Decision label: `PAUSE_G4B_SELECTIVE_PROBE_NO_SAFE_IMPROVEMENT_NO_F5_NO_V3`

Purpose: decide whether global LDHN is a CHD-RM-compatible actionable target or an over-broad post-A0 residual target.

Current policy: locked Haze4K test, D2, RARM, v3, and F5 remain blocked. G4b saved no probe weights/checkpoints.

## Primary Evidence

- route card: `experience_docx/experiment_cards/haze4k-chd-rm-v2g-need-actionability-audit.md`
- central index: `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`
- final closeout: `v2g_final_closeout.json`
- overall summary: `v2g_overall_result_summary.md`
- G1 semantic audit: `ldhn_semantic_audit_summary.json`, `ldhn_actionability_taxonomy.csv`
- G2b oracle gain: `v2g_g2b_oracle_gain_closeout.json`, `ldhn_oracle_gain_by_region.csv`
- G3 target definition: `actionable_need_target_definition.md`, `d7c_actionable_target_gate_summary.json`
- G4a controls: `v2g_g4a_actionability_control_closeout.json`, `g4a_actionability_control_audit_summary.csv`
- G4b selective probe screen: `v2g_g4b_selective_probe_closeout.json`, `v2g_g4b_selective_probe_summary.md`, `g4b_selective_probe_summary.csv`

## Key Result

Global LDHN is over-broad as a hard RARM-positive target. LDHN coverage is about `0.089890`, but the isolated fraction is `0.890713`; only `0.109287` is adjacent to haze. D7c recalls haze-adjacent LDHN better than isolated LDHN (`0.155904` vs `0.022366`), which supports an actionability split rather than a global LDHN recall gate.

G3 defines a three-state target: actionable positive, confident low-risk negative, and ignore/abstain. Under that target, D7c has val action recall `0.548312`, low-adjacent recall `0.155904`, negative false rate `0.002974`, and isolated-LDHN hit rate `0.022366`.

G4a shows D7c beats the deployable density-only matched control under the three-state target: action recall `0.548312` vs `0.454247`, low-adjacent recall `0.155904` vs `0.113905`, negative false rate `0.002974` vs `0.049584`, and AUROC action-vs-negative `0.969589` vs `0.872087`.

G4b ran the authorized small selective-head/probe screen. The best probe was `context_image_density_linear`, but it failed the predeclared safe-improvement gate against D7c: action recall `0.488995` vs `0.548312`, low-adjacent recall `0.076751` vs `0.155904`, negative false rate `0.004045` vs `0.002974`, and AUROC action-vs-negative `0.937536` vs `0.969589`.

## Decision

v2g supports the bottleneck diagnosis: the old `R_need = post-A0 residual magnitude` target is not directly actionable enough for RARM. D7c remains the best deployable prior under the v2g three-state target, and G4b does not authorize F5, v3, RARM, D2, or locked-test access.

Runtime metadata:

- Host: RTX4090
- Branch: `codex/haze4k-v5-v2g-chd-rm-need-actionability-audit`
- Head: `044b779`
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Completed through G4b: `2026-07-09T21:33:17+08:00`

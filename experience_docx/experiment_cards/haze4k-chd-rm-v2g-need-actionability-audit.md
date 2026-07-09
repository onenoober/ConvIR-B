# Haze4K CHD-RM v2g Need Actionability Audit

Status: `PAUSED_AFTER_G4A`

Decision label: `PAUSE_V2G_ACTIONABLE_TARGET_DEFINED_D7C_BEATS_DENSITY_CONTROLS_NO_F5_NO_V3_YET`

v2g is a diagnostic route after v2f/F4b found no safe LDHN operating point. It audits whether the current `R_need = post-A0 residual magnitude` target is actionable enough for a future RARM signal.

Forbidden: locked Haze4K test, D2, RARM connection/training, v3, F5, and F4/F4b strength sweeps.

Evidence root: `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2g_need_actionability_audit_20260709/`.

## Result

G0/G1/G2/G2b/G3/G4a completed on `convir-4090` / `RTX4090` with cloud Python `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`. No locked Haze4K test, D2, RARM, v3, F5, or new head training was run.

The core bottleneck diagnosis is supported: global LDHN is over-broad as a hard RARM-positive target. Most LDHN support is isolated from haze adjacency (`0.890713` isolated vs `0.109287` adjacent-to-haze), while D7c preferentially recalls the haze-adjacent subset (`0.155904`) and mostly avoids isolated LDHN (`0.022366`).

G3 defined a three-state target: actionable positive, confident low-risk negative, and ignore/abstain. Under this target, D7c val action recall is `0.548312`, low-adjacent recall is `0.155904`, negative false rate is `0.002974`, and isolated-LDHN hit rate is `0.022366`.

G4a controls show D7c beats the deployable density-only matched control: action recall `0.548312` vs `0.454247`, low-adjacent recall `0.155904` vs `0.113905`, negative false rate `0.002974` vs `0.049584`, and action-vs-negative AUROC `0.969589` vs `0.872087`.

## Next Gate

The next possible work is only a small G4b selective-head/probe screen under the three-state actionable target, with controls and anti-ignore-collapse checks. This route does not authorize F5, v3, RARM, D2, or locked-test access.

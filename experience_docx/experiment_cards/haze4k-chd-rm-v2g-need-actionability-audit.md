# Haze4K CHD-RM v2g Need Actionability Audit

Status: `PAUSED_AFTER_G4B`

Decision label: `PAUSE_G4B_SELECTIVE_PROBE_NO_SAFE_IMPROVEMENT_NO_F5_NO_V3`

v2g is a diagnostic route after v2f/F4b found no safe LDHN operating point. It audits whether the current `R_need = post-A0 residual magnitude` target is actionable enough for a future RARM signal.

Forbidden: locked Haze4K test, D2, RARM connection/training, v3, F5, and F4/F4b strength sweeps.

Evidence root: `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2g_need_actionability_audit_20260709/`.

## Result

G0/G1/G2/G2b/G3/G4a/G4b completed on `convir-4090` / `RTX4090` with cloud Python `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`. No locked Haze4K test, D2, RARM, v3, F5, RARM connection, or saved probe weights/checkpoints were used.

The core bottleneck diagnosis is supported: global LDHN is over-broad as a hard RARM-positive target. Most LDHN support is isolated from haze adjacency (`0.890713` isolated vs `0.109287` adjacent-to-haze), while D7c preferentially recalls the haze-adjacent subset (`0.155904`) and mostly avoids isolated LDHN (`0.022366`).

G3 defined a three-state target: actionable positive, confident low-risk negative, and ignore/abstain. Under this target, D7c val action recall is `0.548312`, low-adjacent recall is `0.155904`, negative false rate is `0.002974`, and isolated-LDHN hit rate is `0.022366`.

G4a controls show D7c beats the deployable density-only matched control: action recall `0.548312` vs `0.454247`, low-adjacent recall `0.155904` vs `0.113905`, negative false rate `0.002974` vs `0.049584`, and action-vs-negative AUROC `0.969589` vs `0.872087`.

G4b ran the authorized small selective-head/probe screen under the same three-state target. The best probe, `context_image_density_linear`, did not safely improve over D7c: action recall `0.488995` vs `0.548312`, low-adjacent recall `0.076751` vs `0.155904`, negative false rate `0.004045` vs `0.002974`, and action-vs-negative AUROC `0.937536` vs `0.969589`.

## Decision

D7c remains the best deployable prior under the v2g three-state target. G4b does not authorize F5, v3, RARM, D2, or locked-test access. Any continuation needs a new written target/features decision rather than another small probe expansion.

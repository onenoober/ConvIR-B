# Haze4K DPGA Tail-Control Artifact Manifest

Date: 2026-06-04

## Primary Evidence

| Path | Use |
| --- | --- |
| `experience_docx/experiment_cards/2026-06-04-haze4k-convir-v1-1-dpga-tail-control.md` | Route card and final route decision. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/README.md` | Script/output map for this route. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/status.txt` | Chronological status and artifact pointers from AutoDL. |

## Runtime Diagnostics

| Path | Use |
| --- | --- |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/runtime_diagnostics/dpga_runtime_variants_summary.json` | Runtime diagnostic summary. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/runtime_diagnostics/dpga_module_ablation_best_final.csv` | Module ablation summary. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/runtime_diagnostics/dpga_scale_sweep_best_final.csv` | Runtime scale sweep summary. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/runtime_diagnostics/dpga_module_ablation_per_image.csv` | Per-image module ablation evidence. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/runtime_diagnostics/dpga_scale_sweep_per_image.csv` | Per-image scale sweep evidence. |

## v1.1

| Path | Use |
| --- | --- |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_1_decision/dpga_v1_1_training_decision.json` | Machine-readable launch decision. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_1_val_inner_eval/gate_dpga_v1_1_val_inner.json` | v1.1 validation gate. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_1_val_inner_eval/scout_eval_compare_v1_1_val_inner_best_vs_a0.json` | Best-vs-A0 summary. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_1_val_inner_eval/scout_eval_per_image_v1_1_val_inner_best_vs_a0.csv` | Best per-image table. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_1_failure_analysis/dpga_v1_1_val_inner_failure_analysis.md` | Human-readable failure analysis. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/train_ConvIR-Haze4K-DPGA-v1.1-tail-control-shallow-scale0p25-seed3407-20260604.log` | Training log. |

## v1.2

| Path | Use |
| --- | --- |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_2_decision/dpga_v1_2_training_decision.json` | Machine-readable launch decision. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_2_val_inner_eval/gate_dpga_v1_2_val_inner.json` | v1.2 validation gate. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_2_val_inner_eval/scout_eval_compare_v1_2_val_inner_best_vs_a0.json` | Best-vs-A0 summary. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_2_val_inner_eval/scout_eval_per_image_v1_2_val_inner_best_vs_a0.csv` | Best per-image table. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/v1_2_failure_analysis/dpga_v1_2_val_inner_failure_analysis.md` | Human-readable failure analysis. |
| `experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/train_ConvIR-Haze4K-DPGA-v1.2-hard-gain-shallow-scale0p5-anchor0p04-seed3407-20260604.log` | Training log. |

## Excluded

Model checkpoints under `Dehazing/ITS/results/.../Training-Results/*.pkl` stay
on AutoDL and are not part of this GitHub sync.

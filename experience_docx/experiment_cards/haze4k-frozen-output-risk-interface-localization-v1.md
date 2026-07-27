# Haze4K Frozen-Output Risk Interface Localization v1

Status: PLANNED

- Route id: haze4k-frozen-output-risk-interface-localization-v1
- First operation: HAZE4K_FROZEN_OUTPUT_RISK_INTERFACE_LOCALIZE
- Program contract: experience_docx/research_programs/haze4k_frozen_output_risk_interface_localization_v1.json
- Experiment spec: experience_docx/experiment_specs/haze4k-frozen-output-risk-interface-localization-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The archived parent trained each harm logit with positive-class BCE weight 2.0, then treated q=sigmoid(z) as an unweighted probability and applied q<=0.10. Under the ideal weighted-BCE population relation q=2p/(1+p), that rule corresponds to p<=0.05263, not the frozen 0.10 risk limit. This one-shot development route therefore reuses the exact parent OOF and historical test-development raw predictions and scene caches, verifies their full SHA inventory, reproduces archived P00 exactly, and changes only the declared probability semantics to p=q/(2-q). The candidate is not assumed calibrated. It is tested against utility-only, observable-utility plus GT-risk, GT-utility plus observable-risk, GT/GT, uniform, shuffled, and prediction-permutation controls. Original clear scene remains the only independent unit. Complete image replay covers PSNR, RGB SSIM, and RGB mean-bias safety; selected-area harm uses scene-block uncertainty. The main 0.05 dB utility margin, 0.10 risk limit, 0.025 dB precision target, action scales, OOF folds, and data roles do not change. A 1 percent action-area LCB and 20 percent active-scene LCB prevent safe near-total abstention from passing. Candidate-confirmation, NH-Haze, ITS, and OTS remain unread.

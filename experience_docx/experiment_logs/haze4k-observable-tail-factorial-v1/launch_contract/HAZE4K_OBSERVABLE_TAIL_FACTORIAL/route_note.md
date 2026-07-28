# Haze4K Observable Tail Factorial v1

Status: PLANNED

- Route id: haze4k-observable-tail-factorial-v1
- First operation: HAZE4K_OBSERVABLE_TAIL_FACTORIAL
- Program contract: experience_docx/research_programs/haze4k_observable_tail_factorial_v1.json
- Experiment spec: experience_docx/experiment_specs/haze4k-observable-tail-factorial-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The archived weighted-BCE probability-semantics route was precise and nondegenerate but recovered only 0.015888 dB, while privileged controls localized 57.5 percent of regret to risk and 42.5 percent to utility. This orthogonal development route therefore changes the observable representation and learner rather than retrying calibration or thresholds. It executes a frozen 3x3 factorial: R0 is the exact 104-channel baseline; R1 adds learned 8x8-subtile shape-preserving residual encoding; R2 adds multiscale ConvIR context and deterministic flip/0.75-scale disagreement. L0 is the exact shared Huber/weighted-BCE control, L1 separates utility and risk heads, and L2 learns a distributional ordinal tail tied to -0.10 dB harm and +0.05 dB material utility. Each cell has five group-respecting folds and three paired seeds. The 150 calibration scenes are group-cross-fitted for correction and selection only. The selected cell is hash-frozen before any 600-scene OOF outcome is replayed. All nine cells receive OOF and historical test-development PSNR replay; the selected cell receives complete PSNR, RGB SSIM, color, selected-area harm, uniform, shuffle, permutation, utility-only, mixed privileged, and GT/GT controls. Candidate-confirmation, NH-Haze, ITS, and OTS remain unread.

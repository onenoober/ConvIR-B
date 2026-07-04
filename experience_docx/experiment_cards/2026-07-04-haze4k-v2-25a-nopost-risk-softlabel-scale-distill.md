# Haze4K v2.25A NoPost Risk Soft-Label / Scale Distillation

Date: 2026-07-04
Status: V225A_RISK_CALIBRATION_GATE_FAIL_NORMAL_PAUSE

## Purpose

Follow v2.24 Case A: train-time risk-head collapse. This route trains only the
NoPost risk/context head with v2.21 `unsafe_action_probability` soft labels and
`risk_scale` distillation. Action heads are frozen. No new action training and
no locked Haze4K test are allowed by this screen.

## Gates

- probability std must recover (`>= 0.05`);
- ROC-AUC vs v2.21 unsafe label `>= 0.85`;
- AP `>= 0.45`;
- ECE10 `<= 0.12`;
- target probability MAE `<= 0.20`;
- locked test untouched.

Failure pauses the route. Passing only authorizes a post-train factorial rescue
review, not locked test or promotion.

## Result

The cloud run completed on `convir-4090` from the immutable official
architecture anchor. Preflight passed, action heads stayed frozen, and the
locked Haze4K test was not touched.

OOF risk calibration did not recover the collapsed risk head:

- probability std: `0.0016692509020246116` (gate `>= 0.05`, failed);
- ROC-AUC: `0.5500802065808521` (gate `>= 0.85`, failed);
- AP: `0.4937745923792122` (gate `>= 0.45`, passed);
- ECE10: `0.05257191310326259` (gate `<= 0.12`, passed);
- target probability MAE: `0.24880989863237726` (gate `<= 0.20`, failed).

Each fold produced a constant validation probability (`trained_prob_std=0.0`),
so this is a scientific calibration failure rather than an infrastructure
failure. Per the predeclared gate, v2.25A stops here; no post-train factorial
rescue, action joint training, or locked-test evaluation is authorized from
this result.

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_25a_nopost_risk_softlabel_scale_distill_20260704/`.

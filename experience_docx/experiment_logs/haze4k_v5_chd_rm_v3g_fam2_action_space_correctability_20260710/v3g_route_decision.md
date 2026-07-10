# v3g Route Decision

Date: 2026-07-10

Decision:
`V3G_AUTHORIZE_NO_TRAINING_FAM2_ACTION_SPACE_CORRECTABILITY_AUDIT_ONLY`

v3g is authorized only to test whether the v3f output-space oracle is realizable
inside the true FAM2 action space. It may compute alpha gradients and
finite-difference counterfactual forwards on internal val-inner data. No
training, checkpoint-producing run, v3d continuation, v4/RARM expansion, canary
expansion, or locked-test access is authorized.

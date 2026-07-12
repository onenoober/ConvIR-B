# v3v Safety-Curriculum Evidence

Status: `COMPLETED_GATE_FAIL`; the safety phase loses rendered activity.

Raw cloud runtime root:
`/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3v_safety_curriculum_activation_20260713`.

v3v uses the v3u output-side head with an eight-epoch render-only warmup and an
eight-epoch anchor/margin/harm/CVaR phase. Repair penalty remains zero. S0 must
pass exact no-op before S1; S1 can authorize only a later safety-training
contract design.

Only compact manifests, closeouts, summary, history, and this README belong
here. Raw cloud outputs remain outside Git.

S0 exact no-op passed. In S1, the render-only warmup passed, but the abrupt
full-weight anchor/margin/harm/CVaR switch made final rendered MSE `0.04228%`
worse than initialization. Safety diagnostics were non-worse than v3u. No
training, policy, canary, or locked-test action is authorized.

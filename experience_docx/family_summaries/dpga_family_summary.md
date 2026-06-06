# DPGA Family Summary

Date: 2026-06-05

Status: UDP-Lite/frozen small-adapter family is sufficiently diagnosed and low
success. Official FullUDP remains useful as a hard-expert signal. The fixed
v1.6 A0+UDP expert switch failed locked-test promotion, and the v1.7 full-train
risk-controlled shrink/mix router failed OOF plus train-heldout gates. v1.8 is
the planned post-diagnosis execution queue for stronger table-only router
analysis, data/domain preflight, and BiDPFM1 fusion-neighbor multi-seed
training/eval.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-0-dpga-lite.md`
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-1-dpga-tail-control.md`
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-3-hsdf.md`
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-4-udp-lite.md`
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-4b-bidpfm1.md`
  - `../experiment_cards/2026-06-05-haze4k-convir-v1-5-full-udpnet.md`
  - `../experiment_cards/2026-06-05-haze4k-convir-v1-7-rc-expert-mix.md`
  - `../experiment_cards/2026-06-06-haze4k-convir-v1-8-execution-queue.md`
- Evidence roots:
  - `../experiment_logs/haze4k_dpga_lite_20260604/`
  - `../experiment_logs/haze4k_dpga_tail_control_20260604/`
  - `../experiment_logs/haze4k_dpga_v13_hsdf_20260604/`
  - `../experiment_logs/haze4k_udp_lite_v14_20260604/`
  - `../experiment_logs/haze4k_udp_lite_v14b_bidpfm1_20260604/`
  - `../experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605/`
  - `../experiment_logs/haze4k_rc_expert_switch_v16_20260605/`
  - `../experiment_logs/haze4k_v17_rc_expert_mix_20260605/`
  - `../experiment_logs/haze4k_v18_execution_queue_20260606/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| DPGA-Lite v1.0 | `Best.pkl` full-test mean `+0.0312 dB`, SSIM positive, hard `+0.0146 dB`, easy `+0.0209 dB`, strong-reference regressions `105/250`; exact `model_20`/`Final` mean `+0.0193 dB` and hard `+0.0037 dB`. | `DPGA_LITE_ADAPTER_ONLY_MIN_POSITIVE_BEST_BORDERLINE_FINAL`; directionally positive but not promotion-ready. |
| DPGA tail-control v1.1 | Shallow-only scale `0.25`, anchor `0.08`; Best mean `+0.037036 dB`, hard bottom-25% `+0.023367 dB`. | Failed `val_inner` hard gate `>= +0.030 dB`; locked Haze4K test blocked. |
| DPGA tail-control v1.2 | Shallow-only scale `0.5`, anchor `0.04`; Best mean `+0.042656 dB`, hard bottom-25% `+0.026225 dB`, worst `<= -0.20 dB` regressions rose to `16/300`. | Failed hard gate and worsened tail risk; no higher-scale follow-up without new diagnostic. |
| DPGA-v1.3A HSDF | Best `val_regular` mean `+0.026333 dB`; Best `val_hard` hard bottom-25 `+0.022099 dB`. | Loss-mask mechanism improved safety but missed hard gate; authorized only v1.3B diagnostic, not locked test. |
| DPGA-v1.3B HSDF | Best `val_regular` mean `+0.025839 dB`; Best `val_hard` hard bottom-25 `+0.023642 dB`; positive ratio `0.586667`; strong regression ratio `0.200000`; corrected bottleneck-only runtime ablation mean about `+0.000824 dB`. | `FAIL_STOP_V13B_HARD_GATED_BOTTLENECK`; locked test blocked. |
| ConvIR-Dehaze-v1.4-UDP-Lite | v1.4A adapter-only completed and failed gate: Best `val_regular` mean `+0.028294 dB`, Best `val_hard` mean `+0.020340 dB`, hard bottom-25 `+0.022275 dB`, positive ratio `0.586667`, worst count `19`. Ablation shows `DPFM1-only` is safer/stronger (`val_hard` mean `+0.026774 dB`, worst `0`) while `DPFM2-only` is negative. | `FAIL_V14A_ADAPTER_ONLY_FULL_DPFM123`; locked test blocked. Do not micro-tune full DPFM123 scale/gate; only DPFM1-focused diagnostic or v1.4B fusion-neighbor partial unfreeze is evidence-supported. |
| ConvIR-Dehaze-v1.4B-BiDPFM1 | `udp_bi`, `active_adapters=dpfm1`, `active_adapter_only` completed. Best `val_regular` mean `+0.028624 dB`, positive ratio `0.536667`, worst count `17`, strong ratio `0.28`; Best `val_hard` mean `+0.023429 dB`, hard bottom-25 `+0.020760 dB`, worst count `8`. | `FAIL_STOP_V14B_BIDPFM1_ADAPTER_ONLY`; locked test blocked; do not rerun BiDPFM1-only scale/gate tuning. |
| ConvIR-Dehaze-v1.5-FullUDP Phase 0 | Official checkpoint sha256 `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291` evaluated on `val_regular` and `val_hard`. `val_hard` is strong (`+0.4260 dB` mean, `+0.6212 dB` hard bottom-25), but `val_regular` fails (`-0.3020 dB` mean, easy top-25 `-0.7969 dB`), SSIM deltas are negative, strong regression ratios are `0.6133` regular and `0.44` hard, and worst counts are `148/300` regular plus `104/300` hard. | `PHASE0_REPRODUCTION_GATE_FAIL`; do not start transplant/distillation/locked test from this checkpoint protocol; keep only as hard-gain diagnostic evidence. |
| ConvIR-Dehaze-v1.6-RCExpertSwitch | A0+UDP oracle passes strongly: mean `+0.7417 dB`, hard bottom-25 `+1.0038 dB`, easy top-25 `+0.5958 dB`, no strong/worst regressions. True 5-fold OOF threshold switch over `udp_switch_feature_table` passes internal gates: mean `+0.2353 dB`, hard bottom-25 `+0.5127 dB`, easy top-25 `+0.0557 dB`, SSIM `+0.000095`, coverage `0.195`, strong ratio `0.0667`, worst ratio `0.0467`. Fixed median policy `udp_a0_luma_shift_mean <= -0.003969017509371043` also passes internal gates. One-shot locked test is positive but fails promotion: mean `+0.0946 dB`, hard bottom-25 `+0.1552 dB`, easy top-25 `-0.0712 dB`, SSIM `+0.000361`, coverage `0.164`, worst ratio `0.066`. | `LOCKED_TEST_FAIL_NO_FURTHER_SELECTION`; UDPNet remains a hard expert, not a global replacement. Do not tune threshold/feature/expert set from locked-test results. |
| ConvIR-Dehaze-v1.7-RCExpertMix | Full-train feature extraction produced `3000` train-derived A0/UDP rows. GT oracle alpha mix is strong: mean `+0.8689 dB`, hard bottom-25 `+0.9623 dB`, easy top-25 `+0.8245 dB`, worst/strong ratios `0`. The selected low-capacity OOF policy had coverage `0.1557`, mean `+0.1079 dB`, hard bottom-25 `+0.1417 dB`, easy top-25 `+0.1020 dB`, worst ratio `0.0067`, strong ratio `0.0107`, and fold utility pass count `0/5`. Train-heldout confirmation was mean `+0.0945 dB`, hard bottom-25 `+0.1297 dB`, easy top-25 `+0.0597 dB`, worst ratio `0.0033`, strong ratio `0.0282`. | `COMPLETED_GATE_FAIL_LOCKED_TEST_BLOCKED`; the expert-bank oracle remains useful, but the tested full-train low-capacity risk-control router is not deployable. |
| ConvIR-Dehaze-v1.8-ExecutionQueue | Completed queue from the latest diagnosis: table-only stronger A0/UDP router policy grid from the v1.7 feature table, Haze4K train-derived data/domain preflight, BiDPFM1 `fusion_neighbor` partial-unfreeze stop20 across 10 seeds, multi-metric checkpoint selection, multi-seed aggregation, and Q5 data/domain-adaptation coverage. Q1 corrected router gate failed, Q2 completed, Q5 completed as domain evidence but failed gates while recording no visible real-haze target-domain data, and Q3/Q4 finished negative after repaired `3407/2026` eval evidence. | `MULTISEED_SCREEN_FAIL_CONTINUE_OTHER_EXPERIMENTS`; all declared items ran to completion on `dehaze1`, locked test stayed blocked, and the final 10-seed aggregate closes this exact BiDPFM1 partial-unfreeze route as negative. |

## Family Verdict

DPGA moved away from unsafe output RGB residuals and places depth/prior
information inside ConvIR feature paths. The frozen ConvIR-B plus
A0-equivalent small-adapter branch is now sufficiently diagnosed as low
success rather than promotion-ready.
DPGA-Lite v1.0 gave the first recent small positive full-test directional
signal without APDR output residuals, full-backbone training, FFT boost,
teacher distillation, or token-wise routing. That signal is still below the
current Haze4K noise-aware promotion standard and was partly test-observed, so
it is not a final improvement claim.

v1.1/v1.2 showed that shallow scale control can keep mean movement positive but
is hard-gain limited. v1.3 showed that hard-selective masking and hard-gated
bottleneck capacity did not deliver the needed hard-bottom gain, and corrected
runtime ablation found almost no useful bottleneck-only contribution.

v1.4-UDP-Lite tested the currently preferred reopen mechanism: zero-init
multi-scale depth/prior fusion (`DPGA_prior_encoder`, `DPGA_dpfm1/2/4`) with
independent zero-init, module-ablation, and depth-quality audit tooling. The
cloud A0-equivalence preflight passed, but v1.4A adapter-only failed the
internal regular+hard gate. The most useful evidence is scale attribution:
`DPFM1-only` is the only strong/safe contributor, full `DPFM1+2+4` raises tail
risk, and `DPFM2-only` is a negative contributor.

v1.4B-BiDPFM1 was the authorized DPFM1-focused follow-up. Its `udp_bi`
A0-equivalence and projection-gradient liveness preflight passed, but
adapter-only training did not clear the internal continue line. The no-training
matrix found `DPFM1+4` has better mean than DPFM1-only but not a clean enough
tail profile for the first route, while DPFM2 remains blocked. The completed
BiDPFM1-only route is stopped; this is not permission to run locked Haze4K test,
revive DPFM2, or perform full multi-scale scale search.

v1.5-FullUDP Phase 0 first hit a checkpoint-acquisition blocker, then reopened
after the official checkpoint was provided on the replacement `dehaze1`. The
controlled internal eval confirms that full UDPNet can move hard samples much
more than UDP-Lite (`val_hard` mean `+0.4260 dB`, hard bottom-25 `+0.6212 dB`),
but the same checkpoint/protocol is not preservation-safe: `val_regular` mean
`-0.3020 dB`, easy top-25 `-0.7969 dB`, negative SSIM deltas, strong regression
ratios `0.6133`/`0.44`, and worst counts `148/300`/`104/300`. This is a
scientific gate failure for using the official checkpoint as an immediate
teacher or transplant authorization, not evidence that depth priors are useless.

v1.6 changes the family conclusion from "FullUDP global replacement failed" to
"official UDPNet is a hard expert candidate." The A0+UDP oracle proves the
expert bank has a large upper bound, and the first true 5-fold OOF
risk-calibrated switch clears the internal Utility and Promotion-style gates
without using PSNR/SSIM delta columns as router features. The fixed policy was
a safe post-router internally: run A0 and UDPNet, compute
`udp_a0_luma_shift_mean`, choose UDPNet only when it is
`<= -0.003969017509371043`, otherwise use A0. However, the one-shot locked
Haze4K confirmation failed the written promotion gate, so the route remains
diagnostic rather than deployable.

v1.7 tested the user's proposed next calibration step without changing the
expert bank: full 3000-image train-derived A0/UDP feature extraction, alpha
shrink/mix, low-capacity gain/risk heads, and train-derived heldout
confirmation. The result strengthens the mechanism reading because the oracle
alpha mix is even stronger than v1.6's fixed-output oracle, and fixed shrinkage
shows why partial UDP residuals are attractive. But the deployable router still
misses the required margins: OOF mean and hard gains are only `+0.1079 dB` and
`+0.1417 dB`, and heldout mean and hard gains are only `+0.0945 dB` and
`+0.1297 dB`. This is a scientific gate failure for the tested low-capacity
risk-control policy, not authorization to touch locked test.

v1.8 was the planned execution response to the latest root-cause diagnosis. It
does not reopen FAM, HardFreq, HazePrior, APDR, DPFM2, or UDPNet-only global
replacement. Instead it uses existing v1.7 intermediate evidence for a stronger
table-only router audit, adds a data/domain preflight, and tests the materially
new capacity mechanism that v1.4A/v1.4B left open: BiDPFM1 with
`fusion_neighbor` partial unfreeze. It also fixes the evidence weakness called
out in the diagnosis by evaluating multiple checkpoints with a regular+hard
multi-metric selector and aggregating across 10 seeds. The queue is explicitly
not allowed to stop after one independent failure; failures become evidence and
the next planned item continues.

The `2026-06-06 05:09 +08:00` v1.8 remote-access blocker was an infrastructure
event, not a scientific gate or training failure. After the user confirmed the
replacement `dehaze1` endpoint `connect.bjb1.seetacloud.com:16124`, local
`~/.ssh/config` and `AGENTS.md` were updated, strict-host-key access was
restored, and the queue resumed at `2026-06-06T10:28:51+08:00` without
rerunning completed seeds. `seed_1701` resumed from `model.pkl` at epoch `6`,
`v18_eval_repair` resumed its wait loop, and refreshed progress artifacts now
label the early `3407/2026` eval import/path failures as engineering states
`EVAL_FAILED_ENGINEERING_REPAIR_PENDING`. The blocker history remains recorded
in
`../experiment_logs/haze4k_v18_execution_queue_20260606/remote_access_blocker_20260606_0509.md`.
Do not run local model runtime as a fallback. Follow-up monitoring at
`2026-06-06T10:56:46+08:00` confirmed that `seed_1701` finished the full
resume train/eval/selection chain and still landed at
`NO_CHECKPOINT_PASSES_ALL_MULTIMETRIC_CHECKS` with `Best` as the diagnostic
label, after which the queue advanced directly into fresh `seed_2222`
training.

Final closeout changed the route from "running queue" to "completed negative
screen." Main queue training/eval reached `seed_5151` completion at
`2026-06-06T13:20:01+08:00`, post-queue repair regenerated missing `3407/2026`
regular+hard compare JSON plus selection JSON and rebuilt the aggregate by
`2026-06-06T13:38:33+08:00`, and final remote verification at
`2026-06-06T14:28:47+08:00` confirmed no active tmux sessions, no related
processes, and idle GPU.

The final 10-seed aggregate is unambiguously negative:

- regular mean PSNR delta mean `-0.05399 dB`, CI95 half-width `0.00794 dB`;
- regular easy-top25 mean `-0.04441 dB`;
- regular mean SSIM delta `-0.0000961`;
- regular strong-regression ratio mean `0.508`;
- hard mean PSNR delta mean `-0.09085 dB`, CI95 half-width `0.01434 dB`;
- hard hard-bottom25 mean `-0.12387 dB`;
- hard mean SSIM delta `-0.0001369`;
- hard strong-regression ratio mean `0.532`;
- all `10/10` selected checkpoint labels are `Best`;
- all `10/10` seed decisions are `NO_CHECKPOINT_PASSES_ALL_MULTIMETRIC_CHECKS`.

Only `n_ge_5` passes in the written multi-seed gate set; every quality and tail
safety gate is `false`. The repaired `3407/2026` runs remain negative after
engineering recovery, so the import-path bug was only an evidence-generation
failure, not the scientific reason for route failure.

## Do Not Repeat Without New Evidence

- Do not promote v1.0 from `Best.pkl` alone; exact stop20/final was borderline
  and the effect size is small relative to the route noise policy.
- Do not run locked Haze4K test for v1.1, v1.2, v1.3A, or v1.3B.
- Do not launch higher-scale shallow DPGA as the next step; v1.2 already raised
  worst-tail regressions to `16/300`.
- Do not continue the current HSDF hard-gated bottleneck route as-is; corrected
  ablation shows bottleneck-only mean contribution about `+0.000824 dB`.
- Do not treat v1.4 as permission to run locked Haze4K test; v1.4A failed
  internal `val_regular`/`val_hard` gates.
- Do not micro-tune full `DPFM1+2+4` scale/gate after v1.4A; ablation shows
  `DPFM2-only` is negative and full DPFM123 increases tail risk.
- Do not run locked Haze4K test for v1.4B before the written internal
  regular+hard gate passes.
- Do not continue v1.4C small adapter, BiDPFM1 scale/gate/loss search,
  DPFM1+4 training, or UDP-Lite DPFM2 revival without a materially new
  mechanism.
- Do not start FullUDP transplant, teacher distillation, or locked Haze4K test
  from the current official ConvIR+UDP checkpoint/protocol; Phase 0 failed
  regular/easy/SSIM/tail safety despite hard gains.
- Do not treat UDPNet-only as a global model after v1.6. The positive result is
  the A0-fallback expert switch, not a full UDPNet replacement.
- Do not change the v1.6 fixed switch threshold, feature, checkpoint, or expert
  bank after seeing locked-test results; the locked confirmation failed and the
  route is closed under this exact policy.
- Do not run locked Haze4K test from the current v1.7A risk-controlled
  shrink/mix policy; both OOF and train-heldout gates failed.
- Do not micro-tune v1.7A `tau_gain`, `tau_risk`, OOD cutoff, feature set,
  alpha set, or low-capacity heads from the completed v1.7 results and call it
  the same route.
- Do not reinterpret v1.8 as permission to touch locked Haze4K test. The queue
  is train-derived and diagnostic/candidate-screening only until its written
  gates pass and a separate locked-test card is written.
- Do not continue this exact v1.8 BiDPFM1 `fusion_neighbor` route by adding more
  seeds, more epochs, or checkpoint-choice tweaks under the same evidence
  contract; the full repaired 10-seed screen is already negative.

## Reopen Condition

A DPGA follow-up must preserve the v1.6 expert-switch reading: UDPNet is a hard
expert behind an A0 fallback, not a replacement checkpoint. Because the fixed
v1.6 threshold failed locked confirmation, any later route must introduce a new
predeclared calibration source or stronger deployable router before touching
locked test again. Because v1.7A already tested full-train low-capacity
gain/risk/OOD shrink-mix and failed OOF plus heldout gates, the next credible
reopen requires a materially stronger deployable router or calibration
objective, not threshold polishing. Any later transplant/distillation must be
conditional on the router or teacher, not UDPNet-only.
v1.8 satisfied the "materially new" requirement for its predeclared
partial-unfreeze capacity route, stronger table-only router audit, and the
newly added data/domain-adaptation evidence path. Because no real-haze target
domain data was visible on `dehaze1`, the domain path is split into an explicit
data blocker plus a Haze4K internal domain-conditioned policy diagnostic. Since
v1.8 failed after full queue completion and repaired closeout, do not continue
by simply adding more BiDPFM1 scale/gate tuning under the same route id.

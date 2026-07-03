# NoPost Feature Lowband Family Summary

Date: 2026-07-03

Status: WLDB-A as a concrete global final-feature lowband decoder is closed.
NoPost lowband remains open. v2.18 closes the immediately proposed WLDB-A2
global pooled-policy route because O1 action learnability is not tail-safe, but
it leaves positive evidence for a future spatial WLDB-B learnability route.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-07-03-haze4k-v2-16-nopost-wavelet-lowband-decoder.md`
  - `../experiment_cards/2026-07-03-haze4k-v2-17-nopost-lowband-alignment-tail-audit.md`
  - `../experiment_cards/2026-07-03-haze4k-v2-18-nopost-tailaware-lowband-policy.md`
- Evidence roots:
  - `../experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/`
  - `../experiment_logs/haze4k_v2_17_nopost_lowband_alignment_tail_audit_20260703/`
  - `../experiment_logs/haze4k_v2_18_nopost_tailaware_lowband_policy_20260703/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| Haze4K v2.16 NoPost Wavelet Lowband Decoder | T0 showed WD0375 severe-risk is decoupled from lowband need: WD0375 severe vs lowband-need Jaccard `0.027917`; WD0375 severe vs A0 hard-bottom25 Jaccard `0.000000`. T1 LL oracle was very strong: all-image mean `+14.998694`, hard `+18.939359`, easy `+11.853745`, severe `0`, lowband-need rate `1.000000`. T2 source/identity passed with forbidden symbol hits `0` and identity max abs diff `1.7881393432617188e-07`. WLDB-A then trained seed `3407` for `20` epochs with only `2128` trainable `nopost_wldb.*` params. Best checkpoint `model_5` had mean/hard/easy `+0.081889/+0.105887/+0.020994`, positive `0.662500`, but severe `67/480` and strong-reference regressions `48/120`. | `WLDB_A_SCREEN_FAIL_STOP_NO_MORE_TRAINING`; do not expand WLDB-A seeds, epochs, hidden width, locked test, or promotion from this form. |
| Haze4K v2.17 NoPost Lowband Alignment Tail Audit | R1 confirmed the v2.16 failure shape: `model_5` mean/hard/easy `+0.081889/+0.105887/+0.020994`, p05 `-0.438669`, severe `67/480`, action-budget activation `0`. R2 proved internal feature-lowband oracle headroom: O1 global final-feature LL mean/hard/easy `+0.842954/+1.591207/+0.359026`, p05 `+0.001803`, severe `0`; O2 spatial final mean `+6.160490`; O3 mid+final mean `+6.832469`; O4 RGB LL reference mean `+14.998694`. R3 showed the average objective improved final/lowband L1 but failed CVaR/tail safety: CVaR5 `-0.646619`, severe `67/480`, strong-reference regressions `48/120`, budget activation `0.0`. | `NO_TRAINING_PAUSE_DESIGN_TAIL_AWARE_WLDB_A2_OR_WLDB_B_OBJECTIVE`; close WLDB-A as trained, keep NoPost lowband open, require a materially changed tail-aware objective and gate before training. |
| Haze4K v2.18 NoPost Tail-Aware Lowband Policy | P4 passed after an engineering broadcast fix: forward `(self, x)`, forbidden symbol hits `0`, zero-init max_abs_vs_A0 `0.0` for global and spatial modes, new params global `3168` and spatial `19552`, official params `8630665`. P1 regenerated O1 global final-feature LL oracle actions and tested deployable pooled-LL MLP learnability. It beat shuffled control and moved hard samples, but was not tail-safe: mean/hard/easy `+0.263178/+0.859418/-0.183929`, p05 `-1.164642`, CVaR5 `-2.050251`, severe `568/2400`, strong-reference regressions `303/600`, positive `0.592083`, wrong-direction `0.193750`, control gap vs shuffled `+0.308958`. P2 passed: tail/preserve objective replay covered v2.16 `model_5` severe and strong/easy failures with positive activation rates `0.0`. P3 passed: `3` predeclared nonzero action-budget thresholds were calibrated. | `V218_PAUSE_P1_GLOBAL_POLICY_LEARNABILITY_FAIL`; do not train WLDB-A2 global pooled final-feature LL policy. Use P2/P3/P4 as positive evidence for a future spatial WLDB-B learnability route. |

## Family Verdict

The precise conclusion is: close WLDB-A and the v2.18 WLDB-A2 global pooled
policy form; do not close NoPost lowband.

v2.16 established that lowband correction is a real source of headroom inside
ConvIR-B. The RGB LL oracle and the proposed zero-init WLDB insertion were both
source-clean enough to justify a first trainable screen. That screen is now
closed because the concrete WLDB-A form only moved mean/hard metrics modestly
and failed tail safety. The severe failures are not a locked-test artifact:
they are train-derived fold evidence and no WLDB-A checkpoint passed the
predeclared gate.

v2.17 explains why the direction should not be closed. Internal feature-lowband
oracles are strong, especially spatial and mid+final insertion oracles. The
WLDB-A failure is better explained by objective and constraint mismatch than by
absence of lowband capacity. The current average objective can improve L1 while
still leaving p05/CVaR/severe and strong/easy preservation failures, and the
existing action-budget term did not activate.

v2.18 tested the recommended next filter before training. The contract and
identity checks passed, the tail/preserve replay catches the known WLDB-A
failure mode, and a nonzero action budget can be calibrated. The blocker is P1:
the deployable O1 global pooled final-feature LL policy learned average and
hard movement but damaged easy and tail cases too severely. This is enough to
pause WLDB-A2 global policy training. It is not evidence against spatial
WLDB-B, because v2.17 O2/O3 showed much larger spatial/internal oracle headroom
than the O1 global form.

## Do Not Repeat Without New Evidence

- Do not expand WLDB-A with more seeds, epochs, hidden width, checkpoint
  selection, or locked-test use.
- Do not train WLDB-A2 global pooled final-feature LL policy from the current
  P1 result.
- Do not treat mean or hard-bucket improvement as sufficient if p05, CVaR,
  severe count, strong-reference regressions, or easy preservation fail.
- Do not use locked Haze4K feedback to tune lowband actions, objectives,
  thresholds, checkpoints, or route choice.
- Do not reopen the older NoPost severe-risk selector line unless it introduces
  a materially new signal beyond the v2.13-v2.15 failures.

## Reopen Condition

A credible follow-up should be a new spatial WLDB-B learnability route, not a
WLDB-A rerun. It should use the v2.17 O2/O3 headroom as the positive capacity
source, keep the v2.18 source-clean and identity contract requirements, and
make p05/CVaR/severe, strong/easy preservation, and nonzero action-budget
activation primary gates before any training promotion or locked-test request.

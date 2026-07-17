# Haze4K v5 R3 Cloud Evidence Audit
Date: 2026-07-17
Decision: `R3_CLOUD_EVIDENCE_AUDIT_COMPLETE_S0_A0_DESIGN_ONLY`
## Outcome
The most defensible current bottleneck is a joint deployable proposal-to-value bottleneck: the system has privileged safe direction headroom and an adequate transport interface, but it has not yet shown that an inference-only candidate bank contains that headroom or that a candidate-conditioned critic can select it safely. The present direct-target heads conflate those two questions.
This sharpens, rather than rejects, the earlier representation-to-action diagnosis. It rules out action absence, exact-half transport loss, simple local/global readout width, action-ladder density, and fixed optimizer/window repair as primary explanations. It does not justify the stronger claim that candidate response is generally uninformative.
## Decisive Evidence
- A1F privileged direction-over-shrink LCB95 is `+0.105475 dB`; repairable-fraction LCB95 is `0.6953125`. Safe direction actions exist.
- A1C exact-half privileged gain LCB is `+0.134024 dB` and worst native-size retention LCB is `0.865395`. The exact-half interface is adequate.
- A1R context-spatial gain is `0.017954 dB` with LCB/UCB `0.015205/0.020654`; oracle retention is `0.105174` with LCB/UCB `0.092305/0.116634`. The retention UCB is far below the `0.25` gate, so this is not merely low precision.
- A1X-global gain is `0.015680 dB` with LCB/UCB `0.013200/0.018182`; retention is `0.091667` with LCB/UCB `0.080069/0.102841`. More same-distribution samples cannot plausibly rescue this frozen head contract.
- A1X still beats shuffle (`+0.007963 dB` LCB) and local readout (`+0.000890 dB` LCB), proving real signal but sub-material utility.
## Cloud Raw-Table Audit
The v3p candidate table contains `1,088,675` paired block rows per operator. D_ref/D_rep row keys match exactly.
- Across all blocks, `53.72%` have best-versus-second candidate MSE gap at or below `1e-10`, and about `69.6%` are at or below `1e-6`. Many blocks are ties or near-ties and should not receive equal hard-classification weight.
- Fixed `.125` is block-optimal for about `55.33%` of rows.
- Among active blocks, the best action is dominated by no-op and full action: D_ref counts are `192,933` for `0` and `224,510` for `1.0`; only about `17.2%` use the three intermediate actions.
- For active blocks with margin above `1e-5`, only actions `0` and `1.0` remain. This supports a first-stage signed/binary decision with abstention before amplitude refinement.
- Cross-operator best-action agreement is `0.942934`; first-step sign agreement is `0.968734`. The severe failure pattern is not operator noise.
- Using the other operator's block action causes only about `0.0051/0.0047 dB` median image regret, with `0.0325/0.0263 dB` p95. The two operators are robustness checks, not independent sample multiplication.
- Image-level block-oracle gain over fixed `.125` has median about `0.255 dB`, but the deployed v3m policy retained only about `23.2%` of oracle lift and created `148/146` severe regressions. High-value errors are sparse but repeatable.
## Cross-Route Boundary
- FAM2 v3i-C already tested a fixed-action counterfactual RGB response and failed: best OOF mean was `+0.008543 dB`, and its paired delta versus hard D7c had CI95 low `-0.009492 dB`. Repeating a single fixed-action `Y1-Y0` probe is low value.
- v3i-C did not test an explicit multi-action identity, within-image candidate comparison, or direct regret objective. A multi-action candidate-conditioned critic remains open.
- DTA-v3.7 output-difference features strict-passed train-derived D8 (`+0.078297 dB`) but failed one-shot locked D9 (`+0.020946 dB`; positive ratio `0.53175`). Candidate response can work in-domain, but distribution-shift calibration is a separate bottleneck.
## Data Sufficiency And Role Audit
- Haze4K train-derived data is locked as `2,400` train-inner plus `600` val-inner with no overlap.
- The v3p action-label chain uses `1,200` train-inner images. A1F `256`, A1R/A1X development `512`, and the historical A1X withheld `432` partition those `1,200` exactly.
- The other `1,200` train-inner images have not entered the v3p canonical action-label chain and are the highest-value source for a new R3 split.
- Bootstrap draws and paired operators improve uncertainty estimation but do not create new independent images. A1R, A1C, and A1X reuse the same 512 development images and are mechanism contrasts, not independent replications.
- Haze4K is sufficient for the next mechanism-identifiability route, but not for a general real non-uniform haze claim. External real-data validation remains necessary after a frozen Haze4K candidate exists.
## Data-Role Deviation
During this retrospective audit the full v3p 1,200-image candidate-loss table was scanned before the historical A1X 432-name remainder was classified as a withheld subset. Aggregate action outcomes therefore include that subset. No per-name result was retained or selected and no image/GT/prediction was decoded, but the 432 range is no longer independent confirmation evidence for future R3 work. It is reclassified as `historical_audit_only`.
## Authorized Next Action
Only a new R3 S0/A0 design is authorized. It must:
1. build a fresh data-role ledger from the 1,200 train-inner images outside v3p;
2. keep a new hash-stratified confirmation partition sealed;
3. test a GT-free multi-candidate bank oracle before training a value model;
4. compare state-only, state+action, and state+action+response under grouped outer OOF;
5. use pairwise/listwise regret, tie handling, asymmetric harm cost, and explicit abstention;
6. report risk-coverage curves and distribution-shift controls; and
7. stop before architecture training unless proposal and value gates both pass.
Implementation, model training, the historical 432 range, canary, and locked-test execution remain blocked.

The detailed design recommendation, gates, data ledger, factor cells,
architecture boundary, and time-critical stop list are in
`recommended_r3_route.md`. It is a design recommendation only and does not
itself authorize runtime.

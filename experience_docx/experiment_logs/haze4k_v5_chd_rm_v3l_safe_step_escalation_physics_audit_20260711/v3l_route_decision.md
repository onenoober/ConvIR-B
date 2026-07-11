# v3l Route Decision

Decision at start: `V3L_START_A0_ONLY_NO_CANARY_NO_LOCKED_TEST`.

The v3k closeout supports `context alpha=0.125` as a provisional safe step, but
it does not freeze a deterministic correction operator and does not provide a
new sealed split. v3l therefore starts with canonical operator artifact freeze
before any oracle granularity audit or physics-risk audit.

Forbidden at route start:

- no Haze4K locked test;
- no canary;
- no backbone or FAM/RARM training;
- no alpha/threshold selection from v3j route-confirm;
- no confidence/router training;
- no larger direct head or architecture promotion.

A0 pass may authorize only v3l-A1 oracle granularity audit.

## Closeout

Final decision:
`V3L_B_PRIVILEGED_TRANSMISSION_RISK_WEAK_STOP_NO_PHYSICS_POLICY`.

What passed:

- A0 froze two canonical context direct operators (`D_ref` seed 3407 and
  `D_rep` seed 3408) with exact double-replay equivalence and stable SHA.
- A1 showed large OOF oracle step-size headroom for both operators. Image,
  16x16 block, 32x32 block, and pixel-scalar oracles all beat fixed
  `alpha=0.125` with zero severe regressions.

What failed:

- B found only `trans/` metadata available; no airlight/beta/depth/atmos files
  were found.
- Privileged transmission features gave only weak-to-moderate direct-severe
  risk signal on OOF: best direct-severe AUC was about `0.635` for `D_ref` and
  `0.631` for `D_rep`, below the pre-registered `0.65` gate.

Conclusion:

The bottleneck is now localized to deployable step-size/risk observability. The
direct operator has real oracle upside, but Haze4K transmission metadata alone
is not strong enough to justify a physics-risk policy. No canary, locked test,
confidence/router training, deployable t/A estimator, or larger head is
authorized from this route.

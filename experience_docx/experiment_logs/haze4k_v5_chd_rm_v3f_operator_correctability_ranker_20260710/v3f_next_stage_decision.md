# v3f Next Stage Decision

Date: 2026-07-10

Decision:
`V3F_A_SCALAR_PROXY_SEPARABILITY_WEAK_NO_RANKER_TRAINING`

v3f-A completed the authorized no-training correctability target/separability
audit on internal val-inner 600. The best deployable scalar proxy for actual
current-FAM2 positive marginal gain was FAM2 correction magnitude with AUROC
`0.532034`, below the predeclared `0.56` ranker-screen gate. D7c score and D7c
hard gate were near random for this target.

The D7c-vetoed gain oracle has useful upper-bound value (`+0.078254` mean PSNR
delta with zero `<= -0.2 dB` regressions), but the deployable audited proxies do
not recover that oracle. Therefore no v3f-B lightweight ranker training is
authorized.

Blocked continuations:

- no v3f-B ranker training from the audited scalar features;
- no v3d continuation or 20-epoch extension;
- no v4/RARM expansion;
- no neighbor/FAM1/backbone unfreeze;
- no canary expansion;
- no Haze4K locked-test access.

Any future route must introduce genuinely new operator-context features,
operator target semantics, or a different correction operator before proposing
another correctability-ranker screen.

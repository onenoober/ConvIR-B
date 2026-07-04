# Haze4K v2.22 NoPost Gated Lowband N3 Microfit Evidence

Status: V222_N3_MICROFIT_PASS_REVIEW_ONLY_NO_LOCKED_TEST

This route trains only the new `nopost_gated_lowband_policy.*` modules.
It uses train-derived microfit stages and does not touch locked Haze4K test data.

## Closeout

- decision: `V222_N3_MICROFIT_PASS_REVIEW_ONLY_NO_LOCKED_TEST`
- locked test touched: `false`
- completed stages: `microfit16,microfit64,microfit256`

## Stage Summaries

### microfit16

- mean dPSNR: `0.027545475662347663`
- hard bottom25 dPSNR: `-0.0001698401072776079`
- p05 dPSNR: `-0.14138678868463117`
- severe rate: `0.0`
- mean mid/final unsafe prob: `0.1813696101307869` / `0.18147677835077047`
- mean mid/final delta RMS: `0.0005349676721380092` / `0.0003913115851901239`

### microfit64

- mean dPSNR: `-0.0023181556235576384`
- hard bottom25 dPSNR: `0.00172227399930458`
- p05 dPSNR: `-0.11836013134912556`
- severe rate: `0.0`
- mean mid/final unsafe prob: `0.17964996374212205` / `0.18049771012738347`
- mean mid/final delta RMS: `0.000887058294210874` / `0.000249117419571121`

### microfit256

- mean dPSNR: `-0.002880426983692619`
- hard bottom25 dPSNR: `0.01065834722624004`
- p05 dPSNR: `-0.21463632932851412`
- severe rate: `0.0546875`
- mean mid/final unsafe prob: `0.1817413717508316` / `0.1816737800836563`
- mean mid/final delta RMS: `0.0012126435758545995` / `0.0003303966950625181`

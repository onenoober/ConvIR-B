# v4.4 Decision After Diagnosis

Decision: `AUTHORIZE_V45_SDC_LITE_AND_V46_DCFSB_INDEPENDENT_FROM_OFFICIAL_ANCHOR`

Locked test touched: `false`

Test split enumerated: `false`

## Primary Internal256 Result

- A1 mean delta PSNR: `0.060228`
- A2 mean delta PSNR: `0.066864`
- A3 mean delta PSNR: `0.028960`
- expected additive mean delta PSNR: `0.127093`
- A3 interaction delta PSNR: `-0.098133`
- A3 positive ratio: `0.484375`

## Interpretation

The broader internal256 diagnostic prevents a first128-only false negative: A3 is slightly positive on this train-derived diagnostic split. However, A3 remains weaker than A1 and A2 individually and is far below the additive expectation. The negative interaction is the stable signal.

Therefore A3 is not promotion-ready and must not be extended with density auxiliary loss, DCFSB, longer training, seed sweep, canary expansion, or locked-test access.

Independent follow-up is authorized:

- v4.5 SDC-Lite from the official anchor, with no `GST_1_2` and no `SDFM_1_4` in the first version.
- v4.6 DCFSB-bottleneck from the official anchor, independent of A3.

Both must train on `haze4k_train_adapter_train.txt`, audit on `haze4k_train_internal_holdout256.txt`, and keep locked test blocked.

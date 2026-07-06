# Haze4K v2.38B Rich Target-Only NoOp/Unsafe Separability Audit

Date: 2026-07-06

Branch:
`codex/haze4k-v2-38b-rich-target-only-noop-unsafe-separability-audit`

Route identity: diagnostic rescue after v2.37 P4 target-only separability
failure and v2.38 micro-alpha P0 failure.

## Primary Question

Did v2.37 P4 fail because the 15 target-only proxy features were too weak, or
because unsafe/no-op cases are still not deployably separable even with richer
target-only ConvIR-B internal features?

## Not Allowed

- No bridge/generator training.
- No masked/oracle runtime contract.
- No teacher output, teacher delta, GT, sample id, crop coordinate leak, canary80,
  or locked test in deployable features.
- Positive controls may use illegal features only as controls and must be labeled
  as forbidden for deployment.

## Metric Contract

The deployable feature set may use only input, A0, input-A0 residual, and
ConvIR-B internal activations available at runtime. It is evaluated with OOF
folds inherited from v2.37.

Gate:

```text
unsafe_AUROC >= 0.90
unsafe_AUPRC >= 0.50
severe_recall_at_FPR_0.05 >= 0.80
strong_reference_unsafe_recall_at_FPR_0.05 >= 0.80
easy_noop_precision >= 0.90
fold_pass >= 4/5
shuffle_AUPRC near base_rate
no forbidden feature hits
```

This route is diagnostic. Passing it does not directly authorize bridge or
generator training; it would authorize a separately written follow-up route.

## Result

Decision: `P0_FAIL_RICH_TARGET_ONLY_SEPARABILITY_DIAGNOSTIC`.

The rich target-only audit completed as a diagnostic. No deployable feature
variant passed the predeclared unsafe/no-op separability gate; bridge/generator
training, canary80, and locked test remain blocked. The result supports closing
the current high-gain M0 selective bridge route unless a materially new runtime
no-op/unsafe signal is introduced.

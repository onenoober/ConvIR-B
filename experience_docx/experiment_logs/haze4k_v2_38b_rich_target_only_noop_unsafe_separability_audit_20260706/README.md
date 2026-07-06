# Haze4K v2.38B Rich Target-Only NoOp/Unsafe Separability Evidence

Status: `COMPLETED_DIAGNOSTIC_GATE_FAIL`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-38b-rich-target-only-noop-unsafe-separability-audit.md`

Central index path:
`experience_docx/EXPERIMENT_INDEX.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-38b-rich-target-only-noop-unsafe-separability-audit`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked.

## Evidence Files

Compact sync candidates:

- `status.txt`
- `run_v238b_rich_target_separability.sh`
- `v238b_p0_rich_target_oof_per_fold.csv`
- `v238b_p0_rich_target_separability_summary.json`
- `v238b_closeout.json`

Cloud-only runtime/raw evidence:

- `v238b_p0_rich_target_feature_manifest.csv`

## Metric Contract

Deployable features may use only input, A0, input-A0 residual, and ConvIR-B
internal activations available at runtime. GT, teacher output, teacher_delta,
sample-id leak, crop-coordinate leak, canary80, and locked test are forbidden in
deployable features. Illegal controls are labeled as controls only.

| variant | AUROC | AUPRC | severe recall @ FPR0.05 | strong-ref recall @ FPR0.05 | easy noop precision | fold pass | gate pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `proxy15_replay` | 0.8684 | 0.2249 | 0.2381 | 0.2692 | 0.2593 | 0/5 | False |
| `proxy_plus_residual` | 0.8872 | 0.2404 | 0.2381 | 0.2692 | 0.2903 | 0/5 | False |
| `proxy_plus_internal` | 0.8981 | 0.3076 | 0.3333 | 0.3077 | 0.3103 | 0/5 | False |
| `all_rich_target_only` | 0.8946 | 0.2961 | 0.4762 | 0.4231 | 0.3429 | 0/5 | False |

Controls:

- Label shuffle AUPRC `0.0453` with near-base-rate gate `True`.
- Teacher-delta leak positive-control AUPRC `1.0000`.
- A0-PSNR illegal upper-bound AUPRC `0.3881`.

## Result

Decision: `P0_FAIL_RICH_TARGET_ONLY_SEPARABILITY_DIAGNOSTIC`.

Richer target-only ConvIR internal features improved AUROC relative to the old
proxy-only view but still failed AUPRC, severe/strong-reference recall,
easy-noop precision, and fold gates. This diagnostic does not authorize a
selective M0 bridge/generator route.

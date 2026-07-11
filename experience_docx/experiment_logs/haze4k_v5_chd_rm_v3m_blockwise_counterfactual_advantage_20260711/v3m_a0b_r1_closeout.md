# v3m A0b-r1 Dense And Continuous Cross-Audit Closeout

Date: 2026-07-11

Decision:
`V3M_A0B_QUANTIZATION_GAP_SMALL_AUTHORIZE_A1_FEASIBLE_LOCAL_ACTUATION_ONLY`.

## Validity

A0b-r1 is a read-only audit of existing v3l 33-level-grid/continuous-pixel and
v3m five-level common-action OOF rows. All eight input SHA256 values pinned in
`v3m_a0b_metric_contract.md` matched. No model, checkpoint, dataset, or
inference entrypoint was opened; no raw per-image output was written. Locked
test, route-confirm selection, canary, and training remained false.

Both frozen operators had exact fixed `alpha=0.125` PSNR-delta replay over the
same 1,200 OOF names: maximum absolute difference `0 dB` for `D_ref` and
`D_rep`. Each raw policy mean also matched its compact source summary exactly.

## Quantization Result

`gap = 33-level/continuous source PSNR delta - five-level common PSNR delta`.
The threshold was a paired bootstrap mean-gap 95% upper bound of at most
`0.005 dB`, with 4,000 deterministic draws and seed `3407`.

| Operator | Pair | Mean gap (dB) | CI95 upper (dB) | p95 gap (dB) | Grid monotonicity | Both-policy tail safety | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `D_ref` | image dense vs common | `0.0011122` | `0.0013377` | `0.0065471` | pass | pass | pass |
| `D_ref` | block16 dense vs common | `0.0007468` | `0.0008024` | `0.0024219` | pass | pass | pass |
| `D_ref` | block32 dense vs common | `0.0008010` | `0.0008642` | `0.0028414` | pass | pass | pass |
| `D_ref` | pixel continuous vs common | `0.0006006` | `0.0006462` | `0.0020174` | n/a | pass | pass |
| `D_rep` | image dense vs common | `0.0011145` | `0.0013426` | `0.0059508` | pass | pass | pass |
| `D_rep` | block16 dense vs common | `0.0007425` | `0.0007999` | `0.0024200` | pass | pass | pass |
| `D_rep` | block32 dense vs common | `0.0007951` | `0.0008565` | `0.0027411` | pass | pass | pass |
| `D_rep` | pixel continuous vs common | `0.0005989` | `0.0006439` | `0.0019148` | n/a | pass | pass |

The continuous-pixel analytic alpha is a pre-clamp ceiling diagnostic, so it
does not receive a pointwise nested-action check. It does receive the same
paired mean-gap and p10/severe tail-preservation checks. The first A0b output
that treated it as pointwise nested is retained as `FAILED_METRIC_CONTRACT` and
is not a scientific stop.

## Interpretation And Boundary

The maximal upper bound, `0.0013426 dB`, is below one quarter of the prior
`0.02 dB` meaningful-escalation gate. Thus moving from five actions to the
33-level grid or continuous pixel ceiling does not materially change the oracle
headroom. The remaining bottleneck is feasible local actuation/observability of
the block16 oracle action, not action-ladder density.

Only A1 feasible-local-actuation audit is authorized. It must be non-training,
train-derived OOF first, and use a separate predeclared signal/metric contract.
No controller training, canary, locked test, physics/proxy route, or
route-confirm strategy selection is authorized.

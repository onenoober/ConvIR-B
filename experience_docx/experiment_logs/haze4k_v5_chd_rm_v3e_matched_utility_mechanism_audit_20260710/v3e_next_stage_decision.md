# v3e Next Stage Decision

Date: 2026-07-10

Decision:
`V3E_OPERATOR_CORRECTABILITY_MISMATCH_PRIMARY_HARD_GATE_SAFETY_TRADEOFF_SECONDARY_NO_RARM_EXPANSION`

## Result

v3e completed A-D mechanism audits without training and without locked-test
access.

The primary bottleneck is not D7c actionability. It is the mismatch between
D7c actionability and the current FAM2 adapter's operator-specific
correctability.

## Evidence

v3e-A:

- D7c-control paired mean CI95: `[-0.01676, -0.00365, +0.00930]`.
- D7c tail-regression reduction CI95 for `<= -0.2 dB`: `[26, 41, 57]`.
- D7c win-rate CI95: `[0.4717, 0.5117, 0.5517]`.

Interpretation: single-seed mean is inconclusive; tail safety is stable.

v3e-B:

| Weights | Gate | mean delta | `<= -0.2 dB` regressions |
| --- | --- | ---: | ---: |
| `W_D` | `G_D` | `+0.02947` | `50` |
| `W_D` | `G_1` | `+0.03899` | `113` |
| `W_U` | `G_D` | `+0.01278` | `23` |
| `W_U` | `G_1` | `+0.03307` | `91` |

Interpretation: D7c hard gate is a real safety valve, but it also drops mean
utility from ungated weights. Opening the gate on D7c-trained weights improves
mean slightly but makes tail risk unacceptable.

v3e-C:

- D7c score vs ungated FAM2 positive gain AUROC: `0.4921`.
- D7c score vs D7c-gated FAM2 positive gain AUROC: `0.4904`.
- Ungated positive-gain precision inside D7c gate: `0.4800`.
- Ungated positive-gain rate outside D7c gate: `0.4979`.

Interpretation: D7c score is near-random for current FAM2 positive operator
gain. D7c is not an operator-value router for this adapter.

v3e-D:

- D7c audited pre-clip total grad norm mean: `0.03056`; clip scale: `0.03605`.
- Control audited pre-clip total grad norm mean: `0.05550`; clip scale: `0.01923`.
- Audited batches clipped ratio: `1.0` for both.
- CLI `weight_decay=0.0001`, effective Adam weight decay: `0`.
- Resume checkpoints contain `model`, `optimizer`, `epoch`, but no scheduler.
- Content/FFT and action/preserve gradient cosines were positive in this small
  audit.

Interpretation: the training contract is imperfect and should be fixed in a
new contract version, but this audit does not show loss-gradient conflict as
the main blocker.

## Do Not Do Next

- Do not continue v3d to 20 epochs.
- Do not enter v4/RARM expansion.
- Do not unfreeze neighbor/FAM1/backbone.
- Do not train a new generic D7c probe.
- Do not touch locked Haze4K test.
- Do not silently fix weight decay or scheduler and compare directly to v3d.

## Authorized Next Route

Open a separate v3f design/audit route:

`D7c safety veto + FAM2 operator-correctability ranker`

The correctability target must come from internal/OOF actual FAM2 marginal gain
under matched split policy, not from locked test and not from generic global
LDHN need. The first v3f phase should be a target/separability/no-training
audit before any training can be authorized.

# v3m A0b Dense And Continuous Cross-Audit Metric Contract

Date: 2026-07-11

Status: `COMPLETED_GATE_PASS`.

## Route And Scope

A0b is a no-inference, no-training mechanism audit authorized solely by the
A0a common-action pass. It reads the existing frozen v3l A1 OOF rows and the
existing frozen v3m A0a OOF rows. It does not access route-confirm for any
calculation or selection, and it cannot access Haze4K locked test.

The question is whether the A0a five-level action ladder loses a material amount
of oracle value to the v3l 33-level ladder or to the continuous pixel ceiling.
This is a quantization-mechanism audit, not a deployable policy evaluation.

## Immutable Inputs

| Source | Artifact | SHA256 |
| --- | --- | --- |
| v3l | `v3l_a0_canonical_operator_artifact_manifest.json` | `1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84` |
| v3l | `v3l_a0_canonical_operator_closeout.json` | `2ca39ec1e17f4be794121603e3392a4e042e4d93b0e823454f7cf539f172d05d` |
| v3l | `v3l_a1_oracle_policy_oof_rows_cloud_only.csv` | `2a1b3a45cbeab6e646da4c45d17d7a8ad8c45f4ba477d06dbf5d3ab630e284cc` |
| v3l | `v3l_a1_oracle_policy_summary.csv` | `7b538152bdac38526d50b500148c961819a78c6c0f9219be4626b962ca795d78` |
| v3l | `v3l_a1_oracle_granularity_summary.json` | `fca8e73dcf86e58cd2b60cfb8fc74967167a6f91d70ca2b3b3cb4d9c959964db` |
| v3m | `cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv` | `b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1` |
| v3m | `v3m_a0_common_action_oracle_summary.csv` | `925da8410154d16a20ab54ba0e3996dd99224fd94cad6d655c092dc745b940ec` |
| v3m | `v3m_a0_source_manifest.json` | `8966996c9c93f6f2f3fbdda536b69ea6aa03e1bf5432f127de47ca8ea95dd8a5` |

The v3l dense grid is exactly `0, 0.03125, ..., 0.96875, 1.0`; the v3m common
grid is exactly `{0, 0.125, 0.25, 0.5, 1.0}`. Both sets include the fixed
reference `alpha=0.125`.

## Pairing And Metrics

For each `D_ref` and `D_rep`, use exactly the 1,200 train-derived
clean-reference OOF names. First require exact name identity and fixed
`alpha=0.125` PSNR-delta agreement between v3l and v3m, with maximum absolute
difference no larger than `1e-12 dB`.

Then calculate one paired per-image gap in PSNR delta:

`gap = dense_or_continuous_policy - common_five_level_policy`.

| Dense/ceiling source | Five-level source | Pair name |
| --- | --- | --- |
| v3l `ORACLE_IMAGE_GRID` | v3m `ORACLE_IMAGE_GRID` | image dense vs common |
| v3l `ORACLE_BLOCK16_GRID` | v3m `ORACLE_BLOCK16_GRID` | block16 dense vs common |
| v3l `ORACLE_BLOCK32_GRID` | v3m `ORACLE_BLOCK32_GRID` | block32 dense vs common |
| v3l `ORACLE_PIXEL_SCALAR_CONTINUOUS` | v3m `ORACLE_PIXEL_GRID` | pixel continuous vs common |

The audit also checks each source raw mean against its compact policy-summary
mean to `1e-12 dB`. Bootstrap is paired by image, 4,000 deterministic draws,
seed `3407`, and reports the two-sided 95% interval.

## Gate

For every one of the eight operator-policy pairs:

1. all paired names must match, and fixed `alpha=0.125` must replay exactly;
2. for the three finite-grid pairs, every per-image gap must be at least
   `-1e-6 dB`, the established replay numerical precision;
3. for the continuous-pixel pair, do not apply pointwise dominance: its alpha
   is analytically solved before the final clamp, so it is only a ceiling
   diagnostic under this metric;
4. for both policies in every pair, p10 must be no lower and severe count no
   higher than their respective fixed `alpha=0.125` reference; and
5. the paired bootstrap mean-gap 95% upper bound must be `<= 0.005 dB`.

`0.005 dB` is one quarter of v3l's `0.02 dB` meaningful-escalation threshold;
it is intentionally an adequacy test for quantization, not a new claim of
practical model utility.

If all eight pairs pass, record
`V3M_A0B_QUANTIZATION_GAP_SMALL_AUTHORIZE_A1_FEASIBLE_LOCAL_ACTUATION_ONLY`.
If any input, pairing, monotonicity, or upper-bound check fails, record
`V3M_A0B_QUANTIZATION_GAP_OR_PROVENANCE_FAIL_NO_A1`.

Either outcome keeps controller training, threshold selection, canary, physics
or proxy policy work, and locked-test access blocked. A pass authorizes only a
separate A1 feasible-local-actuation audit with its own contract.

## Correction Record

The first A0b read-only command used the stronger pointwise-dominance condition
for every pair. It found exact fixed-alpha replay and upper bounds below
`0.005 dB`, but the continuous pair had rare negative gaps after final output
clamping. The continuous alpha is solved from the unclamped residual in
`pixel_scalar_oracle`, whereas the five-level grid evaluates already clamped
candidates. They are therefore not pointwise nested under the reported metric.
The first command is retained as `FAILED_METRIC_CONTRACT`; it is not a
scientific gate result. A0b-r1 is the first valid decision-producing audit.

## A0b-r1 Result

All pinned input hashes matched. The two fixed-step OOF replays were exact
over 1,200 names per operator. Every one of the eight pairs passed the
corrected gate: the largest paired mean-gap 95% upper bound was `0.0013426 dB`,
all finite-grid checks met the `1e-6 dB` numerical tolerance, and the existing
p10/severe tail checks passed for both policies in every pair.

Decision:
`V3M_A0B_QUANTIZATION_GAP_SMALL_AUTHORIZE_A1_FEASIBLE_LOCAL_ACTUATION_ONLY`.

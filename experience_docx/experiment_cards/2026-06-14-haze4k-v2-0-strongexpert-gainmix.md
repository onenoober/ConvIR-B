# Haze4K v2.0 StrongExpert-GainMix Route

Date: 2026-06-14

Status: `C0_CAPACITY_OPEN_POSITIVE_COVERAGE_RISK_MAP_REQUIRED`

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Route name: Haze4K-v2.0 StrongExpert-GainMix.
- Short name: SEG-Mix.
- Branch: `codex/haze4k-v2-0-strongexpert-gainmix`.
- Official anchor commit: `2d529d4` from `github/codex/haze4k-official-arch-anchor`.
- Runtime host: `convir-4090`.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v20-strongexpert-gainmix`.
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Evidence root: `experience_docx/experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/`.
- Locked test policy: blocked. C0/C1/C2/C3 are train-derived or internal-validation only; no locked Haze4K test command is authorized.

## Objective

Build a stronger Haze4K model line by treating A0 as the preservation anchor and
strong dehazing systems as high-gain but high-risk experts. The first decision is
whether available experts provide enough oracle headroom to justify router and
distillation work.

## Strategy Shift

DTA-v3.7 is no longer the strong-model mainline. Its T/A/U features,
output-difference features, A0 preservation, and shrink-mix evidence remain
useful, but the current DTA/FDF action bank is a safe-small adapter family with
about `+0.14 dB` oracle headroom. SEG-Mix reopens the strong-expert space:

```text
Y = A0 + g(x, patch) * alpha(x, patch) * clamp(E_strong - A0)
    + optional small safe DTA correction
```

Where `E_strong` is FullUDP/UDPNet or another reproducible strong expert, `g`
is an abstaining router, and A0 remains the fallback for uncertain or already
good samples.

## Phase Plan

### C0 Strong Candidate Zoo Oracle

Purpose: decide whether available experts provide enough capacity before any
router training.

Inputs for C0a:

- DTA-v3.7 D8/D9 evidence from `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9`.
- Official FullUDP internal validation evidence from `haze4k_fulludp_v15_phase0_repro_20260605`.
- Existing A0 checkpoint and Haze4K internal validation split evidence.

Required outputs:

```text
v20_candidate_zoo_manifest.json
v20_candidate_zoo_per_image_metrics.csv
v20_candidate_zoo_single_expert_summary.csv
v20_candidate_zoo_alpha_grid.csv
v20_candidate_zoo_oracle_grid.csv
v20_candidate_zoo_oracle_composition.csv
v20_candidate_zoo_failure_bins.csv
v20_candidate_zoo_decision.md
```

C0 capacity gate:

```text
mean dPSNR >= +0.30
hard bottom25 >= +0.30
positive ratio >= 0.75
easy top25 >= -0.02
dSSIM >= 0
worst <= 5/600, or oracle worst is exactly 0
```

If mean/hard/worst pass but strict positive coverage is low, continue only to
C1 risk/correctability mapping before any deployable router claim.

## C0 Result

C0a completed on `convir-4090` at `2026-06-15T00:08:18+08:00` from commit
`885a9c0`. It used existing train-derived/internal validation evidence only;
locked test stayed untouched.

Decision:

```text
C0_CAPACITY_OPEN_POSITIVE_COVERAGE_RISK_MAP_REQUIRED
```

Key FullUDP/A0 endpoint-oracle metrics over the available `val_regular +
val_hard` 600-image internal-validation evidence:

| Candidate | Scope | mean dPSNR | hard bottom-25 | easy top-25 | dSSIM | positive | nonnegative | worst/600 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullUDP endpoint | all | `+0.062005` | `+0.685523` | `-0.686496` | `-0.00031039` | `0.5300` | `0.5300` | `252.0` |
| A0/FullUDP endpoint oracle | all | `+0.741695` | `+1.110910` | `+0.397112` | `+0.00022958` | `0.5300` | `1.0000` | `0.0` |

Interpretation:

- FullUDP remains unsafe as a global endpoint because it damages easy/regular
  samples and has `252/600` severe regressions when forced globally.
- The A0-preserving endpoint oracle is strong-model capacity: mean is above
  `+0.70 dB`, hard bottom-25 is above `+1.10 dB`, and worst tail is zero.
- Strict positive coverage is only `0.53`, so the next step is not immediate
  C2 router training; it is C1 risk/correctability mapping to decide whether
  the high-gain subset can be predicted without locked-test tuning.
- ConvIR-L, DehazeFormer, and PromptIR checkpoints/protocols were not available
  on `convir-4090` during C0a; they remain future candidate slots, not silently
  skipped evidence.

D8/D9 hygiene completed in parallel:

- D8 status confirms folds `0..4`, seeds `3407,3411,2026`, and `15/15`
  outputdiff groups completed, but D8 summary/aggregate metadata retained D7
  labels (`phase`, `outer_groups`, and `raw_d1_full_5x3_run`).
- D9 forensic recorded the locked failure profile without authorizing any
  DTA-v3.7 repair or threshold tuning.

Next action: launch C1 Strong Expert Risk/Correctability Map using the
FullUDP-A0 endpoint evidence and DTA output-difference/quality features as
train-derived risk signals.

### D8/D9 Evidence Hygiene In Parallel

Purpose: clean evidence interpretation, not model optimization.

Outputs:

```text
v37_d8_d9_reconciliation_audit.json
v37_d8_d9_reconciliation_audit.md
v37_d8_d9_reconciliation_inconsistencies.csv
v37_d9_forensic_bucket_summary.csv
v37_d9_forensic_top_regressions.csv
v37_d9_forensic_feature_drift.csv
v37_d9_forensic_summary.json
v37_d9_forensic_summary.md
```

Rules:

- Do not tune DTA-v3.7 from D9 locked feedback.
- Do not change thresholds, action membership, outputdiff features, or
  checkpoints from D9.
- Use D9 forensic only to understand failure modes for the new train-derived
  strong-expert route.

### C1 Strong Expert Risk/Correctability Map

Launch only after C0 shows enough capacity. Bucket gains and risks by A0 PSNR,
haze/depth/airlight proxies, sky/highlight/low-texture proxies, and
FullUDP-A0/DTA-A0 output-difference features.

### C2 A0-Preserving StrongExpert Router

Train a deployable abstaining router only if C1 shows high-gain and high-risk
regions are separable on train-derived/internal validation evidence.

Screen gate:

```text
mean >= +0.12
hard >= +0.20
easy >= -0.02
positive >= 0.65
dSSIM >= 0
worst <= 48/600
```

Strong formal target:

```text
mean >= +0.20
hard >= +0.30
easy >= 0
positive >= 0.70
dSSIM >= 0
worst <= 48/600
max outer worst <= 60/600
```

### C3 Train-Only Shifted Validation

Run leave-one-bin validation only after a train-derived router screen passes.
Bins include A0 quality, haze/transmission/depth, airlight/brightness,
sky/highlight/low-texture, and residual magnitude.

### C4 Distillation

Distill only from high-confidence A0-preserving expert mixes, never from global
FullUDP output. Easy and uncertain samples must preserve A0 or GT clean targets.

## Initial Stop Rules

- If C0 oracle mean is below `+0.20 dB`, stop router work and find or train
  stronger experts.
- If C0 oracle mean is `+0.30..+0.50 dB`, proceed to C1 and C2 only if risk
  bins are separable.
- If C0 oracle mean is `>= +0.70 dB`, prioritize router and selective
  distillation after C1.
- Any locked-test contact before internal gates pass invalidates promotion
  claims for this route.

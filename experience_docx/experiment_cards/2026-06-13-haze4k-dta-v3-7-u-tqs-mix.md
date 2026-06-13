# Haze4K DTA-v3.7 U-TQS-Mix Route

Date: 2026-06-13

Status: `PHASE_B_TABLE_POLICY_STRICT_FAIL_NEEDS_FEATURE_ENRICHMENT_OR_REAL_BLEND`

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Family: Depth-transmission adapters.
- Route name: DTA-v3.7 U-TQS-Mix.
- Expansion: Uncertainty-aware, Transmission/Airlight supervised, Quality-aware, Utility-constrained Soft Action Bank / Shrink-Mix.
- Branch: `codex/haze4k-dta-v3-7-u-tqs-mix`.
- Official anchor commit: `2d529d4` from `github/codex/haze4k-official-arch-anchor`.
- Continuation evidence base: DTA-v3.6 HRCS commit `4f74f08`.
- Runtime host selected for first execution: `convir-4090`.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix`.
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Evidence root: `experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/`.

## Final Route Judgment

The route decision is now strict:

```text
The main direction is not eliminating all negative samples, not continuing
DTA-v3.6 hard accept/reject threshold search, and not adding more router/FiLM
capacity first. The main direction is an A0-preserving ConvIR-B anchor plus a
utility-aware soft action mixture over the conservative DTA/FDF action family,
with explicit transmission, airlight, quality, and uncertainty signals to make
deployable gain-risk separability better.
```

This card supersedes v3.6 as the next DTA mainline. DTA-v3.6 remains evidence;
its hard reject policy is not promotion-ready and must not be the main strategy.

## Host Selection

On 2026-06-13, both candidate hosts were probed before launch.

- `convir-4090`: reachable; 8 RTX 4090 GPUs; no tmux sessions; GPUs 1-7 idle at about 16 MiB, GPU0 had a small Python process around 1908 MiB; required Python, Haze4K data, checkpoint, and v3.6 workspace exist.
- `convir-5090`: SSH failed with `Permission denied (publickey,password)` for the configured `caozhiyang@202.207.1.21` alias, so it is unavailable for this run.

Decision: use `convir-4090` now. If 5090 credentials are fixed later, rerun the
resource probe and move only independent queued jobs, never an active run.

## Why This Route Exists

DTA-v3.6 formal validation produced the decisive signal: the conservative
FDF action family has headroom, but hard reject deployable selection loses too
many positive samples.

Key v3.6 evidence:

- `45` train runs, `180` train-derived depth-control evals, `45` aggregates, and `27000` formal OOF rows completed.
- L3 deployable hard reject reached mean `+0.065404`, hard `+0.067793`, worst `47.07/600`, and max outer worst `59.33/600`, but coverage was only `0.8887` and positive ratio only `0.5817`, so strict gates failed.
- L3 high-coverage oracle at `0.95` coverage reached mean `+0.087922`, positive ratio `0.6362`, and worst `31.87/600`, strict-pass.
- Oracle choose `{A0,L2,L3,L1}` reached mean `+0.143298`, positive ratio `0.6623`, and zero worst regressions.

Positive budget implication for L3:

```text
allowed positive loss around (0.6362 - 0.6300) * 600 = 3.7 / 600
observed v3.6 hard-reject loss around (0.6362 - 0.5817) * 600 = 32.7 / 600
budget overrun about 8.8x
```

Therefore, the v3.7 policy must stop treating uncertain action as binary
accept/reject. It must shrink, mix, or route action strength continuously under
severe-tail constraints.

## Mechanism Sentence

```text
If we replace hard accept/reject with a utility-aware soft action bank over
A0-preserving conservative DTA/FDF candidates, and add deployable
transmission/airlight/quality/uncertainty features, mean and hard gains should
increase while positive-ratio and severe-tail gates hold because uncertain
positive samples receive bounded weak action instead of being discarded to A0.
```

## Model And Policy Change

### Soft action mixture

Primary policy form:

```text
output = A0 + sum_k pi_k(x) * alpha_k(x) * clamp(C_k - A0)

k in {L2_tail_safe, L3_balanced, L1_high_gain}
alpha_k in [0, 1]
sum_k pi_k <= 1
```

First action bank:

```text
A0
0.25 * L2, 0.50 * L2, 1.00 * L2
0.25 * L3, 0.50 * L3, 0.75 * L3, 1.00 * L3
0.25 * L1, 0.50 * L1, 0.75 * L1, 1.00 * L1
```

This is a strategy-space change. Small negative deltas are allowed when utility
and severe-tail constraints justify them; severe regressions remain the hard
constraint.

### Differential gain-risk head

Policy must predict per-action quantities, not a binary reject flag:

```text
E[dPSNR_k]
P(dPSNR_k > 0)
P(dPSNR_k > +0.02)
P(dPSNR_k <= -0.05)
P(dPSNR_k <= -0.20)
E[dSSIM_k]
P(dSSIM_k < -0.000005)
```

Deployable feature families:

```text
hazy image stats
A0 output/candidate difference stats when available without GT
Depth Anything depth stats
FDF gate/action stats
L1/L2/L3 disagreement stats
predicted transmission and airlight
transmission/airlight uncertainty
NR-IQA or quality/naturalness/brightness/contrast/color-cast features
edge, sky/high-brightness, low-texture region stats
```

### Transmission, airlight, and uncertainty branch

Integrated v3.7 model should add lightweight auxiliary heads on ConvIR-B/FDF
features:

```text
t_head: transmission map
A_head: atmospheric light
u_head: t/A uncertainty
```

Loss family:

```text
L_t = L1(t_hat, t_gt) + edge-aware smoothness
L_A = L1(A_hat, A_gt)
L_u = heteroscedastic NLL or calibration loss
L_consistency = depth-transmission consistency / disagreement regularization
```

Before training these heads, run data availability preflight. If Haze4K GT
transmission/A exists, use it. If not, build train-derived proxies only; locked
Haze4K test feedback must not tune proxies.

## Aggressive Execution Policy

The user explicitly asked not to waste time on conservative experiments. This
route therefore uses the smallest decisive intermediate checks, then parallelizes
training/evaluation when a gate is passed.

Mandatory but not conservative:

- Phase A table-only soft-oracle/gain-risk diagnosis is required because it is the fastest way to prove whether the new strategy space has enough headroom.
- Skip continued v3.6 hard-reject threshold sweeps.
- Skip broad router/FiLM/residual capacity increases unless Phase A/B show feature separability is still the blocker after T/A/Q/U features.
- Use available `convir-4090` GPUs in parallel for independent folds/seeds after launch preflight.

## Phase Plan

### Phase A: table-only, no-new-training diagnostics

Input: formal DTA-v3.6 OOF action table and selector error table.

Required outputs:

```text
v37_positive_loss_budget_report.csv
v37_soft_action_bank_oracle_grid.csv
v37_false_reject_false_accept_taxonomy.csv
v37_feature_ablation_auc_report.csv
v37_tA_quality_uncertainty_preflight.json
v37_phase_a_summary.json
```

Pass line:

```text
soft action oracle has at least one coverage >= 0.95 strict-pass row
positive ratio >= 0.630
worst <= 48/600
mean/hard/dSSIM/true-vs-controls all pass
```

If Phase A fails, stop v3.7 policy training and return to candidate action
family design. If Phase A passes, launch Phase B immediately.

### Phase B: TQS deployable gain-risk predictor

Train the smallest useful deployable predictor first because it is fastest and
more interpretable:

```text
ElasticNet/Ridge logistic
shallow GBDT/stumps
small MLP-lite only if OOF rows and feature stability justify it
```

Targets are differential utility and risk, not hard rejection. Feature groups
must include T/A/Q/U when available, and ablations must prove whether each family
adds deployable separability.

### Phase C: real soft-blend verification and integrated heads

Run real blended-output verification on train-derived folds to replace Phase A's
linear metric proxy. In parallel, prepare integrated A0-preserving candidate
training:

```text
ConvIR-B A0 anchor
+ conservative FDF L1/L2/L3 branches
+ t/A/u heads
+ differential quality/gain-risk head
+ utility-constrained soft mixture
```

### Phase D: formal validation

Formal remains train-derived and nested:

```text
5 folds x 3 seeds
nested policy selection
outer-fold-only reporting
fixed frozen policy before any locked test
```

Only a formal strict pass may authorize one locked-test confirmation. Locked-test
results must not tune thresholds, action membership, proxy labels, or checkpoints.

## Strict Gates

```text
coverage >= 0.95 for soft-oracle Phase A, >= 0.93 for deployable formal policy
mean_dPSNR >= +0.055
hard_bottom25 >= +0.040
dSSIM >= -0.000005
positive_ratio >= 0.630
true-vs-zero >= +0.040
true-vs-shuffle >= +0.035
true-vs-normal >= +0.030
worst <= 48/600
max_outer_worst <= 60/600-run equivalent
```

## Leakage Rules

Deployable policies may not use GT PSNR/SSIM deltas, locked-test outcomes, or
GT transmission/A at inference. GT transmission/A may supervise train-derived
auxiliary heads and may appear in diagnostic-only ablations.

## Decision Logic

```text
if Phase A soft oracle fails:
    stop v3.7 policy training; candidate family lacks enough soft-mix headroom
elif Phase A soft oracle passes and deployable T/A/Q/U features improve AUC:
    launch Phase B policy and real soft-blend verification in parallel
elif Phase A passes but deployable features remain weak:
    prioritize T/A/U auxiliary candidate retraining before any locked test
if formal 5x3 strict pass:
    seal one fixed policy for locked confirmation
else:
    no locked test and no post-hoc threshold search
```

## Phase A Result

Phase A completed on `convir-4090` from commit `71d1f88` with marker:

```text
DTA_V3_7_U_TQS_MIX_PHASE_A_OK rows=27000 soft_rows=18 strict_soft=13 gate=PASS_SOFT_ORACLE_HEADROOM
```

Key results:

- Soft action-bank oracle headroom is confirmed: `13/18` table-only soft-oracle rows strict-pass.
- Best row `A0_L2_L3_L1_full` / `max_dpsnr` has mean `+0.143298`, hard bottom-25 `+0.121101`, dSSIM `+0.00002551`, positive ratio `0.6623`, worst `0/600`, max outer worst `0/600`, and intervention rate `0.6623`.
- Shrink-enabled banks tie the best full bank because alpha `1.0` remains available; the forced no-A0 shrink bank still strict-passes in the linear proxy with mean `+0.130658`, positive ratio `0.6623`, worst `0.53/600`, and intervention rate `1.0`.
- The positive-loss budget report confirms v3.6 hard reject is structurally over budget: L3 selector positive loss is `32.73/600`, about `8.77x` the allowed L3 budget; L1 is `29.87/600`, about `6.79x` budget; L2 full action is below the strict positive-ratio line and is tail-safe only, not a standalone main action.
- T/A/Q/U preflight: transmission GT, predicted transmission, transmission uncertainty, airlight proxy, action stats, and input quality proxies are present; explicit airlight GT and NR-IQA/MANIQA/MUSIQ/CLIP-IQA features are missing and must be added or proxied train-derived before integrated training.
- Deployable severe-risk separability remains weak in current table features: best deployable severe AUC is only about `0.608` from input brightness/action stats, so Phase B must add T/A/Q/U features rather than reusing v3.6 hard reject.

Decision: `PHASE_A_PASS_SOFT_ORACLE_HEADROOM`. Proceed immediately to Phase B
TQS deployable gain-risk predictor and real soft-blend verification on
train-derived folds. The Phase A soft-alpha values are linear table proxies, not
final blended-image proof.


### Phase B table-only TQS implementation

Added script:

```text
experience_docx/tools/train_haze4k_dta_v37_tqs_policy.py
```

This trains nested ridge TQS gain-risk policies over the soft action bank using
only train-derived OOF rows. It reports deployable feature groups separately
from diagnostic `trans_gt` features. Required outputs:

```text
v37_tqs_policy_nested_report.csv
v37_tqs_policy_aggregate.csv
v37_tqs_policy_action_table.csv
v37_tqs_feature_group_ablation.csv
v37_tqs_summary.json
```

Phase B is allowed because Phase A passed soft-oracle headroom. It remains
train-derived and table-only; it does not touch locked Haze4K test.

## Phase B TQS Result

Phase B completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-phaseb` with
marker:

```text
DTA_V3_7_TQS_PHASE_B_OK rows=27000 groups=6 strict_pass=0 decision=PHASE_B_TABLE_POLICY_STRICT_FAIL_NEEDS_FEATURE_ENRICHMENT_OR_REAL_BLEND
```

Best aggregate deployable row:

| Feature group | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | max outer worst/600 | intervention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `T_pred` | `+0.015792` | `+0.013137` | `-0.00000566` | `0.6360` | `0.80` | `2.33` | `0.9993` |

Interpretation:

- The table-only predictor can control severe tail, but it collapses gain: mean,
  hard, dSSIM, and true-vs-controls fail strict gates.
- `T_pred` is the best current deployable group, which supports the T/A/Q/U
  direction, but existing table features are not enough to recover the Phase A
  oracle headroom.
- `diagnostic_with_trans_gt` also fails, so direct use of the existing
  transmission GT columns alone is not sufficient; real feature enrichment and
  blended-output verification are required.

Decision: `PHASE_B_TABLE_POLICY_STRICT_FAIL_NEEDS_FEATURE_ENRICHMENT_OR_REAL_BLEND`.
Proceed to feature enrichment / real soft-blend verification before formal
policy claims. Do not return to v3.6 hard-reject threshold search.

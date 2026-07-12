# Haze4K v5 CHD-RM v3p Canonical Signed-Gain Reconstruction

Date: 2026-07-12

Status: `A2_PASS_B0_PHYSICS_FORWARD_CONTRACT_PLANNED`

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712/`

## Route Identity

- Route type: new numerical-equivalence audit; it is not a v3o continuation.
- Current process-rule commit: `a56c3fbf49ce4930e4a34c5635ac27f90cae9ba9`.
- Historical evidence parent: `a885668b84f0851b2b29ef4f714d18e40db83c83`.
- Runnable source parent: v3o closeout commit
  `cf59d4f21dcc002d9698b1b42766a919183058f0`.
- Route branch: `codex/haze4k-v5-v3p-canonical-signed-gain-20260712`.
- Local edit workspace:
  `/home/ubuntu/workspace/ConvIR-B-v3p-canonical-signed-gain-20260712`.
- Cloud `REMOTE_REPO`:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3p-canonical-signed-gain-20260712`.
- Cloud `RUN_ROOT`:
  `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712`.
- Cloud `EVID_STAGE`:
  `$REMOTE_REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712`.
- Cloud Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

The v3o historical decision remains
`V3O_A0_CANDIDATE_SSE_REPLAY_INTEGRITY_FAIL_STOP`. v3p cannot change that
decision or reuse its float32 candidate losses as v3p evidence.

## Question And Scope

Can a new, preregistered canonical loss contract reconstruct the frozen
candidate losses such that block and full-image signed gains are numerically
equivalent and sign-stable?

Candidate generation remains the frozen v3o deployment semantics:

- the same `D_ref` and `D_rep` artifacts, 1,200 train-derived clean-reference
  grouped OOF names, five folds, block16 partition, action ladder
  `{0, .125, .25, .5, 1}`, candidate dtype, and clamp;
- only the loss path changes: after candidate generation,
  `(candidate.float64 - target.float64).square()` is the single error source.

This route performs no policy selection, threshold search, scorer training,
physics policy, controller replay, canary, route-confirm selection, or locked
test access. It does not change the backbone, FAM, correction head, action
space, or executor.

## Static Contract

The route hash-pins the v3o inputs: official A0 checkpoint, frozen control
checkpoint, v3j split and bounds, v3l operator closeout/manifest, density and
D7c artifacts, and v3m fixed-alpha OOF reference. The runner verifies these
same identities before every stage.

For every candidate and image, v3p computes full and block SSE from the same
pre-square float64 error tensor. The canonical image SSE is CPU longdouble
block SSE combined by `math.fsum`; CPU longdouble full SSE and GPU float64
full/block reductions are independent checks. Image MSE and PSNR derive only
from canonical block SSE.

The numerical preflight uses deterministic non-OOF stress tensors at both
observed image shapes and loss range. Before smoke, it freezes:

```text
tol(S) = atol_sse + rtol * abs(S)
atol_sse = max(1e-12, 8 * max synthetic absolute SSE error)
rtol     = max(1e-12, 8 * max synthetic relative SSE error)
```

Formal OOF results cannot alter this envelope. For G1,
`epsilon_G = tol(L_.125) + tol(L_.25)`; only values outside that gray zone
receive beneficial or harmful labels.

## Execution Profile

Selected profile: `audit/evaluation` with the smallest decisive sequence:

1. numerical preflight;
2. 32-image integrity smoke;
3. 1,200-image formal reconstruction only if smoke explicitly authorizes it.
4. read-only A1 reconstruction only if formal A0 explicitly authorizes it.
5. constrained G1 oracle only if the repaired A1r reconstruction explicitly
   authorizes it. This stage is a fixed, read-only ceiling measurement, not a
   deployed selector or policy replay.

The tracked runner is
`experience_docx/tools/run_v3p_a0_canonical_signed_gain.sh`. Raw block/image
tables, stdout, and `status.txt` remain under `RUN_ROOT`; `EVID_STAGE` receives
only compact JSON/CSV/README evidence after a terminal stage marker.

## Gates

| Stage | Analysis unit | Gate | PASS authorizes |
| --- | --- | --- | --- |
| numerical preflight | synthetic tensor / shape | longdouble available; every coverage map is exact; frozen scale-aware envelope is written before OOF | smoke only |
| A0 smoke | 32 OOF images per operator | row/fold/hash identity; coverage exactly one; fixed `.125` replay `<=1e-6 dB`; every normalized numerical check `<=1`; no non-gray G1 sign flip | formal only |
| A0 formal | 1,200 OOF images per operator | the same structural and numerical checks for every candidate, with worst-case candidate/image/block family enforcement | A1 reconstruction and G1 decomposition only |
| A1 reconstruction | 2,400 paired v3p/v3m image rows and all canonical blocks | reconstruct frozen A2 bin actions; require exact per-image selected-action counts, full pairing, and fixed replay `<=1e-6 dB`; decompose action-path and renderer SSE without a new replay | A2 constrained G1 oracle only |
| A1r engineering repair | same frozen rows and bins | A1's first reader used left-open/right-closed bins, while v3m A3 used `searchsorted(..., side="right")`; re-run only that source-semantic correction under a new run id and require exact action counts | A2 constrained G1 oracle only |
| A2 constrained G1 oracle | 1,200 OOF images per operator | begin from uniform `.125`; select only canonical non-gray beneficial `.125 -> .25` blocks in deterministic descending-G1 order (then block coordinates), with a fixed maximum 25% block cap and the hard non-overlap block executor; require per-operator LCB95 lift over `.125` `> .02 dB`, over uniform `.25` `> .01 dB`, selected-pixel-coverage LCB95 `> .01`, zero severe `<= -.2 dB` fixed-baseline regressions, and zero selected harmful SSE | B0 physics forward contract only |
| B0 scalar-A smoke | first 32 sorted train-derived OOF triplets | enforce 3,000 PNG numeric ids for each train modality, ignore only `train/haze/.DS_Store`, map haze numeric prefixes to `gt/<id>.png` and `trans/<id>.png`, decode source RGB and uint8 transmission without resize/crop, fit a spatially global scalar A per image, require A within `[-1/255, 1+1/255]` and maximum sRGB forward RMSE `<= 8/255`; report linear-space sensitivity only | B0 formal only |
| B0 scalar-A formal | all 1,200 train-derived OOF triplets | repeat the frozen smoke contract, write the block forward-residual MSE p99 noise floor for later abstention, and require no structural or numerical violation | B1 privileged `t+A` `.125/.25` ceiling only |

Structural mismatch, coverage failure, fixed replay mismatch, or a non-gray
signed-gain flip is `FAIL`. Envelope exceedance with intact structure and no
non-gray sign flip is `INCONCLUSIVE`, authorizing only same-stage repair. A
numerical `PASS` does not authorize policy replay, training, or promotion.

## Decision Boundaries

- `PASS`: `V3P_A0_CANONICAL_NUMERICAL_PASS_AUTHORIZE_A1_RECONSTRUCTION_ONLY`.
- `INCONCLUSIVE`: `V3P_A0_CANONICAL_NUMERICAL_INCONCLUSIVE_REPAIR_ONLY`.
- `FAIL`: `V3P_A0_CANONICAL_NUMERICAL_HARD_FAIL_STOP`.

## Completed Evidence

- A0 canonical reconstruction passed on both operators: all 2,400 OOF images
  and 2,177,350 raw block-table rows have exact coverage, fixed `.125` replay
  deltas below `1e-6 dB`, no non-gray G1 sign flip, and normalized numerical
  errors below the preregistered envelope. Its closeout authorizes only A1.
- The first A1 reader is retained as an engineering-invalid result: it used
  left-open/right-closed bins and failed 859/2,400 action-count
  reconstructions. A1r corrected only this source-semantic defect to match
  `searchsorted(..., side="right")`, then matched all 2,400 image action
  counts and showed that selection, rather than hard-block rendering, is the
  active bottleneck.
- A2 passed as a read-only constrained G1 ceiling. With the fixed 25% cap, the
  LCB95 lift versus uniform `.125` is `+0.045132 dB` (`D_ref`) and
  `+0.045011 dB` (`D_rep`); versus uniform `.25` it is `+0.021617 dB` and
  `+0.021320 dB`. Selected-pixel-coverage LCB95 is `17.539%` and `17.549%`.
  Both operators have zero severe fixed-baseline regressions and zero selected
  harmful SSE. The direct hard-mosaic and additive SSE forms agree within
  `2.274e-13`.

## Current Boundary

`V3P_A2_CONSTRAINED_G1_ORACLE_PASS_AUTHORIZE_B0_PHYSICS_FORWARD_CONTRACT_ONLY`
does not authorize selector fitting, threshold tuning, policy replay, canary,
or locked-test access. B0 must first validate the privileged scalar-A
physics-forward data contract on the fixed train-derived OOF triplets; only a
typed B0 result may name any later continuation.

B0 is a train-only privileged data-contract audit, not a target-free selector:
for every frozen OOF haze triplet it estimates one spatially global scalar
`A` from `I ~= tJ + (1-t)A` in the exact RGB/PIL `[0,1]` loader space. The
standard IEC sRGB-to-linear reconstruction is retained only as a sensitivity
diagnostic, never as an after-the-fact alternative B1 semantic. B0 failure
means the mapping, color space, resize, transmission serialization, or data
package is inconsistent; it stops the physics-estimator route rather than
authorizing `t_hat/A_hat` fitting.

The A2 oracle does not fit a selector, tune a threshold, replay a learned
policy, alter the action space, access a canary or locked test split, or
authorize any of those actions. It exists to decide whether the canonical
first-step signal has enough value beyond both uniform baselines to justify a
physics-forward contract. A failed A2 closes the adaptive-controller path at
the uniform frontier.

At any intermediate stage, compact evidence is committed to the route branch.
GitHub `main` is updated only at a terminal decision or explicitly recorded
major handoff.

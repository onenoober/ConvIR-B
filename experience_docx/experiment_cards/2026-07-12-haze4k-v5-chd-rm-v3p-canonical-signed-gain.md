# Haze4K v5 CHD-RM v3p Canonical Signed-Gain Reconstruction

Date: 2026-07-12

Status: `A0_PASS_A1R_ENGINEERING_REPAIR_PLANNED`

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

Structural mismatch, coverage failure, fixed replay mismatch, or a non-gray
signed-gain flip is `FAIL`. Envelope exceedance with intact structure and no
non-gray sign flip is `INCONCLUSIVE`, authorizing only same-stage repair. A
numerical `PASS` does not authorize policy replay, training, or promotion.

## Decision Boundaries

- `PASS`: `V3P_A0_CANONICAL_NUMERICAL_PASS_AUTHORIZE_A1_RECONSTRUCTION_ONLY`.
- `INCONCLUSIVE`: `V3P_A0_CANONICAL_NUMERICAL_INCONCLUSIVE_REPAIR_ONLY`.
- `FAIL`: `V3P_A0_CANONICAL_NUMERICAL_HARD_FAIL_STOP`.

At any intermediate stage, compact evidence is committed to the route branch.
GitHub `main` is updated only at a terminal decision or explicitly recorded
major handoff.

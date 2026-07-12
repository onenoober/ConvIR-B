# Haze4K v5 CHD-RM v3o Signed Adjacent-Advantage Identifiability

Date: 2026-07-12

Status: `COMPLETED_GATE_FAIL_A0_FORMAL_REPLAY_INTEGRITY`

Branch: `codex/haze4k-v5-v3o-signed-adjacent-advantage-identifiability`

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3o_signed_adjacent_advantage_identifiability_20260712/`

## Route Identity

v3o is a new diagnostic audit after the v3m and v3n fail-stops. It is not a
v3m/v3n threshold rescue and it does not authorize a controller, policy replay,
canary, route-confirm selection, or locked Haze4K test access.

Durable evidence parent: GitHub `main` commit
`4d097680160e4f6b433b5e6fb62d31df46c28415`.

Runnable source parent: GitHub v3m commit
`2c9cb511627895981c4c489cacd990326185ced6`. This is an audit/selector route,
not a model-structure route, so it inherits the verified frozen v3m operator
reader rather than changing the immutable architecture anchor.

## Objective

Determine whether the frozen direct correction operator has a safe, nontrivial
first adjacent action `0.125 -> 0.25` that can be identified from candidate
related information. The first task is to measure exact additive block SSE for
all fixed candidates; it must not infer utility from ordinal oracle labels.

## Forbidden Until A Written Gate Passes

- no v3m/v3n threshold retuning or `>=` substitution;
- no policy replay, controller/ranker/estimator training, or new correction head;
- no canary, route-confirm selection, or locked test;
- no backbone, InstanceNorm, FAM, direct-head, or correction-bound changes;
- no use of a candidate-action table to select a policy during A0/A1;
- no test-time adaptation or re-estimation of physics parameters per candidate.

## Stage Gates

1. A0 smoke: reproduce fixed `alpha=0.125` on 32 frozen OOF names and prove
   candidate block SSE sums reconstruct each full-image candidate MSE.
2. A0 formal: repeat exactly on 1,200 clean-reference grouped OOF images for
   both frozen operators. Pass authorizes A1 only.
3. A1: audit whether `direct_step_energy` can identify signed first-step gain
   and image-level harmful burden. It cannot run policy replay. A1 either
   permanently closes energy-only routes or authorizes only the written next
   information audit.
4. B0/B1 physics, deployable proxy training, and a value assessor are not
   authorized until their prior stage passes and the relevant new card/contract
   is written.

## Main Metrics

- block candidate SSE and signed adjacent gains;
- fixed-alpha replay difference in dB on exactly paired OOF rows;
- direct versus block-aggregated candidate MSE difference;
- beneficial/harmful SSE, harmful-to-beneficial ratio, and per-image cumulative
  harmful burden;
- clean-reference grouped summaries by operator and held-out fold.

## A0 Decision Rule

For both operators, A0 passes only when fixed `alpha=0.125` replay differs by
at most `1e-6 dB`, every candidate's block-aggregated image MSE differs from
the direct candidate MSE by at most `1e-10`, row identity and fold mapping are
complete, and no forbidden data was touched. A0 does not assert policy utility.

## A0 Smoke Closeout

Smoke passed on 32 OOF images per frozen operator with
`V3O_A0_SMOKE_REPLAY_INTEGRITY_PASS_AUTHORIZE_FORMAL_OOF_ONLY`. The fixed
`alpha=0.125` replay difference was `0.0 dB` for both operators. Maximum
candidate-MSE aggregation differences were `6.073581362234931e-11` for `D_ref`
and `7.561563202841681e-11` for `D_rep`, below `1e-10`. The source manifest
records no training, canary, or locked-test access. This authorizes formal A0
only; it does not authorize A1 or any policy, training, canary, route-confirm,
or locked-test action.

## A0 Formal Closeout

Formal A0 completed on 1,200 grouped OOF images per frozen operator and stopped
with `V3O_A0_CANDIDATE_SSE_REPLAY_INTEGRITY_FAIL_STOP`. Fixed `alpha=0.125`
replay remained exact at `0.0 dB`, but the maximum direct-versus-block-aggregated
candidate MSE differences were `2.839408504325125e-10` (`D_ref`) and
`3.528073042047275e-10` (`D_rep`), exceeding the preregistered `1e-10` gate.
The route may not loosen that threshold post hoc. A1, any policy replay,
training, canary, route-confirm, and locked-test action are unauthorized.

## Archive Rule

Candidate per-block and per-image raw tables remain cloud-only. The route branch
and GitHub main receive only compact text, JSON, CSV summaries, scripts, and
source manifests after each closed gate.

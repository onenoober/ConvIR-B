# NH-HAZE CDP-RM S1 Region-Target Identifiability

Date: 2026-07-20

Status: PLANNED

- Route id: nhhaze_cdprm_s1_region_target_identifiability_20260720
- Scientific contracts: experience_docx/scientific_contracts/
- Source branch: `codex/nhhaze-cdprm-s1-region-target-identifiability-20260720`
- Protected-data policy: NH-HAZE ids 01-50 are development-only and overlap checkpoint training; ids 51-55, confirmation, canary and locked test remain sealed.

## Scientific rationale

R16 directly contradicted the claim that domain-matched actions have no local
headroom, but its fixed 8x8/full-endpoint contract was nearly one-directional
and cannot support the intended demand-protection module. The closed R5-R13
selector/critic family also showed that a new readout cannot rescue an
unstable or action-generic target. Therefore the smallest useful next question
is upstream of model training: does one frozen content-adaptive region unit and
one frozen nonnegative bounded action define stable, material, protection-aware
regional labels?

S1 changes only the region unit relative to the matched fixed-8x8 control. The
action is exactly 25% of the frozen NH-domain A0-to-WDMamba endpoint residual;
there is no amplitude, checkpoint, region-count or threshold search. Both
partitions use 64 regions. Demand is action-relative regional PSNR gain, while
protection is separately defined by action-added gradient or chroma error. A
content-adaptive safe oracle is compared with no-op, safe-global, fixed-grid
and exact-area spatial-shuffle controls. GT is unavailable to region and action
construction and is opened only after two complete deterministic replays.

PASS proves only that this frozen region/action/target definition is worth
implementing as a zero-initialized CDP-RM architecture. It does not prove
learnability, deployment, external validity or final model quality. FAIL closes
this exact proxy definition, not all possible region or action definitions.
INCONCLUSIVE permits only the written evidence-completion action. The later
conditional sequence is S2 exact no-op architecture, S3 grouped OOF mechanism
validation, then one separately sealed real-domain validation; only the
operation named by the prior typed closeout may be materialized.

This note and the canonical contract are immutable after launch. Terminal
interpretation belongs only in the conclusion JSON; the typed closeout remains
the terminal authority.

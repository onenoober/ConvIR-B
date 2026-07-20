# R15 Real-Target Region-Action Identifiability Qualification

Date: 2026-07-20

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r15_real_target_region_action_identifiability_20260720`.
- Question: before any new model work, are the existing R10 three actions, the
  already-used NH-HAZE paired development population, and the frozen
  image-level precision contract jointly qualified for a target-aligned
  region-action identifiability measurement?
- Rules commit: `github/main@7080dc2c44006f5b62c6a3d302e025c2fb046778`.
- Source branch/commit: R14 compact audit
  `4fa19dc34bbc6255308698b0dbc055a62fcfaa53`; R3 action generator source
  `207581b4abfff2224bc21d4d1ae4ad5c26118936`; R10 action-feasibility source
  `c455577905efa8bb6f5c0daa84c3ec43c2ee6ff5`.
- Route branch: `codex/haze4k-v5-r15-real-target-identifiability-qualification-20260720`.
- Locked test/canary policy: confirmation, canary and locked test remain
  prohibited. NH-HAZE has already been used by v2.7 and is assigned only
  `development_screening`; it cannot be the final independent external set.

## Scientific Contract

- Population and analysis/grouping unit: all 55 paired NH-HAZE images are the
  intended development population. One paired image id is the grouping, split
  and bootstrap unit; regions, actions, operators and future raters are nested
  repeated measurements and never inflate the image count.
- Intervention or factor contrast and reference: S0A is a qualification audit,
  not the final target measurement. It verifies exact transport of R10 actions
  `reference_noop`, `state_positive_full` and `state_negative_full` from the R3
  production path. A single probe image is fixed before outcomes by the lowest
  SHA-256 token of `route_id:image_id`; both frozen D_ref/D_rep operators run.
  No GT is opened until the action/render manifest is sealed.
- Primary outcome, direction and aggregation: all structural identities, all 55
  pairs, exact R3/R10 action-source identities, finite native-resolution
  D_ref/D_rep renders, exact latent positive/negative sign symmetry, exact
  no-op identity and protected-access checks must pass. Precision is assessed
  at image level: report Wilson half-widths for concordance and the exact
  zero-event upper bound; historical v2.7 variability is a planning proxy only.
- Preferred mechanism and strongest competing explanation: the preferred
  explanation is that real-target measurement is possible and can discriminate
  H1 information loss from H3/H4 target/domain mismatch. The strongest
  competitor is that the Haze4K action semantics or 55-image precision do not
  transport sufficiently to make the problem identifiable.
- Evidence roles and candidate/freeze point: R10/R14 compact evidence and v2.7
  pair/variance rows are historical development evidence. NH-HAZE images and
  the one production-path probe are development qualification evidence. Probe
  id algorithm, assets, actions, operators, preprocessing, precision formulas,
  checks and terminal mapping freeze at the route commit.
- Primary gate, uncertainty and threshold source: PASS requires 55 unique
  complete 1600x1200 pairs, no duplicates/mismatches, exact source hashes, the
  R10 action tuple, native-size finite renders for both operators, exact latent
  `negative=-positive`, exact no-op/reference equality, nonzero action response,
  GT access strictly after the action manifest seal, zero protected access and
  image-level zero-event UCB95 `<=0.10`. Concordance Wilson half-widths and the
  v2.7 variance proxy are reported but cannot be used to relax the later frozen
  `0.10` and `0.020 dB` precision gates.
- `PASS` authorizes: only
  `R15_S0B_IDENTIFIABILITY_MEASUREMENT_CONTRACT_REVIEW_ONLY`. It does not
  authorize training, all-image action generation, annotation, confirmation or
  reopening R5-R13.
- `INCONCLUSIVE` authorizes: only `R15_S0A_EVIDENCE_COMPLETION_ONLY` when data
  role, license/provenance, source identity, precision qualification or runtime
  support is insufficient without a valid scientific action-transport failure.
- `FAIL` stops: a valid, complete audit showing that the frozen three-action
  production path cannot produce finite nondegenerate native-resolution paired
  actions closes this exact target-region-action transport and authorizes only
  S3 problem reformulation.

## Implementation Contract

- Exact change and disabled mechanisms: one no-training GPU production-path
  probe plus a complete paired-file/hash ledger and fixed precision audit. No
  selector, critic, fit, candidate search, threshold search, sample exclusion,
  visual scoring, semantic proxy or GT-conditioned action is allowed.
- Checkpoint/load/init/freeze contract: reuse R3 A0 official checkpoint
  `6f42037d...`, control checkpoint `08207119...`, frozen D_ref/D_rep artifacts,
  v4a final state and exact load code. No checkpoint is selected or modified;
  every parameter is eval/frozen; seed 3407 is used only for deterministic
  runtime behavior.
- Input whitelist and prohibited inputs: allow only the declared NH-HAZE
  development directory, v2.7 pair/preflight/per-image rows, R10/R14 compact
  evidence, exact source checkouts and the R3 frozen assets. Prohibit every
  confirmation/canary/locked role, filenames as model features, external labels,
  historical protected outcomes and undeclared checkpoints.
- Dataset/split/preprocessing/metric identities: flat `[image_id]_hazy.png` /
  `[image_id]_GT.png`, ids 01..55, RGB, 1600x1200, full native image, legacy
  pad-to-factor, R3 frozen base/operator/action path and no resizing. File
  SHA-256 values are retained cloud-only; Git receives their canonical manifest
  hash and counts.
- Matched baseline and budget: no-op/positive/negative share the same hazy
  input, base, operator, support and 0.25 render coefficient. D_ref/D_rep receive
  identical input and action definitions. One pre-outcome image only; this is a
  production compatibility qualification, never a utility estimate.
- Resource/cost limits or descriptive-only rationale: one 1600x1200 image, two operators, three renders;
  expected 600 seconds, timeout 1,800 seconds, one GPU, no resume.
- Runner and required assets: unchanged generic runner and the typed S0A asset
  manifest.
- Runtime spec and entrypoint: `route_runtime_specs/R15_S0A_MEASUREMENT_QUALIFICATION.json`
  and `tools/r15_s0a_measurement_qualification.py`.
- Representative engineering fixture: contract phase exercises exact action
  algebra, deterministic probe selection, pair parsing and precision formulas
  without assets. Run phase is the single full-resolution production-path
  qualification.

## Frozen S0B Contract Boundary

S0B is not materialized until the exact S0A PASS closeout exists. Its amendment
must keep all 55 images, both operators and the three actions; freeze four cells
F00/F10/F01/F11 with F11 primary; use three independent action-blind raters, a
SHA-fixed 20% repeat subset after at least seven days, `uncertain` treated as
protect/no-op, 4,000 image bootstrap draws with seed 3407, and the previously
written materiality/safety/coverage/precision gates. A rater roster, conflicts
declaration, blind randomization manifest, annotation rubric and legal data-role
statement must exist before S0B launch.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R15_S0A_MEASUREMENT_QUALIFICATION` | real-development qualification | data/action/provenance/native-runtime/precision eligibility | S0B contract review only |

- First operation: R15_S0A_MEASUREMENT_QUALIFICATION
- Expected wall time and monitor profile: 600 seconds, `standard`.
- Complete-unit resume policy: `none`.
- Cloud workspace/run/output/status/closeout: fresh MCP-owned workspace and run root; output `r15-s0a-measurement-qualification-r1`; status and heartbeat use the generic lifecycle; closeout
  `r15_s0a_measurement_qualification_closeout.json`.
- Compact Git evidence and cloud-only raw artifacts: Git retains qualification,
  dataset/action/provenance/precision/resource summaries and one scientific
  conclusion. Raw hashes, decoded images, tensors and probe renders remain
  cloud-only.
- Required engineering terminal tuple: `FAILED_ENGINEERING / null / NONE`.

The card is immutable after launch. R13 and R14 terminals remain unchanged.

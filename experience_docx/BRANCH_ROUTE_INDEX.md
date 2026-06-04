# ConvIR-B Branch Route Index

Date: 2026-06-04

Status: conservative GitHub branch reading and cleanup map.

## Purpose

Use this file when reading the GitHub branch list for `onenoober/ConvIR-B`.
It separates stable evidence, runnable experiment snapshots, covered historical
heads, and cleanup candidates.

This file is not a deletion authorization. Before deleting or force-updating any
remote branch, re-run the ancestor and evidence checks in the checklist below.

## Reading Rules

- `github/main` is the stable reader-facing entry point for synced text
  evidence, route cards, and governance docs.
- `github/codex/*` branches are runnable experiment snapshots. They may contain
  diagnostic code that is intentionally absent from `main`.
- A route branch can be an archive candidate only when its head is reachable
  from a retained leaf branch and the text evidence needed for reading the
  route is already present on `main` or in a retained branch.
- Failed or diagnostic code should not be merged into `main`; sync only route
  cards, text logs, JSON, CSV, command transcripts, and compact text packages.
- If a branch is the only runnable snapshot for an unresolved or reproducibility
  sensitive route, keep it even if its evidence is summarized elsewhere.

## Current Branch Families

| Family | Branches | Reading status | Cleanup posture |
| --- | --- | --- | --- |
| Stable entry | `main` | Main reader-facing branch. Current public `main` carries consolidated evidence through APDR v0.2RC; this cleanup branch prepares APDR text evidence through v0.4E for sync. | Keep. Sync later route evidence here with evidence-only commits. |
| FAM / hard-frequency / haze-prior | Historical FAM refs already pruned remotely; retained leaves include `codex/haze4k-hardfreq-loss` and `codex/haze4k-haze-prior-scm`. | Diagnostic and closed routes. Read from `EXPERIMENT_INDEX.md` plus compact packages where present. | Keep the two leaves until their unique code snapshots are no longer needed. |
| PFD / RHFD / SafeRHFD | `codex/haze4k-pfd-mainline`, `codex/haze4k-b1r-decoder-rhfd-preserve`, `codex/haze4k-saferhfd-v2-stage-scale`, `codex/haze4k-saferhfd-v2-train`. | Related but not one identical line: `pfd-mainline` is contained by SafeRHFD leaves, while B1r is a separate rescue snapshot. | Keep B1r and SafeRHFD leaves. Consider pruning only strict ancestors after verifying evidence on `main`. |
| APDR v0 to v0.2RC | `codex/haze4k-apdr-convir-v0`, `v0-1`, `v0-2`, `v0-2r-fullimage-router`, `v0-2rc-conservative-budget`. | Early APDR selector and budget line. Their cards and text logs are present in this cleanup branch and indexed by `EXPERIMENT_INDEX.md`. | Archive candidates after this branch is merged/pushed and ancestor checks are re-run. |
| APDR v0.2RC diagnostics to v0.4B | `codex/haze4k-apdr-v0-2rc-oracle-diagnostic`, `v0-3-shed-diagnostics`, `v0-4-cclf-diagnostics`, `v0-4a-low-field-only`, `v0-4b-derived-lowfield-basis`, `v0-4b-mapping-triage`. | Later APDR diagnostic chain. This cleanup branch adds APDR v0.3 through v0.4E cards, logs, and the v0.2RC diagnostic text package for main-branch reading. | Keep the APDR leaf until this evidence is public on `main`; prune intermediate APDR heads only after a fresh ancestor and readability audit. |
| DPGA / root-cause follow-up | `codex/haze4k-convir-v1-0-dpga-lite`, `codex/haze4k-rootcause-preexp`. | Separate follow-up branches that fork after the APDR chain and are not ancestors of `v0-4b-mapping-triage`. | Keep until route evidence is reviewed and either synced or closed. |

## Remote Branch Classification

| Remote branch | Role | Evidence state | Recommended action |
| --- | --- | --- | --- |
| `github/main` | Stable reader-facing branch. | Public remote is synced through APDR v0.2RC budget evidence; this cleanup branch prepares APDR v0.3-v0.4E text evidence for later main sync. | Keep as default entry point. |
| `github/codex/haze4k-hardfreq-loss` | Closed hard-frequency route leaf. | Text evidence and decision are indexed on `main`. | Keep for now as runnable closed-route snapshot. |
| `github/codex/haze4k-haze-prior-scm` | Closed haze-prior SCM route leaf. | Text evidence and compact AI package are indexed on `main`. | Keep for now as runnable closed-route snapshot. |
| `github/codex/haze4k-pfd-mainline` | PFD diagnostic ancestor. | Text evidence is indexed on `main`. | Archive candidate only after confirming SafeRHFD leaves cover all needed code and no exact PFD rerun is pending. |
| `github/codex/haze4k-b1r-decoder-rhfd-preserve` | B1r rescue route leaf. | Text evidence is indexed on `main`. | Keep; not contained by SafeRHFD-v2 train. |
| `github/codex/haze4k-saferhfd-v2-stage-scale` | SafeRHFD stage-scale leaf. | Evidence is branch-local unless separately synced. | Keep pending evidence review. |
| `github/codex/haze4k-saferhfd-v2-train` | SafeRHFD training leaf. | Evidence is branch-local unless separately synced. | Keep pending evidence review. |
| `github/codex/haze4k-apdr-convir-v0` | APDR early scout ancestor. | Card and text logs are present in this cleanup branch. | Archive candidate after merge/push, ancestor check, and readability audit. |
| `github/codex/haze4k-apdr-convir-v0-1` | APDR early scout ancestor. | Card and text logs are present in this cleanup branch. | Archive candidate after merge/push, ancestor check, and readability audit. |
| `github/codex/haze4k-apdr-convir-v0-2` | APDR selector-only ancestor. | Card and text logs are present in this cleanup branch. | Archive candidate after merge/push, ancestor check, and readability audit. |
| `github/codex/haze4k-apdr-convir-v0-2r-fullimage-router` | APDR full-image router ancestor. | Card and text logs are present in this cleanup branch. | Archive candidate after merge/push, ancestor check, and readability audit. |
| `github/codex/haze4k-apdr-convir-v0-2rc-conservative-budget` | APDR v0.2RC budget branch. | Card and text logs are present on `main`; this branch keeps the indexed budget evidence intact. | Archive candidate after ancestor check and public readability audit. |
| `github/codex/haze4k-apdr-v0-2rc-oracle-diagnostic` | APDR diagnostic ancestor. | Diagnostic text package and supporting logs are present in this cleanup branch. | Archive candidate only after merge/push, ancestor check, and readability audit. |
| `github/codex/haze4k-apdr-v0-3-shed-diagnostics` | APDR diagnostic ancestor. | Cards and text logs are present in this cleanup branch. | Archive candidate only after merge/push, ancestor check, and readability audit. |
| `github/codex/haze4k-apdr-v0-4-cclf-diagnostics` | APDR v0.4 diagnostic ancestor. | Cards and text logs are present in this cleanup branch. | Archive candidate only after merge/push, ancestor check, and readability audit. |
| `github/codex/haze4k-apdr-v0-4a-low-field-only` | APDR v0.4A diagnostic ancestor. | Cards and text logs are present in this cleanup branch. | Archive candidate only after merge/push, ancestor check, and readability audit. |
| `github/codex/haze4k-apdr-v0-4b-derived-lowfield-basis` | APDR v0.4B diagnostic ancestor. | Cards and text logs are present in this cleanup branch. | Archive candidate only after merge/push, ancestor check, and readability audit. |
| `github/codex/haze4k-apdr-v0-4b-mapping-triage` | APDR retained diagnostic leaf. | Text evidence through APDR v0.4E has been copied into this cleanup branch for main-branch reading. | Keep as APDR leaf until the sync is public and intermediate APDR heads pass pruning checks. |
| `github/codex/haze4k-convir-v1-0-dpga-lite` | DPGA follow-up leaf. | Branch-local unless separately synced. | Keep pending review. |
| `github/codex/haze4k-rootcause-preexp` | Root-cause follow-up leaf. | Branch-local unless separately synced. | Keep pending review. |

## Verified Ancestor Findings

These findings were checked against remote refs on 2026-06-04:

- APDR v0 through v0.4B derived-basis heads are ancestors of
  `github/codex/haze4k-apdr-v0-4b-mapping-triage`.
- APDR v0 through v0.4B derived-basis heads are also ancestors of the DPGA and
  root-cause follow-up branches.
- `github/codex/haze4k-apdr-v0-4b-mapping-triage` is not an ancestor of the
  DPGA or root-cause follow-up heads; those branches are separate leaves.
- `github/codex/haze4k-pfd-mainline` is an ancestor of both SafeRHFD-v2 leaves.
- `github/codex/haze4k-b1r-decoder-rhfd-preserve` is not an ancestor of
  `github/codex/haze4k-saferhfd-v2-train`; keep it as a separate rescue leaf.
- `github/codex/haze4k-hardfreq-loss` and
  `github/codex/haze4k-haze-prior-scm` do not contain each other; keep both if
  exact runnable closed-route snapshots still matter.

## Guarded Archive Candidate Batches

Do not prune any batch until this cleanup branch has been merged or pushed to
the public `main` branch and the readability audit passes.

| Batch | Candidate refs | Required retained ref | Current action |
| --- | --- | --- | --- |
| APDR covered ancestors | `codex/haze4k-apdr-convir-v0`, `codex/haze4k-apdr-convir-v0-1`, `codex/haze4k-apdr-convir-v0-2`, `codex/haze4k-apdr-convir-v0-2r-fullimage-router`, `codex/haze4k-apdr-convir-v0-2rc-conservative-budget`, `codex/haze4k-apdr-v0-2rc-oracle-diagnostic`, `codex/haze4k-apdr-v0-3-shed-diagnostics`, `codex/haze4k-apdr-v0-4-cclf-diagnostics`, `codex/haze4k-apdr-v0-4a-low-field-only`, `codex/haze4k-apdr-v0-4b-derived-lowfield-basis` | `codex/haze4k-apdr-v0-4b-mapping-triage` | Candidate only. Re-check ancestry and public evidence readability immediately before pruning. |
| Separate leaves | `codex/haze4k-convir-v1-0-dpga-lite`, `codex/haze4k-rootcause-preexp`, `codex/haze4k-saferhfd-v2-stage-scale`, `codex/haze4k-saferhfd-v2-train`, `codex/haze4k-b1r-decoder-rhfd-preserve`, `codex/haze4k-hardfreq-loss`, `codex/haze4k-haze-prior-scm` | None in this cleanup pass | Keep. These are not covered by the APDR retained leaf and need separate evidence review. |

## Prior Remote Cleanup Record

These remote refs had already been removed before the APDR v0.2RC evidence sync.
They were not unique heads at the time of cleanup, and their commits remained
reachable through retained later branches:

| Deleted remote ref | Reason recorded at cleanup time |
| --- | --- |
| `codex/haze4k-repro` | Contained by later Haze4K route branches. |
| `codex/haze4k-fam2-only` | Contained by later FAM2, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-bounded` | Contained by later confidence-gate, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-confidence-gate` | Contained by later selectivity, hardfreq, and haze-prior branches. |
| `codex/haze4k-stop20-noise-floor` | Contained by later selectivity, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-selectivity-or-kill` | Contained by both retained leaf branches. |

## Cleanup Checklist

Before pruning any remote branch:

1. Fetch current remote refs:

   ```bash
   git fetch github '+refs/heads/*:refs/remotes/github/*' --prune
   ```

2. Confirm the candidate is contained by the intended retained branch:

   ```bash
   git merge-base --is-ancestor github/codex/<candidate> github/codex/<retained-leaf>
   ```

3. Confirm route cards, log READMEs, JSON, CSV, logs, transcripts, scripts, and
   compact text packages are readable from `github/main` or another retained
   branch.
4. Confirm no unique experiment code snapshot is still needed for exact rerun,
   audit, or unresolved route analysis.
5. Delete only one small batch at a time, then re-run `git ls-remote --heads`
   and a public raw-file readability audit for the evidence index.

# ConvIR-B Branch Route Index

Date: 2026-06-27

Status: current GitHub branch reading and cleanup map after the 2026-06-27
evidence-first branch consolidation.

## Purpose

Use this file when reading the GitHub branch list for `onenoober/ConvIR-B`.
It separates durable evidence, immutable architecture anchors, retained
runnable scientific snapshots, and deleted redundant route heads.

The cleanup principle is deliberately conservative:

- `github/main` is the stable reader-facing branch for evidence, route cards,
  family summaries, protocols, and branch cleanup records.
- `github/codex/haze4k-official-arch-anchor` is the immutable architecture
  anchor and must not be deleted or repurposed.
- Failed or diagnostic route code should not be merged into `main`.
- A branch can be deleted only when its head is reachable from a retained branch
  and its text evidence is already readable from `github/main`.
- If a branch is a named scientific anchor, a unique diagnostic head, or the
  only obvious runnable snapshot for a still-useful route, keep it even if some
  commits are reachable elsewhere.

## Retained Remote Branches

| Branch | Role | Keep reason |
| --- | --- | --- |
| `main` | Stable evidence entry point. | Canonical reader-facing archive. |
| `codex/haze4k-official-arch-anchor` | Official ConvIR-B Haze4K architecture anchor. | Immutable starting point for new architecture routes. |
| `codex/haze4k-hardfreq-loss` | Hard-frequency diagnostic leaf. | Unique closed-route runnable snapshot. |
| `codex/haze4k-haze-prior-scm` | Haze-prior SCM diagnostic leaf. | Unique closed-route runnable snapshot plus compact package. |
| `codex/haze4k-b1r-decoder-rhfd-preserve` | B1r decoder RHFD rescue leaf. | Separate rescue snapshot, not covered by SafeRHFD-v2 train. |
| `codex/haze4k-saferhfd-v2-stage-scale` | SafeRHFD stage-scale leaf. | Retains PFD/RHFD follow-up lineage. |
| `codex/haze4k-saferhfd-v2-train` | SafeRHFD training leaf. | Retains PFD/RHFD follow-up lineage. |
| `codex/haze4k-apdr-v0-4b-mapping-triage` | APDR retained diagnostic leaf. | Covers v0.4B-v0.4E APDR code lineage. |
| `codex/haze4k-rootcause-preexp` | Root-cause diagnostic leaf. | Separate root-cause branch, not contained by APDR retained leaf. |
| `codex/haze4k-dta-v2-calibrated` | DTA-v2 diagnostic leaf. | Retains CalGate/lowgate lineage. |
| `codex/haze4k-dta-v3-7-u-tqs-mix` | DTA-v3 mainline leaf. | Retains DTA-v3.3-v3.7 lineage and D9 evidence. |
| `codex/haze4k-v1-7-risk-controlled-expert-mix` | DPGA/UDP expert-bank leaf. | Retains DPGA v1.0-v1.8 lineage and reusable expert table code. |
| `codex/haze4k-v2-2-c9-fixed-wdmamba-router` | WD0375 locked-pass anchor. | Named scientific anchor for the strongest Haze4K locked-pass baseline and v2.10/v2.11 locked-grid diagnostics. |
| `codex/haze4k-v2-3-c11-wd-fs-selector` | StrongExpert selector leaf. | Retains v2.0-v2.3 selector lineage. |
| `codex/haze4k-v2-4-c12-wd0375-distill` | C12 distillation diagnostic head. | Unique failed distillation snapshot. |
| `codex/haze4k-v2-5-c13-a0-frozen-residual-distill` | C13 residual distillation diagnostic head. | Unique failed/intermediate distillation snapshot. |
| `codex/haze4k-v2-7-nhhaze-transfer` | NH-HAZE zero-shot transfer leaf. | Contains v2.6 alpha-curve history and v2.7 transfer evidence. |
| `codex/haze4k-v2-8-nhhaze-official-weights` | NH-HAZE official-weight diagnostic head. | Retained as a unique diagnostic head, but use v2.9 correction context from `main` before citing NH-HAZE results. |
| `codex/haze4k-v2-12-ap-ria-in-anchor-adapter` | AP-RIA/anchor-adapter diagnostic head. | Unique current diagnostic head. |

## Deleted Remote Branches

These refs were deleted because each head is either identical to `main` or is
reachable from a retained branch. Their text evidence is in `github/main`, and
their exact heads remain named below by short commit where useful.

| Deleted branch | Retained reachability / reason |
| --- | --- |
| `codex/haze4k-repro` | Contained by later Haze4K route branches. |
| `codex/haze4k-fam2-only` | Contained by later FAM2, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-bounded` | Contained by later confidence-gate, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-confidence-gate` | Contained by later selectivity, hardfreq, and haze-prior branches. |
| `codex/haze4k-stop20-noise-floor` | Contained by later selectivity, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-selectivity-or-kill` | Contained by both retained frequency/prior leaves. |
| `codex/highvalue-evidence-sync-20260618` | Temporary branch identical to `main` at `5abc969`. |
| `codex/haze4k-dta-v3-dapc-finetune` | DTA-v3.3 intermediate head, retained through `codex/haze4k-dta-v3-7-u-tqs-mix`. |
| `codex/haze4k-dta-v3-4-fdf-tsr-finetune` | DTA-v3.4 intermediate head, retained through `codex/haze4k-dta-v3-7-u-tqs-mix`. |
| `codex/haze4k-dta-v3-5-fdf-rcs-lite` | DTA-v3.5 intermediate head, retained through `codex/haze4k-dta-v3-7-u-tqs-mix`. |
| `codex/haze4k-dta-lowgate` | DTA lowgate head `04c356c`, retained through `codex/haze4k-dta-v2-calibrated`. |
| `codex/haze4k-dta-v3-6-hrcs` | DTA-v3.6 head `4f74f08`, retained through `codex/haze4k-dta-v3-7-u-tqs-mix`. |
| `codex/haze4k-pfd-mainline` | PFD diagnostic head `8928eaf`, retained through both SafeRHFD-v2 leaves. |
| `codex/haze4k-convir-v1-0-dpga-lite` | DPGA diagnostic head `e2c8526`, retained through `codex/haze4k-v1-7-risk-controlled-expert-mix`. |
| `codex/haze4k-convir-v1-1-dpga-tail-control` | DPGA diagnostic head `a9def38`, retained through `codex/haze4k-v1-7-risk-controlled-expert-mix`. |
| `codex/haze4k-convir-v1-3-hard-selective-depth-fusion` | DPGA diagnostic head `238e694`, retained through `codex/haze4k-v1-7-risk-controlled-expert-mix`. |
| `codex/haze4k-convir-v1-4-udp-lite-depth-fusion` | DPGA diagnostic head `8e4162d`, retained through `codex/haze4k-v1-7-risk-controlled-expert-mix`. |
| `codex/haze4k-convir-v1-4b-bidirectional-dpfm1` | DPGA diagnostic head `5b335a2`, retained through `codex/haze4k-v1-7-risk-controlled-expert-mix`. |
| `codex/haze4k-convir-v1-5-full-udpnet-transplant` | UDP/DPGA diagnostic head `15aa04a`, retained through `codex/haze4k-v1-7-risk-controlled-expert-mix`. |
| `codex/haze4k-v1-6-risk-calibrated-expert-switch` | Expert-switch head `e7b68fe`, retained through `codex/haze4k-v1-7-risk-controlled-expert-mix`. |
| `codex/haze4k-v2-0-strongexpert-gainmix` | StrongExpert intermediate head `e03d034`, retained through `codex/haze4k-v2-3-c11-wd-fs-selector`. |
| `codex/haze4k-v2-1-segmix-multialpha-local` | StrongExpert intermediate head `7489d0b`, retained through `codex/haze4k-v2-3-c11-wd-fs-selector`. |
| `codex/haze4k-v2-2-c8-mini-expert-oracle` | Complementarity head `a825963`, retained through `codex/haze4k-v2-2-c9-fixed-wdmamba-router` and `codex/haze4k-v2-3-c11-wd-fs-selector`. |
| `codex/haze4k-v2-6-residual-shrinkage-alpha-curves` | Alpha-curve head `ca6bf92`, retained through `codex/haze4k-v2-7-nhhaze-transfer`. |

## Cleanup Checklist

Before pruning any future branch:

1. Fetch current remote refs:

   ```bash
   git fetch github '+refs/heads/*:refs/remotes/github/*' --prune
   ```

2. Confirm the candidate is contained by the intended retained branch:

   ```bash
   git merge-base --is-ancestor github/codex/<candidate> github/codex/<retained-leaf>
   ```

3. Confirm route cards, evidence READMEs, JSON, CSV, logs, transcripts, scripts,
   and compact text packages are readable from `github/main` or another
   retained branch.
4. Confirm the branch is not a named scientific anchor, current active route, or
   only clear runnable snapshot for a still-useful diagnostic.
5. Delete only a small batch at a time, then verify with `git ls-remote --heads`
   and refresh this file plus `EXPERIMENT_INDEX.md`.

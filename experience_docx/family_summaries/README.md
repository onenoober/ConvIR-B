# Haze4K Route Family Summaries

Date: 2026-06-10

Status: reading aids derived from `experience_docx/EXPERIMENT_INDEX.md`, route
cards, and text evidence roots. These files do not replace the per-route cards
or logs; they summarize family-level conclusions and reopening conditions.

## Authority Order

1. Read `../EXPERIMENT_INDEX.md` first for the current route table and evidence
   strength policy.
2. Use the family summary to understand what a route family has already ruled
   in or out.
3. Open the cited route cards and `../experiment_logs/<route_id>/` evidence
   before making a metric claim or launching follow-up work.
4. Prefer the cited text artifacts over chat history.

## Families

Non-family anchor:

- Official architecture anchor: `../OFFICIAL_ARCH_ANCHOR_POLICY.md` and
  `../experiment_cards/2026-06-10-haze4k-official-arch-anchor.md`. This is not
  a performance route and does not change family verdicts; future architecture
  experiments must branch from it rather than modify it directly. The migration
  environment/code-consistency audit is recorded in
  `../CLOUD_PY310_ENVIRONMENT.md` and
  `../experiment_logs/cloud_py310_environment_20260610/`.

| Family | Summary | Current state |
| --- | --- | --- |
| FAM/FAM2 feature modulation | [summary](fam_family_summary.md) | Closed for unchanged deployable FAM routing. |
| Hard-frequency and haze-prior losses | [summary](frequency_prior_family_summary.md) | Closed for the tested weighting/SCM forms. |
| PFD/RHFD preservation | [summary](pfd_rhfd_family_summary.md) | Diagnostic only; preservation improved but hard-gain gates failed. |
| APDR output residual/action-bank | [summary](apdr_family_summary.md) | Broad output-residual/coefficient-mapping forms stopped; safe-subset ideas require fixed-code OOF/held-out evidence. |
| DPGA in-network prior adapters | [summary](dpga_family_summary.md) | Active diagnostic family, not promotion-ready. |
| Depth-transmission adapters | [summary](dta_family_summary.md) | Positive diagnostics, but current DTA-v3 Phase A R0 failed; no locked test. |

## Evidence Strength Reminder

For current Haze4K stop20 work, `../EXPERIMENT_INDEX.md` records seed mean PSNR
std `0.2206 dB` and hard-bucket std `0.4551 dB`. Single-seed deltas below
`+0.10 dB` are directional or mechanism evidence by default, not promotion
evidence.

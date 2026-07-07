# Haze4K v4.1 SDFM-Only Evidence

Date: 2026-07-07

Status: PLANNED

## Read First

- Route card: `../../experiment_cards/2026-07-07-haze4k-v4-1-sdfm-only.md`
- Shared protocol package: `../../../docs/ai_text_packages/haze4k_v4_sfad/`
- A0 baseline lock evidence: `../haze4k_v4_0_baseline_lock_20260707/`

## Purpose

Test only SDFM at ConvIR-B 1/2 and 1/4 multi-scale fusion points. GST, DCFSB, density auxiliary loss, and color/airlight correction are intentionally absent.

## Planned Primary Files

| File | Use |
| --- | --- |
| `run_v4_a1_sdfm_preflight.sh` | Durable cloud preflight script. |
| `v4_a1_sdfm_preflight.log` | Cloud stdout/stderr. |
| `v4_a1_sdfm_preflight.json` | Structured partial-load, no-op, shape, and SDFM stats result. |
| `status.txt` | Start/end markers. |

## Current Decision

Pending Stage 0 preflight.

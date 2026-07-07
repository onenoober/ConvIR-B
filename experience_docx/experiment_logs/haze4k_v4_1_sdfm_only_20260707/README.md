# Haze4K v4.1 SDFM-Only Evidence

Date: 2026-07-07

Status: COMPLETED_PREFLIGHT_PASS

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

## Stage 0 Preflight Result

Status: `COMPLETED_PREFLIGHT_PASS`.

| Check | Result |
| --- | --- |
| Code commit | `73643c4a965e6399ebfb5362c8fff668c4d8e518` |
| Total params | `8,831,629` |
| Added params | `200,964` |
| Adapter-only trainable params | `200,964` |
| Frozen official params | `8,630,665` |
| Trainable prefixes | `SFAD_SDFM1`, `SFAD_SDFM2` |
| Partial load | `602` official keys loaded; `22` new `SFAD_` keys missing |
| Unexpected / shape mismatch | `[]` / `[]` |
| No-op max abs vs A0 | synthetic `0.0`, train crop `0.0` |
| One train-crop multiscale L1 | `0.01309124380350113` |
| SDFM 1/2 `R_s` | mean `0.5245807`, std `0.0071737`, min/max `0.4805435/0.5622713`, alpha `0.0` |
| SDFM 1/4 `R_s` | mean `0.5315088`, std `0.0017463`, min/max `0.5227934/0.5384336`, alpha `0.0` |
| Locked test touched | `false` |
| Test split enumerated | `false` |

Decision: `A1_PREFLIGHT_PASS`. Adapter-only 5-epoch Stage 1 is authorized; locked test remains blocked.

## Invalid Launch Note

The first `adapter5` launch entered the repository default validation path, which reads Haze4K `test`. It was stopped immediately and is invalid for scientific comparison or checkpoint selection.

Corrected action: use `run_v4_a1_sdfm_adapter5_notest.sh`, with `--valid_freq 999`, a fresh output model name, and separate train-derived/internal post-training audit.

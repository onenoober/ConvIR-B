# CHD-RM Haze4K Route

Date: 2026-07-08

Status: v0 route locked.

## Route Name

CHD-RM: Continuous Haze Density-aware Residual Modulation

Chinese name:

```text
连续雾浓度感知的区域自适应残差调制与低雾区域保护去雾方法
```

## Fixed Research Scope

This route is research content one. The fixed research content is:

```text
连续雾浓度感知的区域自适应残差调制与低雾区域保护去雾方法研究
```

The core problem is:

```text
不同雾浓度区域恢复强度不匹配
```

The route targets these cases:

- heavy haze regions are under-restored;
- medium haze regions are restored unevenly;
- low haze regions are over-restored;
- one global restoration strength is not appropriate for non-uniform haze.

Current scope:

- single-image dehazing;
- ConvIR-B backbone;
- Haze4K dataset;
- region-adaptive restoration;
- low-haze restoration-strength suppression;
- multi-scale haze-density modulation.

Out of scope for this route:

- independent color-fidelity modeling;
- independent luminance-fidelity modeling;
- independent texture-fidelity modeling;
- independent structure-fidelity modeling;
- color-correction modules;
- texture-enhancement modules;
- structure-preservation modules;
- Lab, luminance, gradient, or texture as core training targets.

These exclusions are route invariants. Later architecture changes may change how
CHD-RM is implemented, but must not change the route into a separate color,
texture, structure, diffusion, video, multi-image, or backbone-replacement
research direction.

## Main Expression

```text
O = O_A0 + gamma(H_density, R_need) * R_adapt
```

Definitions:

| Symbol | Meaning |
| --- | --- |
| `O_A0` | ConvIR-B baseline output |
| `H_density` | continuous haze-density response |
| `R_need` | regional restoration-need response |
| `gamma` | regional restoration-strength modulation coefficient |
| `R_adapt` | adaptive residual restoration branch |
| `O` | final dehazed output |

Region behavior:

| Region condition | Expected behavior |
| --- | --- |
| high `H_density`, high `R_need` | strong restoration |
| medium `H_density`, medium `R_need` | normal restoration |
| low `H_density`, low `R_need` | weak restoration or protection |
| low `H_density`, high `R_need` | cautious restoration |
| high `H_density`, low `R_need` | suppress over-enhancement |

## Source And Branch Policy

This is a new model-structure route.

Starting source:

```text
github/codex/haze4k-official-arch-anchor
```

Starting commit:

```text
3b4da35440c8c26a7d1bcaf1daf342e11d9a3898
```

Stage branch naming follows the user-requested v5 pattern:

```text
codex/haze4k-v5-<stage-id>-<route-name>
```

Registered stage branches:

| Stage | Branch |
| --- | --- |
| v0 route lock | `codex/haze4k-v5-v0-chd-rm-route-lock` |
| v1 data baseline lock | `codex/haze4k-v5-v1-chd-rm-data-baseline-lock` |
| v2 density-need calibration | `codex/haze4k-v5-v2-chd-rm-density-need-calibration` |
| v3 no-op RARM audit | `codex/haze4k-v5-v3-chd-rm-noop-rarm-audit` |
| v4 single-scale RARM | `codex/haze4k-v5-v4-chd-rm-single-scale-rarm` |
| v5 low-haze protection | `codex/haze4k-v5-v5-chd-rm-low-haze-protection` |
| v6 multiscale haze modulation | `codex/haze4k-v5-v6-chd-rm-multiscale-haze-modulation` |
| v7 OOF candidate lock | `codex/haze4k-v5-v7-chd-rm-oof-candidate-lock` |
| v8 final Haze4K confirmation | `codex/haze4k-v5-v8-chd-rm-final-haze4k-confirmation` |

## Data And Validation Protocol

Dataset contract:

| Field | Value |
| --- | --- |
| Dataset | Haze4K |
| Train | 3000 paired images |
| Test | 1000 paired images |
| Backbone | ConvIR-B |
| Task | single-image dehazing |

Validation layers:

| Layer | Data source | Purpose | Tuning |
| --- | --- | --- | --- |
| Canary | train subset | code, loss, and forward-stability checks | allowed |
| Internal val | train-derived 2400/600 split | variant, loss, and threshold selection | allowed |
| 5-fold OOF | train 3000 only | candidate confirmation and ablation reliability | allowed with pre-registration |
| Locked test | Haze4K test 1000 | final confirmation only | forbidden |

Locked test must not be used to choose checkpoints, thresholds, losses,
architecture variants, gamma caps, low-haze masks, or hyperparameters.

## Stage Gates

| Stage | Gate |
| --- | --- |
| v1 | data counts, leakage audit, A0 baseline traceability, metric stability |
| v2 | calibrated `H_density` and `R_need`; shuffled target control fails |
| v3 | no-op RARM equals A0 within tolerance; cost budget passes |
| v4 | CHD-RM-S1 beats random/shuffled modulation and matched-budget controls |
| v5 | low-haze protection passes without losing heavy-haze benefit |
| v6 | multiscale variant improves over v5 without unacceptable cost |
| v7 | one final candidate passes OOF gates without locked-test access |
| v8 | one-shot locked-test confirmation only after v7 candidate lock |

If a gate fails, the route must pause or return to the matching train-derived
stage. It must not use the locked test as a debugging shortcut.

## Output And Archive Policy

Runnable code remains on the matching stage branch. Compact text evidence is
synced back to GitHub `main` after each closed stage:

- route card;
- evidence README;
- command scripts;
- compact CSV, JSON, Markdown, log, and text summaries;
- decision records.

Do not commit checkpoints, weights, datasets, images, arrays, archives, raw
inference outputs, large per-image tables, selected-action tables, or raw
feature tables by default.

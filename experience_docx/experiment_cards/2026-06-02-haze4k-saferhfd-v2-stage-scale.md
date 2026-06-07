# Haze4K SafeRHFD-v2 Stage-Scale Diagnostic

Date: 2026-06-02

Status: completed low-cost diagnostic; no candidate passed the strict gate.

## Route

SafeRHFD-v2 is a stage-wise calibration of the existing B1-Surgery candidate:

- backbone: A0 official ConvIR-B Haze4K checkpoint;
- RHFD branches: copied from B1 stop20 `Best.pkl`;
- intervention: independently scale only the final conv of `PFD_RHFD1` and
  `PFD_RHFD2`;
- training: none.

This route is not B2/B3 and does not add HSCM, PFFB, physical priors, teacher
loss, or new optimization. It asks whether the already useful B1 RHFD surgery
can be made less dependent on a single hard-image gain while preserving easy
and strong-reference cases.

## Prior Evidence

The unified B1-Surgery sweep found:

- `scale=0.70`: mean PSNR `+0.01064 dB`, hard bottom-25% `+0.03317 dB`, easy
  top-25% `+0.00782 dB`, strong-reference regressions `0`, global regressions
  `9`;
- `scale=1.00`: mean PSNR `+0.01268 dB`, hard bottom-25% `+0.03888 dB`, easy
  top-25% `+0.00980 dB`, strong-reference regressions `9`, global regressions
  `31`.

The risk is that `scale=0.70` is partly carried by one very large hard-image
gain, so a stricter top-1-excluded robustness gate is required before promotion.

## Matrix

| ID | RHFD2 scale | RHFD1 scale | Purpose |
| --- | ---: | ---: | --- |
| current | 0.70 | 0.70 | Current primary B1-Surgery candidate. |
| isolate-2 | 0.70 | 0.00 | Test RHFD2-only contribution. |
| isolate-1 | 0.00 | 0.70 | Test RHFD1-only contribution. |
| safer-deep | 0.70 | 0.50 | Reduce RHFD1 depth risk. |
| safer-shallow | 0.50 | 0.70 | Reduce RHFD2 shallow risk. |
| gain-2 | 1.00 | 0.70 | Increase RHFD2 only. |
| gain-1 | 0.70 | 1.00 | Increase RHFD1 only. |
| asymmetric-a | 1.00 | 0.50 | High RHFD2, low RHFD1. |
| asymmetric-b | 0.50 | 1.00 | Low RHFD2, high RHFD1. |
| conservative | 0.80 | 0.60 | Nearby preservation-first adjustment. |
| conservative2 | 0.60 | 0.80 | Nearby reverse adjustment. |

## Strict Gate

A candidate must pass all checks:

- mean PSNR delta vs A0 `>= +0.005 dB`;
- mean SSIM delta `>= 0`;
- hard bottom-25% mean delta `>= +0.02 dB`;
- easy top-25% mean delta `>= 0`;
- severe regressions, `delta <= -0.20 dB`, `<= 0`;
- strong-reference regressions, top-25% A0 with `delta <= -0.05 dB`, `<= 0`;
- global regressions, `delta <= -0.05 dB`, `<= 10`;
- hard median delta `>= -0.001 dB`;
- hard positive ratio `>= 0.45`;
- mean delta excluding top-1 gain `>= 0`;
- hard delta excluding top-1 hard gain `>= 0`.

## Decision Rule

Promote only if at least one candidate passes the strict gate and improves on
the unified `0.70/0.70` robustness profile. If no candidate passes, keep
SafeRHFD-v1 `0.70` as diagnostic evidence only and do not launch B2/B3 from
this route.

## Outcome

The 11-candidate matrix completed on `autodl-dehaze3` at
`2026-06-02T15:20:16+08:00`.

No candidate passed the strict gate. The best failed diagnostic candidate was
`RHFD2=0.50, RHFD1=0.70`, but it still failed:

- severe regressions: `1`, threshold `0`;
- hard median delta: `-0.00136 dB`, threshold `>= -0.001 dB`;
- hard positive ratio: `0.44`, threshold `>= 0.45`;
- hard delta excluding top-1 hard gain: `-0.00164 dB`, threshold `>= 0`.

The unified `0.70/0.70` candidate also failed the stricter robustness gate:

- hard median delta: `-0.00234 dB`;
- hard positive ratio: `0.416`;
- hard delta excluding top-1 hard gain: `-0.00193 dB`.

Decision: `FAIL_STRICT_ROBUSTNESS_GATE`. Do not promote SafeRHFD-v2 stage-scale
or use it to justify B2/B3 training. Keep the evidence as a diagnostic closure
of the stage-wise scale idea.

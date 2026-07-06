# v2.33 P4 Exact Canary Open Questions

Open question after v2.33: why did P1 table-level WDMamba-alpha0.5 eligible
coverage reach `0.945`, while P4 canary32 had eligible coverage `5/32`.

Required follow-up, now handled by v2.34:

- recompute the exact P4 canary32 direct WDMamba-alpha0.5 benefit;
- compare P1 table eligibility with P4 crop-aligned eligibility;
- classify whether `5/32` came from sampling, mask threshold, crop alignment, or join/key mismatch;
- block projection/training if the exact canary lacks direct teacher benefit.

v2.34 result: the exact first32 canary had negative direct WDMamba-alpha0.5
benefit, and a rebuilt table-positive balanced canary still failed the
crop-aligned direct teacher-benefit gate. Therefore P1 free-tensor projection
was not authorized from those canaries.

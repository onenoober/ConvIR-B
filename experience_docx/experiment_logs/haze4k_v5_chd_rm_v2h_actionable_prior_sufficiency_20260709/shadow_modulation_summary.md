# v2h-B Shadow-Modulation Upper Bound

Status: `COMPLETED_GATE_PASS`

Decision label: `V2H_B_SHADOW_MODULATION_PASS_AUTHORIZE_OOF_NOOP_REVIEW`

Policy: diagnostic oracle shadow only. No training, no locked test, no D2/F5/v3/RARM.

## Alpha 0.3 Summary

| Selector | Global PSNR gain | Removed energy | Selected coverage |
| --- | ---: | ---: | ---: |
| D7c fixed | 1.374164 | 0.271242 | 0.289292 |
| Density matched | 0.977430 | 0.201533 | 0.279711 |
| Action oracle | 2.220821 | 0.400322 | 0.302513 |

## D7c Region Touch At Alpha 0.3

| Region | Touch rate | Region PSNR gain | Mean abs delta |
| --- | ---: | ---: | ---: |
| action_positive | 0.571134 | 1.695614 | 0.00361351 |
| negative_low_risk | 0.002698 | 0.010642 | 0.00000223 |
| isolated_ldhn | 0.023606 | 0.176447 | 0.00017179 |

## Decision

v2h-C OOF stability and v2h-D FAM2 no-op equivalence review only

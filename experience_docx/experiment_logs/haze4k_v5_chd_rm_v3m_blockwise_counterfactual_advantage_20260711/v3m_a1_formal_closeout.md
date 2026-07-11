# v3m A1 Formal Local Observability Closeout

Decision: `V3M_A1_LOCAL_SIGNAL_PASS_AUTHORIZE_A2_OOF_CALIBRATION_AUDIT_ONLY`.

The frozen 1,200-image OOF audit replayed `D_ref` and `D_rep` fixed
`alpha=0.125` exactly (`0 dB` maximum difference). It wrote 2,177,350
cloud-only block records plus header and matched all nine pinned inputs. No
test, route-confirm selection, canary, or training was used.

| Signal | D_ref AUROC CI95 low | D_rep AUROC CI95 low | Result |
| --- | ---: | ---: | --- |
| D7c score | `0.8035` | `0.8031` | pass |
| direct-step energy | `0.8522` | `0.8516` | pass |
| D7c x energy | `0.8501` | `0.8501` | pass |
| alpha1 clip fraction | `0.5433` | `0.5427` | fail |

Valid positive/negative labels covered 99.67% (`D_ref`) and 99.58% (`D_rep`)
of images. Direct-step energy is the fixed primary signal for A2 because it has
the strongest lower bound for both operators; this selection is from the
predeclared A1 result only and A2 must use fold-separated calibration.

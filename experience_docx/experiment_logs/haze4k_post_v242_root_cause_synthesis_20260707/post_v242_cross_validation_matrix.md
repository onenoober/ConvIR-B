# Post-v2.42 Cross-Validation Matrix

| Hypothesis | Evidence | Verdict |
| --- | --- | --- |
| v2.41 table/eval mismatch caused failure | v2.42 mismatch_count=0 | Rejected |
| beta/residual scale caused severe rows | 27/27 severe are direction_bad; 0/27 overshoot_bad | Rejected |
| shrink can rescue v2.41 | no gamma passed; best hard +0.0742 | Rejected |
| perfect no-op selector would make v2.41 useful | oracle clamp mean/hard only +0.0705/+0.1293 | Rejected |
| v2.41 failed due to OOF overfit only | train32 full-image also failed | Rejected |
| teacher positive transfer does not exist | v2.37 oracle and v2.40 WDMamba alignment show teacher headroom | Rejected |
| teacher positive transfer is deployably selectable | v2.37/v2.38B/v2.40 selector/predictability failed | Rejected |
| current frozen small residual architecture is viable | v2.41/v2.42 failed | Rejected |
| non-post model-line improvement is impossible | ConvIR scaling and WDMamba prior support model-level feasibility | Not supported |

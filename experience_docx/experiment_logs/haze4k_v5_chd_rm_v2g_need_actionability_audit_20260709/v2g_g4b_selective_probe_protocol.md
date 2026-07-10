# v2g G4b Selective Probe Screen Protocol

```json
{
  "status": "AUTHORIZED_G4B_SMALL_SCREEN_ONLY",
  "route_identity": "G4b small selective-head/probe screen under v2g three-state target",
  "forbidden": [
    "locked Haze4K test",
    "D2",
    "RARM connection/training",
    "v3",
    "F5",
    "saving checkpoints/weights",
    "model promotion"
  ],
  "training_scope": "balanced train_inner pixel samples; tiny linear/MLP probes only; thresholds selected on train_inner calibration subset",
  "evaluation_scope": "val_inner only; D7c and D3 density controls reported with matched train coverage",
  "primary_screen_gate": {
    "candidate_must_beat_d7c_action_recall_by_at_least": 0.02,
    "candidate_low_adjacent_recall_must_be_at_least_d7c": true,
    "candidate_negative_false_rate_max": 0.005,
    "candidate_ignore_hit_rate_max": 0.05,
    "candidate_isolated_ldhn_hit_rate_max": 0.03,
    "candidate_selected_coverage_range": [
      0.25,
      0.35
    ],
    "passing_does_not_authorize": [
      "F5",
      "v3",
      "RARM",
      "D2",
      "locked test"
    ]
  }
}
```

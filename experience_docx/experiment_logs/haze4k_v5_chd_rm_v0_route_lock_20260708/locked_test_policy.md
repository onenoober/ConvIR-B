# CHD-RM Locked-Test Policy

Date: 2026-07-08

## Policy

Haze4K locked test is final-confirmation only. It must remain untouched through
v1-v7.

## Forbidden Before v8

Do not use locked test to choose:

- checkpoints;
- thresholds;
- route variants;
- gamma caps;
- low-haze masks;
- loss weights;
- scale choices;
- model structures;
- random seeds;
- final candidate identity.

## Allowed Before v8

Only train-derived evidence may guide route decisions:

- canary subsets from Haze4K train;
- fixed internal 2400/600 split from Haze4K train;
- 5-fold OOF from Haze4K train;
- pre-registered matched-budget controls.

## v8 Entry Condition

The locked test may be used only after v7 writes:

- `candidate_lock.md`;
- `final_config_frozen.yaml`;
- `final_thresholds_frozen.json`;
- `final_seed_policy.md`;
- the fixed checkpoint-selection rule.

Any accidental locked-test access before those files exist invalidates the
affected selection step and must be recorded as a protocol failure.

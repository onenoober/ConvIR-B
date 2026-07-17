# Data Role Deviation
Date: 2026-07-17
Status: recorded; historical A1X 432-name confirmation role retired.
The retrospective audit scanned the full v3p 1,200-image canonical candidate-loss table to compute action margins and cross-operator agreement. The historical A1X 432-name remainder was identified after that full-table scan. Consequently, those 432 outcomes are part of aggregate audit statistics and cannot be treated as independent confirmation evidence for a future R3 route.
No Haze4K image, clean target, prediction, array, model, or checkpoint was decoded or loaded. No per-name action outcome was retained, ranked, or used for model selection. Existing compact DTA D9 locked summary evidence was read, but no locked data or locked command was accessed.
Corrective rule:
- reclassify the historical 432 as `historical_audit_only`;
- do not use it for R3 threshold, feature, action, model, checkpoint, or confirmation selection;
- draw the next development/confirmation ledger only from the 1,200 train-inner images outside the v3p action-label chain; and
- freeze that ledger before generating any new action outcome.

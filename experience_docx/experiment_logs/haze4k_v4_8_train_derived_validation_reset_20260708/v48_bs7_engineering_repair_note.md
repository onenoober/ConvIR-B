# v4.8 batch-size engineering repair

Time: 2026-07-08T10:27+08:00

Classification: engineering/preflight repair, not scientific evidence.

The original fold2 and fold4 training commands failed because the final training batch had size 1, which is invalid for the BatchNorm path in `square_att` after global pooling (`torch.Size([1, 72, 1, 1])`). This was caused by train-count modulo batch-size 8 equaling 1 for fold2 (2385 % 8 = 1) and fold4 (2417 % 8 = 1).

Repair decision: rerun all five folds with batch_size=7 and new model names ending in `bs7repair-20260708`. This keeps the five-fold OOF comparison internally consistent instead of mixing batch_size=8 successful folds with batch_size=7 repaired folds. The data splits, seed, A0 init, architecture, train scope, stop epoch, and locked-test policy remain unchanged.

Locked-test policy: train-derived folds only. No locked-test enumeration or evaluation is authorized by this repair.

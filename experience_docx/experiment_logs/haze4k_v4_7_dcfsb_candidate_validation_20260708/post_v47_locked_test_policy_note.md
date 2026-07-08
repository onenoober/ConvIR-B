# Post-v4.7 Locked-Test Policy Note

After the v4.7 internal candidate-lock audit had completed and written `locked_test_touched=false` and `test_split_enumerated=false` for that audit phase, a follow-up command used to locate the evaluation entry listed the Haze4K test directory and file counts.

No locked-test images were opened, no inference or evaluation was run, and no locked-test metric was produced. However, the directory/file-count lookup is an enumeration of the locked test split under this repository's policy.

Consequence: v4.7 remains a valid train-derived internal audit, but subsequent locked-test access should not be described as starting from a clean `touched/enumerated=false/false` state. Do not use the directory count as scientific evidence.

# CHD-RM v1 Split Policy

- Source data: Haze4K train 3000 only for train-derived selection.
- Internal split: stratified 2400 train_inner / 600 val_inner, seed 3407.
- OOF: five folds from train 3000, 600 validation images per fold.
- Locked test: not used for tuning; only counted for leakage accounting.
- Non-image files are ignored by extension filtering.

# v3k-C Open Holdout Contract

This phase evaluates OOF-preselected fixed alpha policies on historical
train-derived `val_inner`. It is useful as an open-holdout diagnostic only.

Predeclared policy roles:
- primary: context alpha=0.125
- secondary: context alpha=0.25
- linear policies: architecture consistency controls

This phase does not use locked test and does not create a new sealed split.
Therefore even a pass cannot authorize canary or promotion.

# v2g Actionable Need Target Definition

Status: `COMPLETED_G3_ACTIONABLE_TARGET_DEFINITION`

This is a target-semantics definition, not a trained model and not a promotion claim. Locked Haze4K test, D2, RARM, v3, and F5 remain unused.

## Three-State Contract

Positive/actionable:

```text
action_positive = (target >= q66 and density > density_q33)
               OR (target >= q66 and density <= density_q33 and adjacent_to_haze)
```

Negative/confident low-risk:

```text
negative_low_risk = density <= density_q33 and target <= q33
```

Ignore/abstain:

```text
ignore_abstain = isolated LDHN
              OR low-density mid-need
              OR boundary/unstable LDHN
```

Other unlabeled pixels are outside this selective low-haze actionability contract.

## Rationale

G2b shows isolated LDHN removes residual energy under oracle replacement, but it is not reliably haze-actionable. Therefore isolated LDHN should not be forced as a hard RARM positive. The low-density adjacent-to-haze subset remains the candidate actionable LDHN subset.

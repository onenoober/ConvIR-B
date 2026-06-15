# Locked C11 Selector One-Shot Decision

Decision: `LOCKED_C11_SELECTOR_ONE_SHOT_RECORDED_DO_NOT_PROMOTE_OVER_WD0375`

mean/hard/easy/positive/severe: `1.449078` / `1.558683` / `1.248566` / `0.896000` / `48.60/600`.

Compared with fixed WD0375 locked (`1.442090` / `1.529767` / `1.182529` /
`0.938000` / `25.80/600`), the selector only slightly improves mean/hard/easy
but materially worsens positive ratio and severe tail risk. Keep WD0375 as the
default locked-pass strong baseline.

Locked output is evidence only. It must not tune alpha, features, checkpoints, profiles, actions, experts, or distillation targets.

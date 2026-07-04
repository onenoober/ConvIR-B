# v2.21 Raw v2.20 Action Anchor

Primary raw action: `P1_final_mid_global_context_predictor`

- mean dPSNR: `2.09111962556839`
- hard bottom25 dPSNR: `4.2276796499888105`
- easy top25 dPSNR: `0.5341034189860026`
- p05 dPSNR: `-0.7040210723876953`
- CVaR5 dPSNR: `-1.4500590960184734`
- severe rate: `0.11`
- strong-reference regression rate: `0.25166666666666665`
- v2.20 mechanism pass: `True`
- v2.20 training authorization pass: `False`

v2.21 does not train from this raw action. It only tests whether safety/no-op calibration can make the action replay-safe.

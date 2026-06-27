# Haze4K v2.1 C7c Local-Alpha Severe-Risk Tightening

Decision: `C7C_RISK_TIGHTEN_STRONG_PASS_START_C9_SHIFTED_STRONG`

C7c reuses C7b patch feature/SSE rows, selects stricter train-fold policies, and re-renders held-out images once for all risk profiles. Locked test data was not touched.

| Profile | Mean | Hard | Easy | dSSIM | Positive | Severe/600 | Screen | Strong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `riskcap36_no075` | `0.341530` | `0.310932` | `0.443958` | `0.00024241` | `0.786667` | `37.0` | `True` | `True` |
| `riskcap42_no075` | `0.354799` | `0.322247` | `0.451988` | `0.00024897` | `0.790000` | `43.0` | `True` | `True` |
| `riskcap48_allow075` | `0.376111` | `0.360949` | `0.443171` | `0.00025762` | `0.793333` | `50.0` | `False` | `False` |
| `riskcap48_no075` | `0.380483` | `0.356692` | `0.460197` | `0.00026164` | `0.795000` | `50.0` | `False` | `False` |

## Interpretation

C9 shifted-strong validation is authorized only if a risk profile passes the strong gate. Locked test remains blocked.

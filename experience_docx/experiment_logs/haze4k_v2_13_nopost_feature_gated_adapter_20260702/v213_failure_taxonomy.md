# v2.13 Failure Taxonomy

case 1: no-op parity fail

=> implementation issue. Fix zero-init, partial load, normalization, or forward
path before any training.

case 2: N1 AUC fail

=> evidence/gating separability fail. Do not train the adapter until stronger
internal evidence or priors are designed.

case 3: N1 pass, N3 train fail

=> optimization, capacity, or gradient path failure.

case 4: N3 train pass, N4 validation fail

=> overfit or calibration failure.

case 5: mean/hard pass, easy/severe fail

=> action budget or risk gate failure.

case 6: low-only pass, detail fail

=> detail branch unsafe. Keep low-only.

case 7: final-only fail, mid feature probe pass

=> consider a new declared mid+final route.

case 8: adapter-only weak but safe

=> adapter only may be underpowered; neighbor unfreeze requires a new written
stage decision.

case 9: 5x3 variance larger than gain

=> below noise standard; no promotion.

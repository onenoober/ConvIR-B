# v2.16 T2 Contract Audit

Decision scope: contract and zero-init identity only. No training is launched.

- Forward input: hazy image only.
- Insertion point: official ConvIR-B final decoder feature, after `Decoder[2]` and before `feat_extract[5]`.
- Active branch: Haar LL lowband branch only.
- Projection: zero-initialized `lowband_project`.
- Output path: original `feat_extract[5]`, then `rgb_residual + hazy`.
- Locked Haze4K test: untouched.

# v3m-A0a Metric Contract

Image metric is per-image PSNR delta relative to A0. Oracle selection itself
uses paired clean-reference ground truth exactly as a privileged upper-bound
audit. Block utility is not added in PSNR units; the A0a policy gate is based on
image-level replay, grouped paired bootstrap, p10, worst, and severe counts.

`<= -0.2 dB` is the severe definition. `<= -0.5 dB` is reported as a harder
tail diagnostic. The route does not claim that zero observed severe implies
per-sample no-harm.

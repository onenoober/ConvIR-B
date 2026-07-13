# v3x Projected Direct Safety Evidence

Status: `PLANNED`. Only compact manifests, closeouts, history, and summaries
may enter this directory. Cloud checkpoints, images, raw outputs, and tables
remain outside Git.

S0-r1 passed exact no-op on all 32 fixed names and both frozen operators:
maximum absolute Delta-u, prediction difference, and reference-PSNR replay
difference were all zero. It authorizes only the fixed32 projected direct-safety
diagnostic S1.

S1-r1 passed the mechanism gate. The render-only warmup reproduced the v3v/v3w
midpoint. With direct gradient projection in epochs 9-16, final rendered-MSE
reduction was `1.69283%`, final `|Delta u|` was `0.00194937`, and all three
fixed v3u safety references were met. Projection affected `87.5%` of updates.
This authorizes safety-contract design only; it does not authorize policy,
canary, candidate training, deployment, or locked-test access.

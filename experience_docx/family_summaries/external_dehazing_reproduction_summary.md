# External Dehazing Reproduction Summary

Date: 2026-09-19

## Current Verdict

DehazeFormer is an external dehazing reproduction route and is separate from
the ConvIR-B Haze4K architecture families. The cloud engineering smoke passed
for the official `IDKiro/DehazeFormer` source at commit
`5af7c34d6f8e0784a88a2132a7add296bf99ca9c`: `dehazeformer-s` imported, ran a
finite CUDA forward, loaded a DataParallel-format checkpoint, and completed the
official one-image `test.py` path.

The smoke used deterministic random weights. Its PSNR/SSIM values are not
scientific evidence and must not be compared with published results. The route
is ready for official-checkpoint testing once the user uploads the matching
checkpoint format.

## Reopen / Next Step

Run a new cloud evaluation with the official checkpoint, a complete
filename-aligned RESIDE-IN test tree, and a new output root. Preserve the
engineering smoke as the pipeline baseline and do not commit weights, images,
datasets, or raw inference outputs.

Primary evidence: `experience_docx/experiment_logs/dehazeformer_repro_20260919/`.

# External Dehazing Reproduction Summary

Date: 2026-09-21

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

AECR-Net and FFA-Net are separate external dehazing reproductions. Their
official source revisions were pinned, and both passed Haze4K train-crop CUDA
forward/backward, one Adam update, and strict checkpoint reload on
`convir-4090`. AECR uses a maintained torchvision modulated deformable
convolution implementation instead of its Python 3.6 DCNv2 binary; numerical
parity with the legacy extension is not yet established. The one-step losses
are engineering checks, not quality results. The two author-provided model
weights are still pending. See the
[route card](../experiment_cards/2026-09-21-aecr-ffa-repro.md) and
[evidence](../experiment_logs/aecr_ffa_repro_20260921/).

## Reopen / Next Step

Run a new cloud evaluation with the official checkpoint, a complete
filename-aligned RESIDE-IN test tree, and a new output root. Preserve the
engineering smoke as the pipeline baseline and do not commit weights, images,
datasets, or raw inference outputs.

Primary evidence: `experience_docx/experiment_logs/dehazeformer_repro_20260919/`.

For AECR-Net and FFA-Net, provenance-check the supplied weights before an
official-checkpoint evaluation. A full Haze4K training comparison would need a
predeclared train-derived split, schedule, metric contract, and quality gates;
the present smoke does not authorize locked-test evaluation.

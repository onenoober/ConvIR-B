# External Dehazing Baseline Reproduction

Date: 2026-09-21

Status: engineering smoke only.

AECR-Net and FFA-Net were checked out at fixed official source commits and
smoked on one Haze4K training crop on `convir-4090`. Both passed finite
forward/backward, one Adam update, and strict checkpoint reload. This neither
reproduces the authors' pretrained results on RESIDE nor establishes a Haze4K
comparison against ConvIR-B. AECR uses current torchvision modulated deformable
convolution in place of the upstream Python 3.6 binary; numerical parity is
unverified. See the [route card](../experiment_cards/2026-09-21-aecr-ffa-repro.md)
and [evidence](../experiment_logs/aecr_ffa_repro_20260921/).

Reopen for a checkpoint-provenance/load audit when the official weights are
provided. A full Haze4K training comparison would additionally require a fixed
train-derived validation split, loss/schedule/budget, and separate quality gates
before any locked-test evaluation.

# ConvIR-Dehaze-v1.5-FullUDP

Date: 2026-06-05

Status: Phase 0 official UDPNet ConvIR reproduction audit completed;
checkpoint acquisition blocker reopened after the official checkpoint was
provided on the replacement `dehaze1`; official eval pending.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: full UDPNet-level depth-guided feature integration for ConvIR.
- Dataset or task: Haze4K, using the existing A0 ConvIR-B baseline and current
  train-derived internal validation policy before any locked test.
- Primary objective: test whether the official ConvIR+UDP checkpoint is
  substantially stronger than A0 under this repository's Haze4K data, metric,
  depth-preprocessing, and evaluation contract.
- Execution environment: cloud server `dehaze1`; the WSL checkout is editing
  and compile/syntax-only.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605/`.
- Branch or isolated workspace:
  `codex/haze4k-convir-v1-5-full-udpnet-transplant`.

## Baseline Contract

- Baseline implementation: official ConvIR-B A0 checkpoint
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Candidate Phase 0 implementation: official UDPNet repository
  `/root/autodl-tmp/workspace/UDPNet`, model file
  `Dehazing/ITS/models/ConvIR_UDPNet.py`.
- Official candidate checkpoint target:
  `/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt`.
- Required checkpoint source evidence: share URL, file name, size, sha256, and
  download/provenance status.
- Evaluation output if checkpoint is available:
  `udpnet_convir_repro_eval.json`,
  `udpnet_convir_bucket_compare.csv`, and
  `udpnet_convir_failure_audit.csv`.
- Locked-test policy: blocked until the reproduction gate passes and the
  selected checkpoint/eval wrapper is fixed.

## Motivation

v1.4A and v1.4B are engineering-valid but too weak under the
A0-equivalent, frozen ConvIR-B, small-adapter contract. v1.4B Best remains near
`+0.0286 dB` regular and `+0.0234 dB` hard while failing positive ratio, SSIM,
strong-regression, and worst-tail checks. The next highest-value question is no
longer a small UDP-Lite tweak; it is whether full UDPNet depth integration is
reproducibly strong under the current Haze4K protocol.

## Phase 0: Official UDPNet ConvIR Reproduction Audit

Question:

```text
Does official ConvIR+UDP beat A0 by a large, tail-safe margin under this
repository's Haze4K data, metrics, RGB/depth preprocessing, checkpoint, and
evaluation wrapper?
```

Required fields:

- mean PSNR/SSIM delta versus A0;
- hard bottom-25 delta;
- easy top-25 delta;
- positive ratio;
- strong regression ratio;
- worst `<= -0.20 dB` count;
- runtime and peak memory;
- depth preprocessing checksum/layout;
- checkpoint source and sha256.

Gate:

- mean delta versus A0 `>= +0.30 dB`;
- hard bottom-25 delta `>= +0.20 dB`;
- easy top-25 delta `>= -0.03 dB`;
- SSIM delta `>= 0`;
- strong/worst tail not explosively high.

If the checkpoint cannot be obtained or the official code cannot be evaluated
under a controlled wrapper, Phase 0 is an acquisition/protocol blocker, not a
scientific UDPNet failure.

## Phase 1: Controlled FullUDP Transplant

Only if Phase 0 passes, implement an A0-initialized FullUDP-ConvIR transplant:

- copy A0 RGB weights into 4-channel input stem/SCM where shapes change;
- zero or near-zero initialize added depth channels;
- include depth refinement, DGAM/input depth attention, and `fusion1/2/3`;
- train UDP modules first, then fusion-neighbor ConvIR interfaces at low LR;
- keep encoder/decoder broad finetune blocked unless Stage 2 is stable.

## Phase 2: Teacher-Guided Distillation

Only if the official ConvIR+UDP checkpoint is strong, use it as a teacher for a
full UDP feature architecture. Do not train a plain output residual to mimic
teacher residuals.

## Stop Items

- Stop v1.4B BiDPFM1-only scale/gate/loss search.
- Do not continue v1.4C small adapter as the direct next route.
- Do not run locked Haze4K test before the written internal gates pass.
- DPFM2 remains blocked only under the UDP-Lite frozen wrapper; do not
  mechanically disable it inside a full UDPNet transplant.

## Analysis Plan

1. Run `run_v15_phase0_repro_audit.sh` on `dehaze1`.
2. If the official checkpoint is available, build a controlled ConvIR_UDPNet
   eval wrapper and evaluate A0 vs official UDPNet.
3. If Phase 0 passes, open Stage 1/Stage 2 transplant scripts and gates as a
   separate controlled run.
4. If the official checkpoint is unavailable, record the blocker and do not
   claim README-level UDPNet reproduction.
5. Sync text evidence back into `experience_docx/`, update the index/family
   summary, then commit and push to GitHub.

## Phase 0 Result

- Audit completed on `dehaze1` at `2026-06-05T01:33:42+08:00`.
- Output JSON:
  `experience_docx/experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605/phase0_repro_audit/udpnet_convir_repro_eval.json`.
- Official Baidu share listed `ConvIR_UDPNet_haze4k.ckpt` with
  `fs_id=883266741305581` and size `108206629` bytes.
- Local official checkpoint path
  `/root/autodl-tmp/workspace/UDPNet_checkpoints/ConvIR_UDPNet_haze4k.ckpt`
  was absent.
- Baidu `api/sharedownload` returned a client-encrypted task list, not a plain
  HTTP `dlink`.
- `BaiduPCS-Go transfer --download` failed to retrieve public share metadata
  without a logged-in account.
- No official UDPNet PSNR/SSIM eval was run; all metric fields are `null`.

## Decision

- Decision label: `PHASE0_BLOCKED_OFFICIAL_UDPNET_CHECKPOINT_UNAVAILABLE`.
- Image/global metric reason: no image metrics are available because the
  official ConvIR+UDP checkpoint could not be obtained as a durable file with
  sha256.
- Mechanism reason: official full UDPNet remains plausible but unverified under
  this repository's Haze4K protocol.
- Preservation or regression reason: no per-image compare, hard/easy bucket, or
  tail audit can be claimed without the checkpoint.
- Cost/deployability reason: the official share currently requires a
  BaiduNetdisk-client/account path rather than a reproducible checkpoint fetch.
- Evidence strength label: completed checkpoint acquisition and protocol audit;
  not a scientific UDPNet eval.
- What this decides next: do not start FullUDP transplant or teacher
  distillation from README-level claims alone. Resume Phase 0 only if the
  official checkpoint is supplied or becomes downloadable with sha256; otherwise
  evaluate available stronger-backbone candidates under a separate controlled
  route.

## Phase 0 Reopen

- Reopen time: 2026-06-05.
- Replacement `dehaze1` endpoint: `root@connect.bjb1.seetacloud.com:42371`
  through the existing `~/.ssh/id_ed25519_seetacloud` key.
- Official checkpoint path now available:
  `/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt`.
- Official checkpoint sha256:
  `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291`.
- Resume action: run the controlled `val_regular` and `val_hard` official
  ConvIR+UDP reproduction eval before any transplant training or locked test.

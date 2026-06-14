# Haze4K v2.0 C1c FullUDP Render Availability

Decision: `C1C_FULLUDP_RENDER_READY`

Locked test data was not touched.

## Checks

- UDPNet repo: `/sda/home/wangyuxin/ConvIR-B/repos/UDPNet`
- Expected commit: `f925387e690ae6016ffbd4b1cfd8490d75d7a334`
- Actual commit: `f925387e690ae6016ffbd4b1cfd8490d75d7a334`
- ConvIR_UDPNet model file exists: `True`
- Official checkpoint path: `/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/ConvIR_UDPNet_haze4k.ckpt`
- Checkpoint exists: `True`
- Checkpoint sha256: `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291`
- Expected sha256: `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291`
- Checkpoint sha match: `True`
- BaiduPCS tool path: ``
- Render ready: `True`

## Interpretation

- C1b showed A0-PSNR-only deployable proxies are not enough for C2.
- C1c checks whether convir-4090 can render FullUDP outputs for real output-difference features.
- Next action: render FullUDP/A0 outputs and compute output-difference features.

# UDPNet ConvIR Reproduction Audit

- Repo dir: `/root/autodl-tmp/workspace/UDPNet`
- ConvIR UDP model file: `/root/autodl-tmp/workspace/UDPNet/Dehazing/ITS/models/ConvIR_UDPNet.py`
- ConvIR UDP model exists: `True`
- README exists: `True`

## README Lines

- L1: # :fire: UDPNet: Unleashing Depth-based Priors for Robust Image Dehazing
- L4: > UDPNet: Unleashing Depth-based Priors for Robust Image Dehazing<br>
- L13: >Image dehazing has witnessed significant advancements with the development of deep learning models. However, a few methods predominantly focus on single-modal RGB features, neglecting the inherent correlation between scene depth and haze distribution. Even those that jointly optimize depth estimation and image dehazing often suffer from suboptimal performance due to inadequate utilization of accurate depth information. In this paper, we present UDPNet, a general framework that leverages depth-based priors from large-scale pretrained depth estimation model DepthAnything V2 to boost existing image dehazing models. Specifically, our architecture comprises two typical components: the Depth-Guided Attention Module (DGAM) adaptively modulates features via lightweight depth-guided channel attention, and the Depth Prior Fusion Module (DPFM) enables hierarchical fusion of multi-scale depth map features by dual sliding-window multi-head cross-attention mechanism. These modules ensure both computational efficiency and effective integration of depth priors. Moreover, the intrinsic robustness of depth priors empowers the network to dynamically adapt to varying haze densities, illumination conditions, and domain gaps across synthetic and real-world data. Extensive experimental results demonstrate the effectiveness of our UDPNet, outperforming the state-of-the-art methods on popular dehazing datasets, such as 0.85 dB PSNR improvement on the SOTS dataset, 1.19 dB on the Haze4K dataset and 1.79 dB PSNR on the NHR dataset.
- L15: :star: If UDPNet is helpful to your projects, please help star this repo. Thank you! :point_left:
- L58: | ConvIR-B **(Baseline)** | TPAMI'24 | 42.72       | 0.997       | 39.42        | 0.996        |
- L63: | **ConvIR + UDP (Ours)** | –        | 43.12       | 0.997       | 40.32        | 0.996        |
- L64: | **FSNet + UDP (Ours)**  | –        | **43.30**   | **0.997**   | **40.53**    | **0.997**    |
- L68: <details> <summary>Table 2. Haze4K</summary>
- L80: | ConvIR-B **(Baseline)** | 34.15     | 0.99     |
- L83: | **ConvIR + UDP (Ours)** | 34.82     | 0.99     |
- L84: | **FSNet + UDP (Ours)**  | **35.31** | **0.99** |
- L99: | ConvIR-B **(Baseline)**  | 31.83     | 0.921     |
- L101: | PoolNet + UDP            | 32.78     | 0.930     |
- L102: | **ConvIR + UDP**         | **33.12** | **0.933** |
- L119: | ConvIR-B **(Baseline)** | 29.49     | 0.983     |
- L121: | FSNet + UDP             | 28.09     | 0.980     |
- L122: | **ConvIR + UDP**        | **29.54** | **0.983** |
- L141: | ConvIR-S **(Baseline)** | 17.45      | 0.65       | 0.6000      | 20.65     | 0.80     | 0.3669     |
- L142: | **ConvIR + UDP**        | 17.55      | **0.67**   | 0.5813      | **20.98** | **0.82** | **0.3567** |
- L143: | **FSNet + UDP**         | **17.85**  | 0.65       | 0.6033      | 20.94     | 0.82     | 0.3732     |
- L160: | ConvIR-S **(Baseline)**  | 25.11     | 0.978     | 26.79         | 0.978         | 22.65      | 0.950      |
- L162: | **ConvIR + UDP**         | 25.48     | 0.979     | 28.07         | **0.981**     | 22.95      | 0.953      |
- L163: | **PoolNet + UDP**        | **26.20** | **0.980** | **28.26**     | 0.979         | **23.13**  | 0.951      |
- L189: | **PromptIR + UDP (Ours)** | *31.33*   | *0.980*   | 37.63     | 0.980     | 31.25     | 0.883     | 28.34     | 0.868     | 23.18     | *0.851*   | 30.35     | 0.912     |
- L190: | **AdaIR + UDP (Ours)**    | **31.41** | **0.980** | 37.85     | 0.980     | 31.28     | 0.888     | 28.62     | 0.870     | **23.53** | **0.854** | **30.55** | **0.915** |
- L199: @article{zuo2026udpnet,
- L200: title={{UDPNet}: Unleashing Depth-based Priors for Robust Image Dehazing},
- L209: This code is based on [FSNet](https://github.com/c-yn/FSNet), [ConvIR](https://github.com/c-yn/ConvIR), [PoolNet](https://github.com/c-yn/PoolNet) and [AdaIR](https://github.com/c-yn/AdaIR).

## Checkpoint-Like URLs

- https://pan.baidu.com/s/1JqB-YBPzZAiQsdLlNcidLQ?pwd=2026

## Next Manual Eval Checks

- Confirm official checkpoint availability and license.
- Run the official UDPNet Haze4K eval on the same data root, RGB order, crop/padding, and metrics used by ConvIR-B.
- Compare official UDPNet ConvIR per-image rows against the local A0 baseline and write bucket deltas.

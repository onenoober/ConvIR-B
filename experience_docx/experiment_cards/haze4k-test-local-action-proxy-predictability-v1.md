# Haze4K development local-action proxy predictability

Status: PLANNED

- Route id: haze4k-test-local-action-proxy-predictability-v1
- First operation: HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY
- Program contract: experience_docx/research_programs/haze4k_test_local_action_proxy_predictability_v1.json
- Experiment spec: experience_docx/experiment_specs/haze4k-test-local-action-proxy-predictability-v1.json
- Scientific contracts: experience_docx/scientific_contracts/

## Scientific rationale

The isolated Haze4K development oracle established large spatial headroom beyond a GT-privileged per-image uniform action, but that headroom is non-deployable. This route asks the next necessary question without designing a module: can a low-capacity predictor using only the hazy image, fixed official output, and frozen decoder features select local keep, weaken, or strengthen actions out of scene? Five outer folds hold out entire canonical clear scenes; all target construction, alpha selection, and conservative threshold selection remain inside each outer training partition. The decisive result is OOF image replay, not tile classification. It must improve the official keep output, beat the same predictor collapsed to one image-uniform action, beat a shuffled-target falsification control, and pass scene-level PSNR, SSIM, and color tail-harm gates. Candidate-confirmation data, NH-HAZE, module construction, and network training remain prohibited.

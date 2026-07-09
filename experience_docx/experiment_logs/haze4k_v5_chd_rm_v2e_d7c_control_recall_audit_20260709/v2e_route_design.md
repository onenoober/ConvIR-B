# CHD-RM v2e Route Design

v2e is a D7c control and recall-protection audit. It keeps ConvIR-B frozen, keeps RARM disconnected, and does not use the locked test. The route first audits the frozen v2d D7c top-k candidate, then evaluates fixed image-level permutation controls, density-only matched-threshold controls, and low-density high-need recall. D7c-RP is authorized only if controls are clean but LDHN recall is the remaining blocker.

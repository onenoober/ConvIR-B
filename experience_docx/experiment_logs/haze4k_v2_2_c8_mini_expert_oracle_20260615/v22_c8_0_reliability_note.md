# C8-0 Command Reliability Note

- `git clone git@github.com:onenoober/ConvIR-B.git` failed on `convir-4090` with host-key verification; corrected to HTTPS clone.
- A broad checkpoint search command with redirection inside a Bash `for` item list failed; corrected to direct `find ... 2>/dev/null`.
- Baidu direct download probes returned landing/auth pages, so external checkpoints are unavailable without authenticated transfer or alternate mirrors.

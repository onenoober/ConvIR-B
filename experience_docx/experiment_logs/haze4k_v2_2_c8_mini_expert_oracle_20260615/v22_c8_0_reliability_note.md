# C8 Command Reliability Notes

- Invalid local shell-boundary form: PowerShell command attempted Bash `<<< $script` redirection into WSL and failed with `Missing file specification after redirection operator`.
- Corrected form: `$script | wsl -d Ubuntu-22.04 bash -lc "tr -d '\r' | bash"`.
- Remote `rg` was unavailable; corrected by using `grep`/`find` for read-only inspection.
- `printf '--- text'` can be parsed as an option by shell builtins; corrected to `printf '%s\n' '--- text'`.
- Initial `pip install causal-conv1d mamba-ssm` without `--no-deps` attempted to resolve/download `torch-2.12.0`; process was killed before torch changed. Corrected WDMamba install used explicit torch2.5/cu12 wheels with `--no-deps`.
- All runtime validation, inference, smoke tests, and aggregation ran on `convir-4090`; local WSL was not used for project runtime.

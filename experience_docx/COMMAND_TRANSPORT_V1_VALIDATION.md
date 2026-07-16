# Command Transport v1 Validation

Date: 2026-07-16

Status: `PASS`; candidate `7dc1296f798cfa4e91b2242d5e4b0597aa975240`;
`model_calls=0`. This was command-control validation only: no experiment, GPU,
dataset, training, evaluation, inference, or model router was invoked.

## Result

The exact candidate versions of `convirctl.py`, `test_convirctl.py`,
`convir_ops_mcp.py`, and `test_convir_ops_mcp.py` were transferred to
`convir-4090` with four verified SHA-256 digests, compiled with the explicit
cloud Python, and tested there. Result: 11/11 command-transport tests and 17/17
MCP-v4 control tests passed; total 28/28. Durable cloud status:
`/sda/home/wangyuxin/ConvIR-B/command_transport_validation/offline_attempt.iAsgt0/status.txt`;
terminal marker: `COMMAND_TRANSPORT_OK`.

The fallback bundle was validation-only and was deleted after closeout. It is
not a general experiment transport. Normal route execution remains the bounded
MCP-v4 lifecycle; unavailable GitHub/cloud infrastructure remains a reported
blocker, not authorization to bypass route identity.

## Historical Failure Coverage

| Prior failure class | v1 control | Automated evidence | Classification | Residual risk |
| --- | --- | --- | --- | --- |
| nested PowerShell/WSL/SSH quotes, regex, pipes, `$()` | literal argv plus committed script on SSH stdin; no shell command parameter | metacharacters remain data; `run_argv` cannot invoke a shell; remote stdin is exact | systemically removed from the supported path | a user can still bypass the rule manually |
| PowerShell expands syntax before WSL | Windows rule is only `wsl.exe --exec <program> <argv...>` | real candidate Git/SHA checks were invoked through this boundary; metacharacter argv test passed | removed from the supported path | PowerShell itself remains external software |
| Windows executables leak into WSL PATH | fixed `/usr/bin/git`, `/usr/bin/ssh`, `/bin/bash`; explicit cloud Python | fixed-interface assertions passed | systemically removed in the controller | programs can genuinely be absent and will fail preflight |
| CRLF or UTF-8 BOM corrupts Bash | bounded byte normalization before `bash -n` and SSH | BOM/CRLF normalization test passed | systemically removed for accepted scripts | other encodings and NUL are rejected |
| SSH consumes wrapper stdin and skips later commands | exactly one committed script is the SSH stdin; result is returned after process exit | full stdin and exact SSH argv test passed | systemically removed from the supported path | remote process behavior can still fail and is typed |
| implicit Python/PATH differs by host | explicit binaries and cloud interpreter | interface assertions and cloud compilation passed | removed from the supported path | missing fixed runtime is a visible preflight failure |
| successful no-output command appears hung | one JSON result always includes state, exit code, and marker; runs use status/closeout markers | every CLI test requires one JSON object and marker | systemically removed for the controller | external tools outside the controller may still be silent |
| Git identity/ref/worktree confusion | exact HEAD/branch/clean/remote-ref checks and SHA-256 | clean/dirty, detached branch, missing ref, remote mismatch, unsafe names, SHA match/mismatch passed | automatically detected or rejected | GitHub/network availability is not controllable |
| host-key, credential, or GitHub transfer stall | non-interactive Git/SSH plus finite timeout and typed failure | actual HTTPS low-speed failure and SSH fetch timeout stopped without experiment execution | detected and bounded, not eliminated | network and GitHub can still be unavailable |
| timeout triggers blind relaunch | remote timeout becomes `REMOTE_STATE_UNKNOWN`, `inspect_once`, `blind_retry_allowed=false` | timeout test passed | blind retry is automatically forbidden | one human/agent inspection is still required |
| untracked, changed, binary, oversized, or escaping script | workspace containment, tracked unchanged Git blob, UTF-8/NUL and 256-KiB limits | path escape, untracked/dirty, NUL, oversize, missing strict mode, syntax error passed | automatically rejected | trusted committed Bash can still contain a logical error |
| excessive stdout/stderr destabilizes control | each stream is drained with a 64-KiB retained cap | large-output truncation test passed | memory growth bounded | full raw output is intentionally not returned |
| dispatcher/watcher amplifies cost or loops | neither is part of command transport; MCP-v4 observation is bounded | 17/17 existing MCP-v4 control tests passed | removed from the default workflow | manually created external automation is out of scope |

## Failures Encountered During Validation

These were correctly treated as command evidence, not experiment results:

- an operator-copied wrong expected SHA was rejected as
  `GIT_IDENTITY_MISMATCH`; the machine-reported SHA was used once;
- cloud GitHub HTTPS hit a low-speed abort before tests; the single correction
  to configured non-interactive SSH was recorded;
- the first cloud matrix ran 11 new tests and exposed one portable-test-fixture
  root cause (8 passed, 3 affected); one fixture repair was made;
- a later exact-ref fetch timed out at 120 seconds, so the GitHub boundary was
  not retried again; exact files were verified through a one-time SHA-pinned
  validation bundle;
- `/usr/bin/rg` was absent in WSL, so no Windows PATH fallback was used.

Conclusion: the frequent deterministic errors from quoting, CRLF/BOM, stdin,
PATH ambiguity, missing markers, identity drift, output growth, and blind retry
are either removed from the supported path or rejected automatically. External
GitHub, SSH, DNS, and network availability cannot be eliminated; they now fail
with finite, explicit state and cannot silently start or duplicate an experiment.

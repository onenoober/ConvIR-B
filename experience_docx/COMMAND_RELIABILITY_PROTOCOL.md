# Command Reliability Protocol

Date: 2026-07-16

Use direct argument arrays for simple Git/path operations. Use the bounded
`convir-ops` tools for their route lifecycle. For an uncovered cloud command,
write a small Bash script and run
`experience_docx/tools/convir_remote_script.sh <script>`; do not build nested
PowerShell -> WSL -> SSH quoting. Use explicit paths and an `*_OK` marker or
status file.

Quoting, CRLF, PATH, stdin and marker failures are `FAILED_COMMAND`, never
experiment evidence. Apply one canonical correction. If the same boundary class
fails again, stop that operation and report both failed forms. A timeout after
launch is unknown state and must be inspected once before any action.

Transport repair cannot change experiment scope or create authorization.

## Current Canonical Corrections

- Invalid: send a PowerShell-unescaped `|` regex through a WSL command string;
  PowerShell splits it into commands. Corrected: use `Select-String` on an
  explicit file list or a direct WSL argument array without nested shells.
- Invalid: run Windows Git against a WSL UNC worktree and hit ownership/path
  translation. Corrected: run `wsl.exe ... git -C /absolute/wsl/path ...`.

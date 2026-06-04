# Agent Instructions

## Cloud And Local Execution

- Default cloud server for this repository: `dehaze1` (`ssh dehaze1`).
- `dehaze1` resolves to `root@connect.bjb1.seetacloud.com` on port `49601` via the SSH key configured in `~/.ssh/config`.
- Highest-priority project rule: the local WSL checkout is for editing and compile/syntax-only checks.
- Do not run tests, smoke tests, training, evaluation, inference, demos, or project runtime commands locally.
- Run all tests, including smoke tests, training, evaluation, and any execution/runtime validation, on `dehaze1` unless the user explicitly overrides this rule for a specific command.
- If runtime validation is needed, sync the code to `dehaze1` and run it there; if the cloud server is unavailable, report that instead of falling back to local execution.

## Project Memory And Evidence Authority

- Treat `experience_docx/` as the repository's authoritative project memory for experiment state, governance, route decisions, and evidence locations.
- Do not use chat history as the authoritative source for experiment results, route status, decisions, commands, or evidence; verify against repository documents first.
- For any experiment, route, training, evaluation, result-summary, or decision task, read `experience_docx/EXPERIMENT_INDEX.md` first.
- After the index, read the relevant `experience_docx/family_summaries/` file when reopening or extending a route family.
- Then open the corresponding route card under `experience_docx/experiment_cards/`, and inspect the matching evidence directory under `experience_docx/experiment_logs/` before making claims or planning follow-up work.
- For branch evidence syncs, read and follow `experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.
- For new or reorganized experiments, read the relevant `experience_docx/` governance docs first, especially `README.md`, `CONVIR_B_EXECUTION_GUIDE.md`, `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `MODEL_EXPERIMENT_START_CHECKLIST.md`, `ROUTE_DESIGN_FRAMEWORK.md`, and `EXPERIMENT_CARD_TEMPLATE.md`.
- When documentation and conversation conflict, prefer `experience_docx/` and current git state; state any uncertainty and cite the file path used.

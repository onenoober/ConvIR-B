#!/usr/bin/env python3
"""Small argv-only transport entrypoint for ConvIR-B control operations."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path


GIT = Path("/usr/bin/git")
SSH = Path("/usr/bin/ssh")
BASH = Path("/bin/bash")
REMOTE_HOST = "convir-4090"
WORKSPACE_ROOT = Path("/home/ubuntu/workspace").resolve()
MAX_SCRIPT_BYTES = 256 * 1024
MAX_CAPTURE_BYTES = 64 * 1024
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SAFE_REMOTE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SAFE_REF = re.compile(r"^refs/(?:heads|remotes)/[A-Za-z0-9][A-Za-z0-9._/-]{0,240}$")
SAFE_BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,240}$")
SAFE_SCRIPT_RELPATH = re.compile(r"^[A-Za-z0-9._/-]+\.sh$")


class ControlError(RuntimeError):
    def __init__(self, message, *, state="REJECTED", failure_class="contract", exit_code=2):
        super().__init__(message)
        self.state = state
        self.failure_class = failure_class
        self.exit_code = exit_code


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ControlError(
            message, state="ARGUMENTS_INVALID", failure_class="contract", exit_code=2
        )


def clipped(raw):
    raw = raw or b""
    truncated = len(raw) > MAX_CAPTURE_BYTES
    return decode(raw[:MAX_CAPTURE_BYTES]), truncated


def result(ok, operation, state, *, failure_class="none", exit_code=0, **fields):
    value = {
        "ok": ok,
        "operation": operation,
        "state": state,
        "failure_class": failure_class,
        "exit_code": exit_code,
        **fields,
    }
    value["marker"] = f"CONVIRCTL_{operation.upper().replace('-', '_')}_{'OK' if ok else 'FAILED'}"
    return value


def run_argv(argv, *, input_bytes=None, timeout=60, env=None):
    try:
        return subprocess.run(
            [str(item) for item in argv],
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise ControlError(
            "command timed out",
            state="COMMAND_STATE_UNKNOWN",
            failure_class="command_infra",
            exit_code=124,
        ) from exc
    except OSError as exc:
        raise ControlError(
            f"command could not start: {exc}",
            state="LOCAL_COMMAND_FAILED",
            failure_class="command_infra",
            exit_code=2,
        ) from exc


def run_argv_limited(argv, *, input_bytes, timeout, capture_limit):
    """Run argv without a shell while draining output into bounded buffers."""
    try:
        process = subprocess.Popen(
            [str(item) for item in argv],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ControlError(
            f"command could not start: {exc}",
            state="LOCAL_COMMAND_FAILED",
            failure_class="command_infra",
            exit_code=2,
        ) from exc

    stdout = bytearray()
    stderr = bytearray()
    thread_errors = []
    store_limit = capture_limit + 1

    def drain(stream, target):
        try:
            while True:
                block = stream.read(8192)
                if not block:
                    break
                remaining = store_limit - len(target)
                if remaining > 0:
                    target.extend(block[:remaining])
        except OSError as exc:
            thread_errors.append(exc)
        finally:
            stream.close()

    def feed():
        try:
            process.stdin.write(input_bytes)
            process.stdin.flush()
        except BrokenPipeError:
            pass
        except OSError as exc:
            thread_errors.append(exc)
        finally:
            process.stdin.close()

    threads = [
        threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True),
        threading.Thread(target=feed, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=5)
        raise ControlError(
            "command timed out",
            state="COMMAND_STATE_UNKNOWN",
            failure_class="command_infra",
            exit_code=124,
        ) from exc
    for thread in threads:
        thread.join(timeout=5)
    if any(thread.is_alive() for thread in threads) or thread_errors:
        raise ControlError(
            "command stream did not close cleanly",
            state="LOCAL_COMMAND_FAILED",
            failure_class="command_infra",
            exit_code=2,
        )
    return subprocess.CompletedProcess(
        [str(item) for item in argv], return_code, bytes(stdout), bytes(stderr)
    )


def require_program(path):
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ControlError(
            f"required program is unavailable: {path}",
            state="LOCAL_PREFLIGHT_FAILED",
            failure_class="command_infra",
        )


def safe_path(value, name, *, must_be_file=False, must_be_repo=False):
    path = Path(value)
    if not path.is_absolute():
        raise ControlError(f"{name} must be absolute")
    path = path.resolve()
    try:
        path.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ControlError(f"{name} must stay under {WORKSPACE_ROOT}") from exc
    if must_be_file and not path.is_file():
        raise ControlError(f"{name} is not a file")
    if must_be_repo:
        require_program(GIT)
        check = run_argv([GIT, "-C", path, "rev-parse", "--git-dir"], timeout=30)
        if check.returncode:
            raise ControlError(f"{name} is not a Git worktree")
    return path


def decode(raw):
    return raw.decode("utf-8", errors="replace")


def git_environment(timeout):
    value = os.environ.copy()
    value.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "never",
            "SSH_ASKPASS": "/bin/false",
            "GIT_SSH_COMMAND": (
                f"/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout={min(timeout, 30)}"
            ),
        }
    )
    return value


def git_output(repo, *args, timeout=60):
    completed = run_argv(
        [GIT, "-C", repo, *args], timeout=timeout, env=git_environment(timeout)
    )
    stdout = decode(completed.stdout).strip()
    stderr = decode(completed.stderr).strip()
    if completed.returncode:
        raise ControlError(
            stderr or f"git {' '.join(args)} failed",
            state="GIT_COMMAND_FAILED",
            failure_class="command_infra",
            exit_code=completed.returncode,
        )
    return stdout


def git_bytes(repo, *args, timeout=60):
    completed = run_argv(
        [GIT, "-C", repo, *args], timeout=timeout, env=git_environment(timeout)
    )
    if completed.returncode:
        stderr = decode(completed.stderr).strip()
        raise ControlError(
            stderr or f"git {' '.join(args)} failed",
            state="GIT_COMMAND_FAILED",
            failure_class="command_infra",
            exit_code=completed.returncode,
        )
    return completed.stdout


def command_git_state(args):
    repo = safe_path(args.repo, "repo", must_be_repo=True)
    if args.require_remote_match and (not args.remote or not args.ref):
        raise ControlError("require-remote-match needs remote and ref")
    head = git_output(repo, "rev-parse", "HEAD")
    branch = git_output(repo, "branch", "--show-current") or None
    changed = git_output(repo, "status", "--porcelain").splitlines()
    remote_head = None
    if args.remote or args.ref:
        if not args.remote or not args.ref:
            raise ControlError("remote and ref must be supplied together")
        fields = git_output(repo, "ls-remote", args.remote, args.ref, timeout=args.timeout_seconds).split()
        if len(fields) != 2 or not SHA40.fullmatch(fields[0]) or fields[1] != args.ref:
            raise ControlError(
                "remote ref is missing or malformed",
                state="GITHUB_REF_INVALID",
                failure_class="command_infra",
            )
        remote_head = fields[0]
    mismatches = []
    if args.expected_head and head != args.expected_head:
        mismatches.append("head")
    if args.expected_branch and branch != args.expected_branch:
        mismatches.append("branch")
    if args.require_clean and changed:
        mismatches.append("worktree_clean")
    if args.require_remote_match and remote_head != head:
        mismatches.append("remote_head")
    if mismatches:
        return result(
            False, "git-state", "GIT_IDENTITY_MISMATCH",
            failure_class="identity", exit_code=3, repo=str(repo), head=head,
            branch=branch, clean=not changed, changed_paths=changed[:100],
            changed_paths_truncated=len(changed) > 100, remote_head=remote_head,
            mismatches=mismatches,
        )
    return result(
        True, "git-state", "GIT_STATE_OK", repo=str(repo), head=head,
        branch=branch, clean=not changed, changed_paths=changed[:100],
        changed_paths_truncated=len(changed) > 100, remote_head=remote_head,
        mismatches=[],
    )


def command_sha256(args):
    path = safe_path(args.file, "file", must_be_file=True)
    digest_builder = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    if args.expected and digest != args.expected:
        return result(
            False, "sha256", "SHA256_MISMATCH", failure_class="identity",
            exit_code=3, file=str(path), sha256=digest, expected=args.expected,
        )
    return result(True, "sha256", "SHA256_OK", file=str(path), sha256=digest)


def normalize_script(raw):
    if len(raw) > MAX_SCRIPT_BYTES:
        raise ControlError("script exceeds 256 KiB")
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if not raw.strip():
        raise ControlError("script is empty")
    if b"\x00" in raw:
        raise ControlError("script contains a NUL byte")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ControlError("script must be valid UTF-8") from exc
    executable_lines = [
        line.strip() for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not executable_lines or executable_lines[0] != "set -euo pipefail":
        raise ControlError("script must start with set -euo pipefail after comments/shebang")
    return raw


def committed_script_identity(script):
    if script.suffix != ".sh":
        raise ControlError("remote script must use the .sh suffix")
    repo_text = git_output(script.parent, "rev-parse", "--show-toplevel")
    repo = safe_path(repo_text, "script repository", must_be_repo=True)
    relative = script.relative_to(repo).as_posix()
    if not SAFE_SCRIPT_RELPATH.fullmatch(relative):
        raise ControlError(
            "remote script path must use only letters, digits, dot, underscore, slash, and hyphen"
        )
    tracked = run_argv(
        [GIT, "-C", repo, "ls-files", "--error-unmatch", "--", relative],
        timeout=30,
    )
    if tracked.returncode:
        raise ControlError("remote script must be tracked by Git")
    working_blob = git_output(repo, "hash-object", "--", relative)
    committed_blob = git_output(repo, "rev-parse", f"HEAD:{relative}")
    if working_blob != committed_blob:
        raise ControlError("remote script must match the committed Git blob")
    blob_size_text = git_output(repo, "cat-file", "-s", f"HEAD:{relative}")
    try:
        blob_size = int(blob_size_text)
    except ValueError as exc:
        raise ControlError("remote script Git blob size is invalid") from exc
    if blob_size > MAX_SCRIPT_BYTES:
        raise ControlError("script exceeds 256 KiB")
    committed_raw = git_bytes(repo, "show", f"HEAD:{relative}")
    identity = {
        "repo": str(repo),
        "relative_path": relative,
        "repo_head": git_output(repo, "rev-parse", "HEAD"),
        "git_blob": committed_blob,
    }
    return identity, committed_raw


def command_remote_script(args):
    script = safe_path(args.script, "script", must_be_file=True)
    require_program(BASH)
    require_program(SSH)
    identity, committed_raw = committed_script_identity(script)
    raw = normalize_script(committed_raw)
    syntax = run_argv([BASH, "-n"], input_bytes=raw, timeout=30)
    if syntax.returncode:
        stderr, truncated = clipped(syntax.stderr)
        return result(
            False, "remote-script", "SCRIPT_SYNTAX_INVALID",
            failure_class="contract", exit_code=syntax.returncode,
            script=str(script), script_identity=identity, stderr=stderr,
            output_truncated=truncated,
        )
    try:
        completed = run_argv_limited(
            [
                SSH, "-T", "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={min(args.timeout_seconds, 30)}",
                REMOTE_HOST, "/bin/bash", "-s", "--",
            ],
            input_bytes=raw, timeout=args.timeout_seconds,
            capture_limit=MAX_CAPTURE_BYTES,
        )
    except ControlError as exc:
        if exc.state == "COMMAND_STATE_UNKNOWN":
            return result(
                False, "remote-script", "REMOTE_STATE_UNKNOWN",
                failure_class="command_infra", exit_code=124,
                script=str(script), script_identity=identity,
                script_sha256=hashlib.sha256(raw).hexdigest(),
                allowed_next_action="inspect_once", blind_retry_allowed=False,
            )
        raise
    stdout, stdout_truncated = clipped(completed.stdout)
    stderr, stderr_truncated = clipped(completed.stderr)
    if completed.returncode:
        state = "SSH_TRANSPORT_FAILED" if completed.returncode == 255 else "REMOTE_SCRIPT_FAILED"
        return result(
            False, "remote-script", state,
            failure_class="command_infra", exit_code=completed.returncode,
            script=str(script), script_identity=identity,
            script_sha256=hashlib.sha256(raw).hexdigest(),
            stdout=stdout, stderr=stderr,
            output_truncated=stdout_truncated or stderr_truncated,
        )
    return result(
        True, "remote-script", "REMOTE_SCRIPT_OK", script=str(script),
        script_identity=identity, script_sha256=hashlib.sha256(raw).hexdigest(),
        stdout=stdout, stderr=stderr,
        output_truncated=stdout_truncated or stderr_truncated,
    )


def parser():
    root = JsonArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="operation", required=True)

    git_state = commands.add_parser("git-state")
    git_state.add_argument("--repo", required=True)
    git_state.add_argument("--expected-head", type=lambda value: require_sha(value))
    git_state.add_argument("--expected-branch", type=require_branch)
    git_state.add_argument("--require-clean", action="store_true")
    git_state.add_argument("--remote", type=require_remote)
    git_state.add_argument("--ref", type=require_ref)
    git_state.add_argument("--require-remote-match", action="store_true")
    git_state.add_argument("--timeout-seconds", type=bounded_timeout, default=60)
    git_state.set_defaults(handler=command_git_state)

    sha = commands.add_parser("sha256")
    sha.add_argument("--file", required=True)
    sha.add_argument("--expected", type=lambda value: require_digest(value))
    sha.set_defaults(handler=command_sha256)

    remote = commands.add_parser("remote-script")
    remote.add_argument("--script", required=True)
    remote.add_argument("--timeout-seconds", type=bounded_timeout, default=600)
    remote.set_defaults(handler=command_remote_script)
    return root


def require_sha(value):
    if not SHA40.fullmatch(value):
        raise argparse.ArgumentTypeError("expected-head must be 40 lowercase hex")
    return value


def require_digest(value):
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("expected must be 64 lowercase hex")
    return value


def unsafe_git_name(value):
    return (
        ".." in value
        or "//" in value
        or "@{" in value
        or value.endswith(("/", "."))
        or any(part.startswith(".") for part in value.split("/"))
        or any(part.endswith(".lock") for part in value.split("/"))
    )


def require_branch(value):
    if not SAFE_BRANCH.fullmatch(value) or unsafe_git_name(value):
        raise argparse.ArgumentTypeError("expected-branch is not a safe branch name")
    return value


def require_remote(value):
    if not SAFE_REMOTE.fullmatch(value):
        raise argparse.ArgumentTypeError("remote is not a safe Git remote name")
    return value


def require_ref(value):
    if not SAFE_REF.fullmatch(value) or unsafe_git_name(value[5:]):
        raise argparse.ArgumentTypeError("ref is not a safe heads/remotes ref")
    return value


def bounded_timeout(value):
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be an integer") from exc
    if not 1 <= value <= 3600:
        raise argparse.ArgumentTypeError("timeout must be in [1, 3600]")
    return value


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    operation = (
        raw_args[0]
        if raw_args and raw_args[0] in {"git-state", "sha256", "remote-script"}
        else "arguments"
    )
    try:
        args = parser().parse_args(raw_args)
        operation = args.operation
        value = args.handler(args)
    except ControlError as exc:
        value = result(
            False, operation, exc.state, failure_class=exc.failure_class,
            exit_code=exc.exit_code, error=str(exc),
        )
    except OSError as exc:
        value = result(
            False, operation, "LOCAL_IO_FAILED", failure_class="command_infra",
            exit_code=2, error=str(exc),
        )
    except Exception as exc:  # Keep the control-plane result machine-readable.
        value = result(
            False, operation, "INTERNAL_CONTROL_ERROR", failure_class="internal",
            exit_code=70, error=f"{type(exc).__name__}: {exc}",
        )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return value["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

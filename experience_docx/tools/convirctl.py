#!/usr/bin/env python3
"""Small argv-only transport entrypoint for ConvIR-B control operations."""

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path, PurePosixPath


GIT = Path("/usr/bin/git")
SSH = Path("/usr/bin/ssh")
BASH = Path("/bin/bash")
REMOTE_HOST = "convir-4090"
WORKSPACE_ROOT = Path("/home/ubuntu/workspace").resolve()
MAX_SCRIPT_BYTES = 256 * 1024
MAX_CAPTURE_BYTES = 64 * 1024
MAX_REPO_TEXT_BYTES = 256 * 1024
MAX_REPO_RESULTS = 1000
MAX_REPO_BLOB_BYTES = 16 * 1024 * 1024
MAX_REPO_ENUMERATION_BYTES = 16 * 1024 * 1024
MAX_REPO_RESPONSE_BYTES = 32 * 1024
MAX_REPO_PAGE_DATA_BYTES = 12 * 1024
DEFAULT_REPO_PAGE_BYTES = 16 * 1024
MAX_REPO_PAGE_BYTES = 64 * 1024
MAX_SEARCH_EXCERPT_BYTES = 1024
MAX_REPO_CURSOR_CHARS = 2048
REPO_CURSOR_VERSION = 1
REPO_CURSOR_DOMAIN = b"convir-repo-page-v1\0"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_CURSOR = re.compile(r"^[A-Za-z0-9_-]+$")
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


def run_argv_limited(argv, *, input_bytes, timeout, capture_limit, env=None):
    """Run argv without a shell while draining output into bounded buffers."""
    try:
        process = subprocess.Popen(
            [str(item) for item in argv],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
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


def safe_repo_relpath(value, name="path"):
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ControlError(f"{name} must be a non-empty single-line repository path")
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ControlError(f"{name} must be a safe relative repository path")
    return candidate.as_posix()


def require_revision(value):
    if value == "HEAD" or SHA40.fullmatch(value):
        return value
    if SAFE_BRANCH.fullmatch(value) and not unsafe_git_name(value):
        return value
    if SAFE_REF.fullmatch(value) and not unsafe_git_name(value[5:]):
        return value
    raise argparse.ArgumentTypeError("ref is not a safe commit, branch, or heads/remotes ref")


def resolved_commit(repo, revision):
    commit = git_output(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if not SHA40.fullmatch(commit):
        raise ControlError("ref did not resolve to one commit", state="GIT_REF_INVALID")
    return commit


def bounded_count(value):
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-results must be an integer") from exc
    if not 1 <= value <= MAX_REPO_RESULTS:
        raise argparse.ArgumentTypeError(f"max-results must be in [1, {MAX_REPO_RESULTS}]")
    return value


def bounded_page_bytes(value):
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("page-bytes must be an integer") from exc
    if not 4 <= value <= MAX_REPO_PAGE_BYTES:
        raise argparse.ArgumentTypeError(
            f"page-bytes must be in [4, {MAX_REPO_PAGE_BYTES}]"
        )
    return value


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def serialized_json_bytes(value):
    return len(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def encode_repo_cursor(operation, commit, query_sha256, position, object_id=None):
    payload = {
        "version": REPO_CURSOR_VERSION,
        "operation": operation,
        "commit": commit,
        "query_sha256": query_sha256,
        "position": position,
        "object_id": object_id,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    envelope = {
        **payload,
        "checksum": hashlib.sha256(REPO_CURSOR_DOMAIN + canonical).hexdigest(),
    }
    token = base64.urlsafe_b64encode(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    if len(token) > MAX_REPO_CURSOR_CHARS:
        raise ControlError(
            "repository cursor exceeds its size contract",
            state="REPO_CURSOR_INVALID",
        )
    return token


def decode_repo_cursor(token, operation, query_sha256):
    if not isinstance(token, str) or not 1 <= len(token) <= MAX_REPO_CURSOR_CHARS \
            or not SAFE_CURSOR.fullmatch(token):
        raise ControlError("repository cursor is malformed", state="REPO_CURSOR_INVALID")
    try:
        padded = token + "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(padded)
        canonical_token = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
        if canonical_token != token:
            raise ValueError("cursor is not canonical base64url")
        envelope = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error) as exc:
        raise ControlError(
            "repository cursor is malformed", state="REPO_CURSOR_INVALID"
        ) from exc
    expected = {
        "version", "operation", "commit", "query_sha256", "position",
        "object_id", "checksum",
    }
    if not isinstance(envelope, dict) or set(envelope) != expected:
        raise ControlError(
            "repository cursor has an invalid field contract",
            state="REPO_CURSOR_INVALID",
        )
    payload = {key: envelope[key] for key in expected - {"checksum"}}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    checksum = hashlib.sha256(REPO_CURSOR_DOMAIN + canonical).hexdigest()
    if not isinstance(envelope["checksum"], str) or not hmac.compare_digest(
        envelope["checksum"], checksum
    ):
        raise ControlError(
            "repository cursor checksum mismatch", state="REPO_CURSOR_INVALID"
        )
    object_id = envelope["object_id"]
    if not isinstance(envelope["version"], int) \
            or isinstance(envelope["version"], bool) \
            or envelope["version"] != REPO_CURSOR_VERSION \
            or not isinstance(envelope["operation"], str) \
            or not isinstance(envelope["commit"], str) \
            or not SHA40.fullmatch(envelope["commit"]) \
            or not isinstance(envelope["query_sha256"], str) \
            or not SHA256.fullmatch(envelope["query_sha256"]) \
            or not isinstance(envelope["position"], int) \
            or isinstance(envelope["position"], bool) \
            or envelope["position"] < 0 \
            or (object_id is not None and (
                not isinstance(object_id, str) or not SHA40.fullmatch(object_id)
            )):
        raise ControlError(
            "repository cursor values are invalid", state="REPO_CURSOR_INVALID"
        )
    if envelope["operation"] != operation \
            or envelope["query_sha256"] != query_sha256:
        raise ControlError(
            "repository cursor does not match this operation and query",
            state="REPO_CURSOR_IDENTITY_MISMATCH",
            failure_class="identity",
            exit_code=3,
        )
    return envelope


def repo_page_identity(repo, args, operation, query_sha256):
    ref_commit = resolved_commit(repo, args.ref)
    token = getattr(args, "cursor", None)
    if token is None:
        return ref_commit, ref_commit, 0, None, False
    cursor = decode_repo_cursor(token, operation, query_sha256)
    commit = cursor["commit"]
    resolved_commit(repo, commit)
    if SHA40.fullmatch(args.ref) and ref_commit != commit:
        raise ControlError(
            "explicit ref differs from the repository cursor snapshot",
            state="REPO_CURSOR_IDENTITY_MISMATCH",
            failure_class="identity",
            exit_code=3,
        )
    return commit, ref_commit, cursor["position"], cursor["object_id"], ref_commit != commit


def strict_utf8(raw, name):
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ControlError(
            f"{name} is not UTF-8 text", state="REPO_TEXT_NOT_UTF8",
            failure_class="contract", exit_code=3,
        ) from exc


def utf8_prefix(raw, maximum_bytes):
    candidate = raw[:maximum_bytes]
    while candidate:
        try:
            return candidate, candidate.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data":
                raise ControlError(
                    "repository text is not UTF-8", state="REPO_TEXT_NOT_UTF8",
                    failure_class="contract", exit_code=3,
                ) from exc
            candidate = candidate[:exc.start]
    return b"", ""


def limited_git_bytes(repo, argv, operation):
    completed = run_argv_limited(
        [GIT, "-C", repo, *argv], input_bytes=b"", timeout=60,
        capture_limit=MAX_REPO_ENUMERATION_BYTES, env=git_environment(60),
    )
    if completed.returncode not in {0, 1}:
        raise ControlError(
            decode(completed.stderr).strip() or f"git {operation} failed",
            state="GIT_COMMAND_FAILED", failure_class="command_infra",
            exit_code=completed.returncode,
        )
    if len(completed.stdout) > MAX_REPO_ENUMERATION_BYTES:
        raise ControlError(
            "repository enumeration exceeds its internal safety bound",
            state="REPO_ENUMERATION_TOO_LARGE", failure_class="contract",
            exit_code=3,
        )
    return completed


def bounded_string_page(items, start, maximum_results):
    if start > len(items) or (start == len(items) and start != 0):
        raise ControlError(
            "repository cursor position is outside the result set",
            state="REPO_CURSOR_INVALID",
        )
    selected = []
    used = 2
    for item in items[start:start + maximum_results]:
        item_size = serialized_json_bytes(item) + int(bool(selected))
        if selected and used + item_size > MAX_REPO_PAGE_DATA_BYTES:
            break
        if not selected and used + item_size > MAX_REPO_PAGE_DATA_BYTES:
            raise ControlError(
                "one repository record exceeds the response budget",
                state="REPO_RECORD_TOO_LARGE", failure_class="contract",
                exit_code=3,
            )
        selected.append(item)
        used += item_size
    return selected


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


def command_task_context(args):
    repo = safe_path(args.repo, "repo", must_be_repo=True)
    cwd = safe_path(args.cwd, "cwd")
    if not cwd.is_dir():
        raise ControlError("cwd is not a directory")
    head = git_output(repo, "rev-parse", "HEAD")
    branch = git_output(repo, "branch", "--show-current") or None
    changed = git_output(repo, "status", "--porcelain").splitlines()
    cwd_check = run_argv([GIT, "-C", cwd, "rev-parse", "--show-toplevel"], timeout=30)
    cwd_repo_root = None
    if cwd_check.returncode == 0:
        candidate = Path(decode(cwd_check.stdout).strip()).resolve()
        cwd_repo_root = str(candidate)
    cwd_matches_repo = cwd_repo_root == str(repo)
    tracking_ref = "refs/remotes/github/main"
    tracking = run_argv([GIT, "-C", repo, "rev-parse", "--verify", tracking_ref], timeout=30)
    github_main = decode(tracking.stdout).strip() if tracking.returncode == 0 else None
    if github_main is not None and not SHA40.fullmatch(github_main):
        github_main = None
    fields = {
        "repo": str(repo), "cwd": str(cwd), "cwd_repo_root": cwd_repo_root,
        "cwd_matches_repo": cwd_matches_repo, "write_allowed": cwd_matches_repo,
        "required_action": "none" if cwd_matches_repo else "bind_all_writes_to_local_repo_or_handoff",
        "head": head, "branch": branch, "clean": not changed,
        "changed_paths": changed[:100], "changed_paths_truncated": len(changed) > 100,
        "github_main": github_main, "github_main_source": tracking_ref if github_main else None,
        "control_source": str(Path(__file__).resolve()),
    }
    if not cwd_matches_repo:
        return result(
            False, "task-context", "TASK_CONTEXT_MISMATCH", failure_class="identity",
            exit_code=3, **fields,
        )
    return result(True, "task-context", "TASK_CONTEXT_OK", **fields)


def command_repo_show(args):
    repo = safe_path(args.repo, "repo", must_be_repo=True)
    relative = safe_repo_relpath(args.path)
    query_sha256 = canonical_digest({"path": relative})
    commit, ref_commit, start, cursor_object, snapshot_drifted = repo_page_identity(
        repo, args, "repo-show", query_sha256,
    )
    object_id = git_output(repo, "rev-parse", f"{commit}:{relative}")
    if not SHA40.fullmatch(object_id):
        raise ControlError("repository path did not resolve to one blob")
    if cursor_object is not None and cursor_object != object_id:
        raise ControlError(
            "repository cursor blob identity mismatch",
            state="REPO_CURSOR_IDENTITY_MISMATCH", failure_class="identity",
            exit_code=3,
        )
    if git_output(repo, "cat-file", "-t", object_id) != "blob":
        raise ControlError(
            "repository path is not a blob", state="REPO_PATH_NOT_BLOB",
            failure_class="contract", exit_code=3,
        )
    try:
        size_bytes = int(git_output(repo, "cat-file", "-s", object_id))
    except ValueError as exc:
        raise ControlError("repository blob size is invalid") from exc
    if size_bytes > MAX_REPO_BLOB_BYTES:
        return result(
            False, "repo-show", "REPO_TEXT_TOO_LARGE", failure_class="contract",
            exit_code=3, repo=str(repo), ref=args.ref, commit=commit,
            ref_commit=ref_commit, snapshot_drifted=snapshot_drifted,
            path=relative, blob_oid=object_id, size_bytes=size_bytes,
            max_bytes=MAX_REPO_BLOB_BYTES, pagination_available=False,
        )
    raw = git_bytes(repo, "show", f"{commit}:{relative}")
    if len(raw) != size_bytes:
        raise ControlError(
            "repository blob size changed during the read",
            state="REPO_BLOB_IDENTITY_MISMATCH", failure_class="identity",
            exit_code=3,
        )
    strict_utf8(raw, "repository blob")
    if start > size_bytes or (getattr(args, "cursor", None) is not None and start == size_bytes):
        raise ControlError(
            "repository cursor position is outside the blob",
            state="REPO_CURSOR_INVALID",
        )
    if start < size_bytes and raw[start] & 0xC0 == 0x80:
        raise ControlError(
            "repository cursor position splits a UTF-8 character",
            state="REPO_CURSOR_INVALID",
        )
    content_sha256 = hashlib.sha256(raw).hexdigest()
    page_limit = min(
        getattr(args, "page_bytes", DEFAULT_REPO_PAGE_BYTES),
        max(4, size_bytes - start),
    )
    while True:
        page_raw, content = utf8_prefix(raw[start:], page_limit)
        if start < size_bytes and not page_raw:
            raise ControlError(
                "repository page could not contain one UTF-8 character",
                state="REPO_RESPONSE_TOO_LARGE", failure_class="contract",
                exit_code=3,
            )
        end = start + len(page_raw)
        complete = end == size_bytes
        next_cursor = None if complete else encode_repo_cursor(
            "repo-show", commit, query_sha256, end, object_id,
        )
        value = result(
            True, "repo-show", "REPO_SHOW_OK", repo=str(repo), ref=args.ref,
            schema_version=1, query_sha256=query_sha256,
            commit=commit, snapshot_commit=commit, ref_commit=ref_commit,
            snapshot_drifted=snapshot_drifted, path=relative,
            blob_oid=object_id, content_sha256=content_sha256,
            collection_sha256=content_sha256,
            size_bytes=size_bytes, page_start_byte=start, page_end_byte=end,
            page_bytes=len(page_raw), page_sha256=hashlib.sha256(page_raw).hexdigest(),
            content=content, complete=complete, page_complete=True,
            terminal_page=complete, has_more=not complete, next_cursor=next_cursor,
            truncated=not complete, capture_truncated=False,
            scientific_completeness="not_assessed",
        )
        if serialized_json_bytes(value) <= MAX_REPO_RESPONSE_BYTES:
            return value
        if page_limit <= 4:
            raise ControlError(
                "repository response metadata exceeds the response budget",
                state="REPO_RESPONSE_TOO_LARGE", failure_class="contract",
                exit_code=3,
            )
        page_limit = max(4, page_limit // 2)


def command_repo_list(args):
    repo = safe_path(args.repo, "repo", must_be_repo=True)
    paths = [safe_repo_relpath(item) for item in args.path]
    query_sha256 = canonical_digest({"paths": paths})
    commit, ref_commit, start, cursor_object, snapshot_drifted = repo_page_identity(
        repo, args, "repo-list", query_sha256,
    )
    if cursor_object is not None:
        raise ControlError(
            "repository list cursor has an unexpected object identity",
            state="REPO_CURSOR_INVALID",
        )
    argv = ["--literal-pathspecs", "ls-tree", "-r", "-z", "--name-only", commit]
    if paths:
        argv.extend(["--", *paths])
    completed = limited_git_bytes(repo, argv, "ls-tree")
    if completed.returncode:
        raise ControlError(
            decode(completed.stderr).strip() or "git ls-tree failed",
            state="GIT_COMMAND_FAILED", failure_class="command_infra",
            exit_code=completed.returncode,
        )
    if completed.stdout and not completed.stdout.endswith(b"\0"):
        raise ControlError("git ls-tree returned an incomplete record")
    raw_paths = completed.stdout.split(b"\0")[:-1] if completed.stdout else []
    all_paths = [strict_utf8(item, "repository path") for item in raw_paths]
    selected = bounded_string_page(all_paths, start, args.max_results)
    end = start + len(selected)
    complete = end == len(all_paths)
    next_cursor = None if complete else encode_repo_cursor(
        "repo-list", commit, query_sha256, end,
    )
    value = result(
        True, "repo-list", "REPO_LIST_OK", repo=str(repo), ref=args.ref,
        schema_version=1, query_sha256=query_sha256,
        commit=commit, snapshot_commit=commit, ref_commit=ref_commit,
        snapshot_drifted=snapshot_drifted, path_filters=paths, paths=selected,
        page_start=start, page_count=len(selected), total_count=len(all_paths),
        result_count=len(selected), collection_sha256=canonical_digest(all_paths),
        page_sha256=canonical_digest(selected), complete=complete,
        page_complete=True, terminal_page=complete, has_more=not complete,
        next_cursor=next_cursor,
        truncated=not complete, capture_truncated=False,
        scientific_completeness="not_assessed",
    )
    if serialized_json_bytes(value) > MAX_REPO_RESPONSE_BYTES:
        raise ControlError(
            "repository list response exceeds the response budget",
            state="REPO_RESPONSE_TOO_LARGE", failure_class="contract",
            exit_code=3,
        )
    return value


def command_repo_search(args):
    repo = safe_path(args.repo, "repo", must_be_repo=True)
    paths = [safe_repo_relpath(item) for item in args.path]
    terms = []
    for term in args.term:
        if not term or len(term) > 256 or any(item in term for item in ("\x00", "\n", "\r")):
            raise ControlError("each search term must be 1-256 characters on one line")
        terms.append(term)
    if len(terms) > 8:
        raise ControlError("repo-search accepts at most 8 literal terms")
    query_sha256 = canonical_digest({"paths": paths, "terms": terms})
    commit, ref_commit, start, cursor_object, snapshot_drifted = repo_page_identity(
        repo, args, "repo-search", query_sha256,
    )
    if cursor_object is not None:
        raise ControlError(
            "repository search cursor has an unexpected object identity",
            state="REPO_CURSOR_INVALID",
        )
    argv = ["--literal-pathspecs", "grep", "-n", "-I", "-F", "-z"]
    for term in terms:
        argv.extend(["-e", term])
    argv.append(commit)
    if paths:
        argv.extend(["--", *paths])
    completed = limited_git_bytes(repo, argv, "grep")
    records = []
    position = 0
    commit_prefix = f"{commit}:".encode("ascii")
    while position < len(completed.stdout):
        path_end = completed.stdout.find(b"\0", position)
        line_end = completed.stdout.find(b"\0", path_end + 1)
        content_end = completed.stdout.find(b"\n", line_end + 1)
        if path_end < 0 or line_end < 0:
            raise ControlError("git grep returned an incomplete record")
        if content_end < 0:
            content_end = len(completed.stdout)
        raw_header = completed.stdout[position:path_end]
        raw_line = completed.stdout[path_end + 1:line_end]
        raw_content = completed.stdout[line_end + 1:content_end]
        if not raw_header.startswith(commit_prefix) or not raw_line.isdigit():
            raise ControlError("git grep returned an invalid record")
        path = strict_utf8(raw_header[len(commit_prefix):], "repository path")
        content = strict_utf8(raw_content, "repository search line")
        excerpt_raw, excerpt = utf8_prefix(raw_content, MAX_SEARCH_EXCERPT_BYTES)
        line_number = int(raw_line)
        records.append({
            "path": path,
            "line": line_number,
            "excerpt": excerpt,
            "line_bytes": len(raw_content),
            "line_sha256": hashlib.sha256(raw_content).hexdigest(),
            "line_truncated": len(excerpt_raw) < len(raw_content),
            "display": f"{commit}:{path}:{line_number}:{excerpt}",
        })
        position = content_end + int(content_end < len(completed.stdout))
    if start > len(records) or (start == len(records) and start != 0):
        raise ControlError(
            "repository cursor position is outside the result set",
            state="REPO_CURSOR_INVALID",
        )
    selected = []
    used = 2
    for record in records[start:start + args.max_results]:
        public_record = {key: value for key, value in record.items() if key != "display"}
        item_size = serialized_json_bytes(record["display"]) \
            + serialized_json_bytes(public_record) + 2
        if selected and used + item_size > MAX_REPO_PAGE_DATA_BYTES:
            break
        if not selected and used + item_size > MAX_REPO_PAGE_DATA_BYTES:
            raise ControlError(
                "one repository search record exceeds the response budget",
                state="REPO_RECORD_TOO_LARGE", failure_class="contract",
                exit_code=3,
            )
        selected.append(record)
        used += item_size
    end = start + len(selected)
    complete = end == len(records)
    next_cursor = None if complete else encode_repo_cursor(
        "repo-search", commit, query_sha256, end,
    )
    matches = [item["display"] for item in selected]
    match_records = [
        {key: value for key, value in item.items() if key != "display"}
        for item in selected
    ]
    collection_identity = [
        {
            "path": item["path"], "line": item["line"],
            "line_bytes": item["line_bytes"], "line_sha256": item["line_sha256"],
        }
        for item in records
    ]
    value = result(
        True, "repo-search", "REPO_SEARCH_OK", repo=str(repo), ref=args.ref,
        schema_version=1, query_sha256=query_sha256,
        commit=commit, snapshot_commit=commit, ref_commit=ref_commit,
        snapshot_drifted=snapshot_drifted, terms=terms, path_filters=paths,
        matches=matches, match_records=match_records,
        page_start=start, page_count=len(selected), total_count=len(records),
        result_count=len(selected), zero_matches=not records,
        collection_sha256=canonical_digest(collection_identity),
        page_sha256=canonical_digest(match_records), complete=complete,
        page_complete=True, terminal_page=complete, has_more=not complete,
        next_cursor=next_cursor,
        truncated=not complete, capture_truncated=False,
        scientific_completeness="not_assessed",
    )
    if serialized_json_bytes(value) > MAX_REPO_RESPONSE_BYTES:
        raise ControlError(
            "repository search response exceeds the response budget",
            state="REPO_RESPONSE_TOO_LARGE", failure_class="contract",
            exit_code=3,
        )
    return value


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
    changed = git_output(repo, "status", "--porcelain").splitlines()
    if changed:
        raise ControlError("remote script requires a clean Git worktree")
    branch = git_output(repo, "branch", "--show-current")
    if not branch or not SAFE_BRANCH.fullmatch(branch) or unsafe_git_name(branch):
        raise ControlError("remote script requires one safe named branch")
    head = git_output(repo, "rev-parse", "HEAD")
    remote_ref = f"refs/heads/{branch}"
    fields = git_output(repo, "ls-remote", "github", remote_ref).split()
    if len(fields) != 2 or fields[0] != head or fields[1] != remote_ref:
        raise ControlError("remote script HEAD must match its GitHub branch")
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
        "repo_head": head,
        "branch": branch,
        "github_ref": remote_ref,
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

    task_context = commands.add_parser("task-context")
    task_context.add_argument("--repo", required=True)
    task_context.add_argument("--cwd", required=True)
    task_context.set_defaults(handler=command_task_context)

    repo_show = commands.add_parser("repo-show")
    repo_show.add_argument("--repo", required=True)
    repo_show.add_argument("--ref", type=require_revision, default="HEAD")
    repo_show.add_argument("--path", required=True)
    repo_show.add_argument("--cursor")
    repo_show.add_argument(
        "--page-bytes", type=bounded_page_bytes, default=DEFAULT_REPO_PAGE_BYTES,
    )
    repo_show.set_defaults(handler=command_repo_show)

    repo_list = commands.add_parser("repo-list")
    repo_list.add_argument("--repo", required=True)
    repo_list.add_argument("--ref", type=require_revision, default="HEAD")
    repo_list.add_argument("--path", action="append", default=[])
    repo_list.add_argument("--max-results", type=bounded_count, default=200)
    repo_list.add_argument("--cursor")
    repo_list.set_defaults(handler=command_repo_list)

    repo_search = commands.add_parser("repo-search")
    repo_search.add_argument("--repo", required=True)
    repo_search.add_argument("--ref", type=require_revision, default="HEAD")
    repo_search.add_argument("--term", action="append", required=True)
    repo_search.add_argument("--path", action="append", default=[])
    repo_search.add_argument("--max-results", type=bounded_count, default=200)
    repo_search.add_argument("--cursor")
    repo_search.set_defaults(handler=command_repo_search)

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
        if raw_args and raw_args[0] in {
            "git-state", "task-context", "repo-show", "repo-list", "repo-search",
            "sha256", "remote-script",
        }
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

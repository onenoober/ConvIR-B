#!/usr/bin/env python3
"""Semantic safety audit for the runtime-only telemetry helper."""

from __future__ import annotations

import ast
from pathlib import Path


ALLOWED_IMPORT_ROOTS = {
    "__future__", "argparse", "json", "os", "pathlib", "re", "sys",
    "tempfile", "time",
}
ALLOWED_FROM_IMPORTS = {
    "__future__": {"annotations"},
    "pathlib": {"Path"},
}
FORBIDDEN_IMPORT_ROOTS = {
    "cupy", "multiprocessing", "pynvml", "signal", "subprocess",
    "tensorflow", "torch",
}
FORBIDDEN_CALLS = {
    "os.exec", "os.fork", "os.kill", "os.killpg", "os.popen", "os.system",
    "signal", "subprocess", "multiprocessing", "torch", "tensorflow",
    "cupy", "pynvml",
}
FORBIDDEN_METHODS = {
    "fork", "kill", "killpg", "pause", "popen", "restart", "resume",
    "send_signal", "spawn", "stop", "system", "terminate",
}
FORBIDDEN_RUNTIME_STRINGS = {
    "/dev/nvidia", "CUDA_VISIBLE_DEVICES", "nvidia-smi",
}
READ_METHODS = {"open", "read", "read_bytes", "read_text", "readlines"}
DYNAMIC_CALLS = {"__import__", "compile", "eval", "exec"}


def dotted_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def docstring_nodes(tree: ast.AST) -> set[int]:
    nodes = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list) and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            nodes.add(id(body[0].value))
    return nodes


def is_exact_proc_stat_read(node: ast.Call) -> bool:
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
        and not node.args
        and len(node.keywords) == 1
        and node.keywords[0].arg == "encoding"
        and isinstance(node.keywords[0].value, ast.Constant)
        and node.keywords[0].value.value == "utf-8"
    ):
        return False
    receiver = node.func.value
    return (
        isinstance(receiver, ast.BinOp)
        and isinstance(receiver.op, ast.Div)
        and isinstance(receiver.right, ast.Constant)
        and receiver.right.value == "stat"
        and isinstance(receiver.left, ast.BinOp)
        and isinstance(receiver.left.op, ast.Div)
        and isinstance(receiver.left.left, ast.Call)
        and isinstance(receiver.left.left.func, ast.Name)
        and receiver.left.left.func.id == "Path"
        and len(receiver.left.left.args) == 1
        and isinstance(receiver.left.left.args[0], ast.Constant)
        and receiver.left.left.args[0].value == "/proc"
        and isinstance(receiver.left.right, ast.Call)
        and isinstance(receiver.left.right.func, ast.Name)
        and receiver.left.right.func.id == "str"
        and len(receiver.left.right.args) == 1
        and isinstance(receiver.left.right.args[0], ast.Name)
        and receiver.left.right.args[0].id == "pid"
    )


def bit_or_terms(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return bit_or_terms(node.left) + bit_or_terms(node.right)
    return [node]


def is_exact_status_append_open(node: ast.Call) -> bool:
    if not (
        len(node.args) == 3
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "path"
        and isinstance(node.args[2], ast.Constant)
        and node.args[2].value == 0o640
    ):
        return False
    terms = bit_or_terms(node.args[1])
    names = {dotted_name(term) for term in terms if dotted_name(term)}
    nofollow = [
        term for term in terms
        if isinstance(term, ast.Call) and dotted_name(term.func) == "getattr"
    ]
    valid_nofollow = len(nofollow) == 1 and (
        len(nofollow[0].args) == 3
        and isinstance(nofollow[0].args[0], ast.Name)
        and nofollow[0].args[0].id == "os"
        and isinstance(nofollow[0].args[1], ast.Constant)
        and nofollow[0].args[1].value == "O_NOFOLLOW"
        and isinstance(nofollow[0].args[2], ast.Constant)
        and nofollow[0].args[2].value == 0
    )
    return names == {"os.O_APPEND", "os.O_CREAT", "os.O_WRONLY"} and valid_nofollow


class TelemetryAudit(ast.NodeVisitor):
    def __init__(self, tree: ast.AST):
        self.findings: list[str] = []
        self.function_stack: list[str] = []
        self.docstrings = docstring_nodes(tree)
        self.proc_read_count = 0

    def finding(self, node: ast.AST, message: str) -> None:
        self.findings.append(f"line {getattr(node, 'lineno', 0)}: {message}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            root = item.name.split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                self.finding(node, f"forbidden import {item.name}")
            if item.asname is not None:
                self.finding(node, f"import alias is forbidden for {item.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        allowed_names = ALLOWED_FROM_IMPORTS.get(node.module or "", set())
        if (
            root in FORBIDDEN_IMPORT_ROOTS
            or root not in ALLOWED_IMPORT_ROOTS
            or any(item.name not in allowed_names or item.asname for item in node.names)
        ):
            self.finding(node, f"forbidden import {node.module or ''}")

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        if name:
            leaf = name.rsplit(".", 1)[-1]
            if name in DYNAMIC_CALLS:
                self.finding(node, f"forbidden dynamic call {name}")
            if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_CALLS):
                self.finding(node, f"forbidden call {name}")
            if name.startswith("os.exec") or name.startswith("os.spawn"):
                self.finding(node, f"forbidden call {name}")
            if leaf in FORBIDDEN_METHODS or leaf.startswith(("exec", "spawn")):
                self.finding(node, f"forbidden control method {name}")
            current = self.function_stack[-1] if self.function_stack else "<module>"
            if leaf in READ_METHODS:
                if current == "process_start_ticks" and is_exact_proc_stat_read(node):
                    self.proc_read_count += 1
                elif current == "append_event" and name == "os.open" and is_exact_status_append_open(node):
                    pass
                else:
                    self.finding(node, f"unexpected file/process read {name} in {current}")
            if name == "getattr":
                allowed = (
                    len(node.args) == 3
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in {"O_NOFOLLOW", "parent_pid"}
                )
                if not allowed:
                    self.finding(node, "forbidden dynamic attribute lookup")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if id(node) in self.docstrings or not isinstance(node.value, str):
            return
        for token in FORBIDDEN_RUNTIME_STRINGS:
            if token in node.value:
                self.finding(node, f"forbidden runtime token {token}")


def audit_source(source: str) -> list[str]:
    tree = ast.parse(source)
    visitor = TelemetryAudit(tree)
    visitor.visit(tree)
    process_functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "process_start_ticks"
    ]
    if len(process_functions) != 1:
        visitor.findings.append("process_start_ticks must exist exactly once")
    if visitor.proc_read_count != 1:
        visitor.findings.append("process_start_ticks must contain exactly one /proc/<pid>/stat read")
    return sorted(set(visitor.findings))


def audit_path(path: Path) -> list[str]:
    return audit_source(path.read_text(encoding="utf-8"))

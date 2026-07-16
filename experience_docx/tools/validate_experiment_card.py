#!/usr/bin/env python3
"""Validate the minimal generic route-card contract."""

import argparse
import hashlib
import re
from pathlib import Path


STATUSES = {"DRAFT", "PLANNED", "RUNNING", "STOPPED", "COMPLETED"}
SECTIONS = {
    "Identity": (
        "Route id", "Question", "Rules commit", "Source branch/commit",
        "Route branch", "Locked test/canary policy",
    ),
    "Scientific Contract": (
        "Population and analysis/grouping unit",
        "Intervention or factor contrast and reference",
        "Primary outcome, direction and aggregation",
        "Preferred mechanism and strongest competing explanation",
        "Evidence roles and candidate/freeze point",
        "Primary gate, uncertainty and threshold source",
        "`PASS` authorizes", "`INCONCLUSIVE` authorizes", "`FAIL` stops",
    ),
    "Implementation Contract": (
        "Exact change and disabled mechanisms", "Checkpoint/load/init/freeze contract",
        "Input whitelist and prohibited inputs",
        "Dataset/split/preprocessing/metric identities",
        "Matched baseline and budget",
        "Resource/cost limits or descriptive-only rationale",
        "Runner and required assets",
    ),
    "Operations And Evidence": (
        "First operation", "Expected wall time and monitor profile",
        "Complete-unit resume policy", "Cloud workspace/run/output/status/closeout",
        "Compact Git evidence and cloud-only raw artifacts",
    ),
}
PLACEHOLDER = re.compile(r"<[A-Za-z][A-Za-z0-9_ /|:+.\-]{0,96}>")
UNRESOLVED = re.compile(r"^(?:|TBD|TODO|pending|n/?a)$", re.I)
ROLE_TOKENS = {"engineering_debug", "development_screening", "confirmation", "sealed_final"}
FORBIDDEN = ("dispatcher", "agent execution routing", "initial-authorization",
             "initial authorization json", "validator selfproof")


def split_sections(text):
    matches = list(re.finditer(r"(?m)^## ([^\n]+)$", text))
    return {
        match.group(1): text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        for index, match in enumerate(matches)
    }


def parse_fields(section):
    result = {}
    for line in section.splitlines():
        match = re.match(r"^- ([^:]+):\s*(.*)$", line)
        if match:
            result[match.group(1).strip()] = match.group(2).strip()
    return result


def complete_table_row(section):
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 4 and all(cells[:4]) and not all(re.fullmatch(r"[-: ]+", cell) for cell in cells):
            if cells[0].lower() not in {"operation", "stage"} and not any(PLACEHOLDER.search(cell) for cell in cells[:4]):
                return True
    return False


def validate(path, launch_ready=False):
    text = path.read_text(encoding="utf-8")
    errors = []
    status = re.search(r"(?m)^Status:\s*`?([A-Z_]+)`?\s*$", text)
    if not status or status.group(1) not in STATUSES:
        errors.append("invalid or missing Status")
    elif launch_ready and status.group(1) != "PLANNED":
        errors.append("--launch-ready requires Status: PLANNED")
    mapped = split_sections(text)
    for section, required in SECTIONS.items():
        if section not in mapped:
            errors.append(f"missing section: {section}")
            continue
        values = parse_fields(mapped[section])
        for name in required:
            value = values.get(name)
            if value is None:
                errors.append(f"{section}: missing field '{name}'")
            elif UNRESOLVED.fullmatch(value) or PLACEHOLDER.search(value):
                errors.append(f"{section}: unresolved field '{name}'")
    if PLACEHOLDER.search(text):
        errors.append("unresolved placeholder")
    scientific = mapped.get("Scientific Contract", "")
    if not any(token in scientific for token in ROLE_TOKENS):
        errors.append("scientific contract has no canonical evidence-role token")
    if not complete_table_row(mapped.get("Operations And Evidence", "")):
        errors.append("Operations And Evidence requires one complete operation row")
    lowered = text.lower()
    for phrase in FORBIDDEN:
        if phrase in lowered:
            errors.append(f"retired control pattern: {phrase}")
    return errors, hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("card", type=Path)
    parser.add_argument("--launch-ready", action="store_true")
    args = parser.parse_args()
    errors, digest = validate(args.card, args.launch_ready)
    if errors:
        for error in errors:
            print(f"ROUTE_CARD_CONTRACT_ERROR {error}")
        raise SystemExit(1)
    print(f"ROUTE_CARD_CONTRACT_OK sha256={digest}")


if __name__ == "__main__":
    main()

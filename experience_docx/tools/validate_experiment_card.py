#!/usr/bin/env python3
"""Validate only the compact, generic route-card contract."""

import argparse
import hashlib
import re
from pathlib import Path


STATUSES = {"DRAFT", "PLANNED", "RUNNING", "STOPPED", "COMPLETED"}
LAUNCH_READY = {"PLANNED"}
SECTIONS = (
    "Identity",
    "Scientific Contract",
    "Design And Evidence Roles",
    "Implementation Contract",
    "Stages",
    "Outputs And Closeout",
)
FIELDS = {
    "Identity": (
        "Route id", "Question", "GitHub rules commit and canonical rule-bundle digest",
        "Source branch/commit", "Route branch", "Local editing workspace",
        "Cloud workspace policy", "Cloud run root", "Explicit cloud Python",
        "Locked test/canary policy",
    ),
    "Scientific Contract": (
        "Target population and analysis/grouping unit", "Intervention or factor contrast",
        "Reference", "Primary outcome, direction, and aggregation", "Claim type",
        "Preferred mechanism", "Null and strongest competing explanation",
        "Cheapest observation that separates them",
        "Minimum worthwhile effect or risk limit and independent source",
        "Primary gate and uncertainty estimator", "`PASS` authorizes",
        "`INCONCLUSIVE` authorizes", "`FAIL` stops",
    ),
    "Design And Evidence Roles": (
        "Design", "Experimental assignment/pairing/blocking",
        "Sample/group/fold/seed count and justification", "Multiplicity treatment",
        "Missing/exclusion policy", "Candidate/operator/threshold freeze point",
        "Forbidden continuations/evidence reuse",
    ),
    "Implementation Contract": (
        "Exact change and enabled mechanism", "Explicitly disabled mechanisms",
        "Checkpoint/load/init/freeze contract", "Input whitelist and prohibited inputs",
        "No-op/neutral behavior", "Dataset/split/preprocessing/metric identities",
        "Matched baseline", "Parameter/MAC hard limit, if decision-relevant",
        "Latency/memory hard limit or descriptive-only rationale", "Required asset manifest",
    ),
    "Stages": (
        "First authorized stage", "Integrated smoke checks",
        "Expected phase/wall-time budget", "Heartbeat and monitor profile",
        "Maximum observation windows and escalation condition", "Unit-boundary resume policy",
    ),
    "Outputs And Closeout": (
        "Runner", "Operations manifest", "Status/log/closeout paths",
        "Required retained states and hashes", "Compact GitHub evidence",
        "Cloud-only raw artifacts", "Terminal archive updates",
    ),
}
PLACEHOLDER = re.compile(r"<[A-Za-z][A-Za-z0-9_ /|:+.\-]{0,96}>")
UNRESOLVED = re.compile(r"^(?:|TBD|TODO|pending|n/?a)$", re.I)
ROLE_TOKENS = {"engineering_debug", "development_screening", "confirmation", "sealed_final"}


def sections(text):
    matches = list(re.finditer(r"(?m)^## ([^\n]+)$", text))
    return {
        match.group(1): text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        for index, match in enumerate(matches)
    }


def fields(section):
    result = {}
    for line in section.splitlines():
        match = re.match(r"^- ([^:]+):\s*(.*)$", line)
        if match:
            result[match.group(1).strip()] = match.group(2).strip()
    return result


def find_field(values, prefix):
    return next((value for name, value in values.items() if name.startswith(prefix)), None)


def table_rows(section):
    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and not all(re.fullmatch(r"[-: ]+", cell or " ") for cell in cells):
            rows.append(cells)
    return rows[1:] if rows else []


def validate(path, launch_ready=False):
    text = path.read_text(encoding="utf-8")
    errors = []
    status_match = re.search(r"(?m)^Status:\s*`?([A-Z_]+)`?\s*$", text)
    if not status_match or status_match.group(1) not in STATUSES:
        errors.append("invalid or missing Status")
    elif launch_ready and status_match.group(1) not in LAUNCH_READY:
        errors.append("--launch-ready requires Status: PLANNED")
    mapped = sections(text)
    for section in SECTIONS:
        if section not in mapped:
            errors.append(f"missing section: {section}")
            continue
        values = fields(mapped[section])
        for prefix in FIELDS[section]:
            value = find_field(values, prefix)
            if value is None:
                errors.append(f"{section}: missing field '{prefix}'")
            elif UNRESOLVED.fullmatch(value) or PLACEHOLDER.search(value):
                errors.append(f"{section}: unresolved field '{prefix}'")
    placeholders = sorted(set(PLACEHOLDER.findall(text)))
    if placeholders:
        errors.append("unresolved placeholders: " + ", ".join(placeholders[:5]))
    role_section = mapped.get("Design And Evidence Roles", "")
    if not any(token in role_section for token in ROLE_TOKENS):
        errors.append("evidence-role ledger has no canonical role token")
    role_rows = table_rows(role_section)
    if not any(len(row) >= 4 and all(row[:4]) and not any(PLACEHOLDER.search(cell) for cell in row[:4]) for row in role_rows):
        errors.append("evidence-role ledger requires one complete row")
    stage_rows = table_rows(mapped.get("Stages", ""))
    if not any(len(row) >= 4 and all(row[:4]) and not any(PLACEHOLDER.search(cell) for cell in row[:4]) for row in stage_rows):
        errors.append("Stages requires one complete row")
    if "Agent Execution Routing" in mapped or "dispatcher" in text.lower():
        errors.append("active route cards must not contain dispatcher/model-task routing")
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

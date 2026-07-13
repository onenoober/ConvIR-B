#!/usr/bin/env python3
"""Statically validate launch-critical fields in an experiment route card."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


ALLOWED_STATUSES = {
    "DRAFT",
    "PLANNED",
    "AUTHORIZED",
    "RUNNING",
    "STOPPED",
    "COMPLETED",
}
LAUNCH_READY_STATUSES = {"PLANNED", "AUTHORIZED"}
EVIDENCE_ROLES = {
    "engineering_debug",
    "development_screening",
    "confirmation",
    "sealed_final",
}
DESIGN_TYPES = {
    "paired_ablation",
    "full_factorial",
    "fractional_factorial",
    "feasibility_oracle",
    "screening_confirmation",
    "adaptive",
}
DECISION_PROFILES = {
    "audit_evaluation",
    "feasibility_oracle",
    "paired_single_intervention_training",
    "factorial_screening",
    "adaptive_decision",
    "policy_replay",
}
REQUIRED_SECTIONS = (
    "Scope",
    "Agent Execution Routing",
    "Baseline Contract",
    "Most Valuable Attempt",
    "Hypothesis",
    "Estimand And Risk Attribution",
    "Design And Identifiability",
    "Change",
    "Preflight",
    "Mechanism Metrics",
    "Controls",
    "Evidence-Role Ledger",
    "Fair Run Contract",
    "Gates",
    "Analysis Plan",
    "Decision",
)
REQUIRED_FIELDS = {
    "Scope": (
        "Project",
        "Dataset or task",
        "Primary objective",
        "Main metric",
        "Secondary metrics",
        "Execution environment",
        "GitHub rules commit",
        "GitHub route branch and source commit",
        "Cloud `REMOTE_REPO`",
        "Cloud `RUN_ROOT`",
        "Cloud `EVID_STAGE`",
        "Explicit cloud Python",
    ),
    "Baseline Contract": (
        "Baseline implementation",
        "Baseline checkpoint or initialization",
        "Evaluation entrypoint",
        "Training entrypoint",
        "Dataset and split",
        "Preprocessing and decoding",
        "Metric implementation",
        "Reproduced baseline result",
        "Known reproduction gap",
        "Reference entrypoints that must remain stable",
        "Checkpoint/export/resume contract",
    ),
    "Most Valuable Attempt": (
        "Why this is the highest-value next attempt",
        "Target failure or opportunity",
        "Cheap preflight evidence",
        "Earliest decisive gate",
        "Expected cost or attempt-count saving",
        "What success decides",
        "What failure decides",
        "Why a cheaper diagnostic is not enough",
    ),
    "Hypothesis": (
        "Observed failure",
        "Target mechanism",
        "Null hypothesis",
        "Preferred causal hypothesis",
        "Competing hypothesis or confound",
        "Cheapest observation that separates them",
    ),
    "Estimand And Risk Attribution": (
        "Target population",
        "Analysis unit and grouping unit",
        "Intervention or factor contrast",
        "Reference/direct predecessor",
        "Outcome, direction, and aggregation",
        "Claim type",
        "Identification assumptions and sensitivity limits",
        "Minimum worthwhile effect or risk limit",
        "Equivalence/non-inferiority margin and independent source",
        "Common safety anchor",
        "Inherited-harm estimand",
        "Candidate-total-harm estimand",
        "Intervention-added-harm estimand",
    ),
    "Design And Identifiability": (
        "Design type",
        "Why this is the cheapest design that identifies the estimand",
        "Experimental unit and randomization/pairing",
        "Blocking, exclusion, failure, and missing-cell policy",
        "Formal subgroup definitions and pre-intervention/independent source",
        "Primary comparison family and multiplicity treatment",
        "Fractional-design resolution and alias structure",
        "Negligible-interaction assumptions and targeted de-alias follow-up",
        "Paired seeds/folds/data order/evaluation operators",
        "Natural groups and repeated grouped-split or leave-one-group-out plan",
        "Split/seed uncertainty required for the claim",
        "Uncertainty estimator and dependence/group structure",
        "Sample/group/split/seed count justified by power or target interval width",
        "Fixed-data attainable precision or smallest reliably detectable effect",
    ),
    "Change": (
        "Code branch",
        "Exact code/config change",
        "Enabled mechanisms",
        "Explicitly disabled mechanisms",
        "Parameter/runtime/memory impact expected",
        "Initialization or no-op behavior",
        "Resume policy",
        "Defaults changed",
        "Defaults intentionally preserved",
    ),
    "Evidence-Role Ledger": (
        "Candidate/threshold/operator freeze point",
        "Independent confirmation contract",
        "Nested group-respecting resampling contract",
        "Final sealed-use authorization and one-use policy",
        "Post-sealed rule",
    ),
    "Fair Run Contract": (
        "Training or inference budget",
        "Batch/sample policy",
        "Optimizer",
        "Schedule",
        "Loss weights",
        "Random seed policy",
        "Evaluation cadence",
        "Checkpoint cadence",
        "Hardware/runtime assumptions",
        "Allowed resume behavior",
        "Sample-size policy",
        "Dependency/version assumptions",
        "Selected decision profile",
        "Learned-state retention required",
        "Omitted or specialized stage rationale",
    ),
    "Analysis Plan": (
        "Per-sample or subgroup analysis",
        "Robustness or held-out analysis",
        "Regression analysis",
        "Main-effect/interaction and alias analysis",
        "Group/split/seed uncertainty and sensitivity analysis",
        "Screening-selection versus confirmation analysis",
        "Required docs to update",
        "Required artifacts to retain",
        "Required artifacts to delete or keep external",
        "Evidence package contents",
    ),
}
LEARNED_STATE_FIELDS = (
    "Retained steps/epochs/factor cells and why each is needed",
    "Model/checkpoint state path and hash contract",
    "Optimizer/scheduler state contract",
    "RNG states required and unavailable-state disclosure",
    "Data-order/sampler identity",
    "Config hash, code commit, Python/environment identity, and parent checkpoint",
    "Trace-manifest path and schema",
    "Cloud retention/deletion policy",
    "Compact GitHub evidence",
)
UNRESOLVED_VALUE = re.compile(r"(?i)^(?:tbd|todo|pending|unknown|n/?a|none)$")
PLACEHOLDER = re.compile(r"<[A-Za-z][A-Za-z0-9_ /|:+.\-]{0,96}>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", type=Path, help="filled Markdown route card")
    parser.add_argument(
        "--launch-ready",
        action="store_true",
        help="also require PLANNED or AUTHORIZED status",
    )
    return parser.parse_args()


def section_map(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[match.end() : end]
    return sections


def bullet_fields(section: str) -> dict[str, str]:
    lines = section.splitlines()
    fields: dict[str, str] = {}
    index = 0
    while index < len(lines):
        match = re.match(r"^-\s+(.+?):\s*(.*)$", lines[index])
        if not match:
            index += 1
            continue
        label = match.group(1).strip()
        parts = [match.group(2).strip()]
        cursor = index + 1
        while cursor < len(lines):
            line = lines[cursor]
            if re.match(r"^(?:-|#|\||```)", line) or not line.strip():
                break
            if line.startswith((" ", "\t")):
                parts.append(line.strip())
                cursor += 1
                continue
            break
        fields[label] = " ".join(part for part in parts if part).strip()
        index = max(index + 1, cursor)
    return fields


def find_field(fields: dict[str, str], prefix: str) -> tuple[str, str] | None:
    for label, value in fields.items():
        continuations = (prefix + " ", prefix + " (", prefix + ",")
        if label == prefix or label.startswith(continuations):
            return label, value
    return None


def explicit_value(value: str) -> bool:
    stripped = value.strip().strip("`").strip()
    if not stripped or PLACEHOLDER.search(stripped) or UNRESOLVED_VALUE.fullmatch(stripped):
        return False
    lowered = stripped.lower()
    if lowered.startswith(("n/a:", "none:", "not applicable:")):
        return len(stripped.split(":", 1)[1].strip()) >= 8
    return True


def canonical_token(value: str, allowed: set[str]) -> bool:
    normalized = value.strip().strip("`").lower()
    if normalized in allowed:
        return True
    if normalized.startswith("hybrid:"):
        return len(normalized.split(":", 1)[1].strip()) >= 8
    return False


def table_data_rows(section: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in section.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    return rows[1:] if rows else []


def require_filled_table(section_name: str, section: str, errors: list[str]) -> None:
    rows = table_data_rows(section)
    if not any(all(cell and not PLACEHOLDER.search(cell) for cell in row) for row in rows):
        errors.append(f"{section_name}: requires at least one fully specified table row")


def validate(text: str, launch_ready: bool) -> tuple[list[str], str]:
    errors: list[str] = []
    sections = section_map(text)

    title = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    if not title or not explicit_value(title.group(1)):
        errors.append("document title is missing or unresolved")

    status_match = re.search(r"^Status:\s*`?([A-Z_]+)`?\s*$", text, re.MULTILINE)
    status = status_match.group(1) if status_match else ""
    if status not in ALLOWED_STATUSES:
        errors.append("Status must use an exact template token")
    elif launch_ready and status not in LAUNCH_READY_STATUSES:
        errors.append("--launch-ready requires Status PLANNED or AUTHORIZED")

    for name in REQUIRED_SECTIONS:
        if name not in sections:
            errors.append(f"missing required section: {name}")

    for section_name, prefixes in REQUIRED_FIELDS.items():
        section = sections.get(section_name, "")
        fields = bullet_fields(section)
        for prefix in prefixes:
            found = find_field(fields, prefix)
            if found is None:
                errors.append(f"{section_name}: missing field '{prefix}'")
            elif not explicit_value(found[1]):
                errors.append(f"{section_name}: unresolved field '{found[0]}'")

    placeholder_hits = sorted(set(PLACEHOLDER.findall(text)))
    if placeholder_hits:
        preview = ", ".join(placeholder_hits[:5])
        errors.append(f"unresolved angle-bracket placeholders remain: {preview}")

    scope_fields = bullet_fields(sections.get("Scope", ""))
    rules_field = find_field(scope_fields, "GitHub rules commit")
    if rules_field and not re.search(r"\b[0-9a-f]{40}\b", rules_field[1]):
        errors.append("Scope: GitHub rules commit must contain a full 40-character SHA")

    estimand_fields = bullet_fields(sections.get("Estimand And Risk Attribution", ""))
    claim_field = find_field(estimand_fields, "Claim type")
    if claim_field:
        claim = claim_field[1].strip().strip("`").lower()
        if claim not in {"causal", "associational", "predictive"}:
            errors.append("Estimand And Risk Attribution: Claim type must be causal, associational, or predictive")

    role_section = sections.get("Evidence-Role Ledger", "")
    roles_found = {
        role for role in EVIDENCE_ROLES if re.search(rf"\b{role}\b", role_section)
    }
    if not roles_found:
        errors.append("Evidence-Role Ledger: no canonical evidence-role token found")
    require_filled_table("Evidence-Role Ledger", role_section, errors)
    require_filled_table(
        "Agent Execution Routing", sections.get("Agent Execution Routing", ""), errors
    )
    require_filled_table("Preflight", sections.get("Preflight", ""), errors)
    require_filled_table("Controls", sections.get("Controls", ""), errors)
    require_filled_table("Gates", sections.get("Gates", ""), errors)

    design_fields = bullet_fields(sections.get("Design And Identifiability", ""))
    design_field = find_field(design_fields, "Design type")
    design_value = design_field[1].strip().strip("`").lower() if design_field else ""
    if design_field and not canonical_token(design_value, DESIGN_TYPES):
        errors.append(
            "Design And Identifiability: Design type must use a canonical token"
        )
    if "factorial" in design_value:
        require_filled_table(
            "Design And Identifiability factorial table",
            sections.get("Design And Identifiability", ""),
            errors,
        )
    if "adaptive" in design_value:
        adaptive = sections.get("Adaptive Decision Paths")
        if adaptive is None:
            errors.append("adaptive design requires the Adaptive Decision Paths section")
        else:
            require_filled_table("Adaptive Decision Paths", adaptive, errors)

    fair_fields = bullet_fields(sections.get("Fair Run Contract", ""))
    profile = find_field(fair_fields, "Selected decision profile")
    profile_value = profile[1].strip().strip("`").lower() if profile else ""
    if profile and not canonical_token(profile_value, DECISION_PROFILES):
        errors.append(
            "Fair Run Contract: Selected decision profile must use a canonical token"
        )
    retention = find_field(fair_fields, "Learned-state retention required")
    retention_value = retention[1].strip().lower() if retention else ""
    if retention_value == "yes":
        learned = sections.get("Learned-State Retention")
        if learned is None:
            errors.append("learned-state retention=yes requires the Learned-State Retention section")
        else:
            learned_fields = bullet_fields(learned)
            for prefix in LEARNED_STATE_FIELDS:
                found = find_field(learned_fields, prefix)
                if found is None or not explicit_value(found[1]):
                    errors.append(f"Learned-State Retention: unresolved field '{prefix}'")
    elif retention_value.startswith("no:"):
        if len(retention_value.split(":", 1)[1].strip()) < 8:
            errors.append("Learned-state retention no: value requires a scientific rationale")
    elif retention is not None:
        errors.append("Learned-state retention required must be 'yes' or 'no: <scientific rationale>'")

    return errors, status


def main() -> int:
    args = parse_args()
    card = args.card.resolve()
    if not card.is_file() or card.suffix.lower() != ".md":
        print("ROUTE_CARD_CONTRACT_FAILED")
        print(f"- card must be an existing Markdown file: {card}")
        return 2

    try:
        raw = card.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print("ROUTE_CARD_CONTRACT_FAILED")
        print(f"- unable to read UTF-8 card: {exc}")
        return 2

    errors, status = validate(text, args.launch_ready)
    if errors:
        print("ROUTE_CARD_CONTRACT_FAILED")
        for error in sorted(set(errors)):
            print(f"- {error}")
        return 1

    digest = hashlib.sha256(raw).hexdigest()
    print(f"ROUTE_CARD_CONTRACT_OK path={card} sha256={digest} status={status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

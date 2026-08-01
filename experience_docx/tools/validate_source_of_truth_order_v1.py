from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
checks = {
    "experience_docx/README.md": [
        "## Reading Order And Authority",
        "exact GitHub-main commit",
        "a scientific conclusion or next-stage authorization by themselves",
    ],
    "experience_docx/SCIENCE_FASTPATH.md": [
        "## Source-of-Truth Order",
        "not an\nexperiment-results store",
        "source of truth for current\nrules and compact terminal evidence",
    ],
    "experience_docx/CONVIR_OPS_MCP.md": [
        "never a result reader",
        "first route-bound call but not the",
        "scientific no-terminal conclusion",
    ],
    "experience_docx/ROUTE_READY_FASTPATH.md": [
        "binds the local route identity and worktree safety",
        "not evidence of a result or authorization",
        "next-stage permission from local files",
    ],
    "AGENTS.md": [
        "local identity and worktree-safety check",
        "cannot establish a metric, verdict, terminal",
        "After identity binding, read the exact GitHub-main authoritative",
    ],
}
for path, terms in checks.items():
    text = (root / path).read_text()
    missing = [term for term in terms if term not in text]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
policy = json.loads((root / "experience_docx/AI_POLICY_SNAPSHOT.json").read_text())
if policy["rules_commit"] != "bd73a099803dcca6ded09401ea7568ebb0e6ba71":
    raise SystemExit("rules commit mismatch")
sources = [item["source"] for item in policy["source_of_truth_order"]]
if sources != ["local_route_worktree", "github_main", "convir_4090_or_receipt_bound_mcp"]:
    raise SystemExit(f"source order mismatch: {sources}")
if any(route["snapshot_is_authority"] for route in policy["change_routes"].values()):
    raise SystemExit("snapshot authority flag is true")
print("SOURCE_OF_TRUTH_ORDER_CLOUD_ASSERTIONS_OK")

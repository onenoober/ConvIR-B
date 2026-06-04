import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "Dehazing" / "ITS"))
sys.path.insert(0, str(TOOLS_ROOT))

from audit_haze4k_apdr_v0_4b_mapping_triage import write_csv_union  # noqa: E402
from audit_haze4k_apdr_v0_4e_oof_calibration import (  # noqa: E402
    e1_gate_pass,
    fold_summary_rows,
    locked_rule_rows,
)
from audit_haze4k_apdr_v0_4e_risk_action_bank import (  # noqa: E402
    accepted_rejected_group_rows,
    attach_rank_groups,
    bad_labels,
    calibration_curve,
    candidate_table_rows,
    failure_signature_rows,
    parse_mapper_list,
    parse_rules,
    risk_feature_auc_rows,
    write_json,
)


STRING_KEYS = {
    "mapper",
    "family",
    "split",
    "name",
    "feature_set",
    "rank_group",
    "open_rank_group",
}
BOOL_KEYS = {"eval_target_split"}
INT_KEYS = {"fold", "low_size", "K", "index"}


def coerce_value(key, value):
    if value in (None, "", "None"):
        return None
    if key in STRING_KEYS:
        return value
    if key in BOOL_KEYS:
        return str(value).lower() == "true"
    if key in INT_KEYS:
        return int(float(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_action_rows(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: coerce_value(key, value) for key, value in row.items()})
    return rows


def unique_image_rows(action_rows):
    by_index = {}
    for row in action_rows:
        by_index.setdefault(int(row["index"]), row)
    return [by_index[index] for index in sorted(by_index)]


def fold_rows(image_rows):
    rows = []
    for fold_id in sorted({int(row["fold"]) for row in image_rows}):
        members = [row for row in image_rows if int(row["fold"]) == fold_id]
        rows.append(
            {
                "fold": fold_id,
                "eval_count": len(members),
                "eval_open_count": sum(float(row["P_benefit"]) >= 0.5 for row in members),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_image_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="apdr_v0_4e_oof_calibration_sigma3_seed3407")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--locked_rules", default="")
    parser.add_argument(
        "--risk_feature_keys",
        default=(
            "pred_abs_mean,pred_abs_max,pred_coeff_norm,pred_low_energy,weighted_residual_norm,"
            "nn_distance,confidence_proxy,kenel_confidence,proxy_score,M_safe_mean,M_safe_nonzero_frac"
        ),
    )
    parser.add_argument("--calibration_score_key", default="pred_abs_mean")
    parser.add_argument("--calibration_bins", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_rows = read_action_rows(args.per_image_csv)
    if not action_rows:
        raise ValueError(f"No action rows found in {args.per_image_csv}")

    attach_rank_groups(action_rows, "oof")
    locked_rules = parse_rules(args.locked_rules)
    risk_feature_keys = parse_mapper_list(args.risk_feature_keys)

    candidate_rows = candidate_table_rows(action_rows)
    locked_rows, group_rule_rows = locked_rule_rows(locked_rules, action_rows)
    fold_locked_rows = fold_summary_rows(action_rows, locked_rules)
    auc_rows = risk_feature_auc_rows(action_rows, risk_feature_keys, "oof")
    curve_rows = calibration_curve(action_rows, args.calibration_score_key, "oof", args.calibration_bins)
    group_rows = accepted_rejected_group_rows(group_rule_rows, action_rows)
    failure_rows = failure_signature_rows(action_rows, risk_feature_keys, "oof")

    write_csv_union(output_dir / "v04e_oof_candidate_action_table.csv", candidate_rows)
    write_csv_union(output_dir / "v04e_oof_locked_threshold_by_fold.csv", fold_locked_rows)
    write_csv_union(output_dir / "v04e_oof_risk_feature_auc.csv", auc_rows)
    write_csv_union(output_dir / "v04e_oof_calibration_curve.csv", curve_rows)
    write_csv_union(output_dir / "v04e_oof_accepted_vs_rejected_groups.csv", group_rows)
    write_csv_union(output_dir / "v04e_oof_strong_failure_signature.csv", failure_rows)

    image_rows = unique_image_rows(action_rows)
    bad_count = sum(bad_labels(action_rows))
    summary = {
        "stage": "APDR-v0.4E 5-fold OOF candidate-action risk calibration",
        "tag": args.tag,
        "status": "OOF calibration finalized from per-image intermediate table; no training",
        "source_per_image_csv": str(args.per_image_csv),
        "sigma": args.sigma,
        "data_count": len(image_rows),
        "active_open_count": sum(float(row["P_benefit"]) >= 0.5 for row in image_rows),
        "fold_count": len({int(row["fold"]) for row in image_rows}),
        "folds": fold_rows(image_rows),
        "locked_rules": locked_rows,
        "all_locked_rules_pass": all(row.get("gate_pass") for row in locked_rows),
        "action_row_count": len(action_rows),
        "bad_label_count": bad_count,
        "candidate_mappers": sorted({row["mapper"] for row in action_rows}),
        "candidate_k_values": sorted({int(row["K"]) for row in action_rows}),
        "candidate_scales": sorted({float(row["candidate_scale"]) for row in action_rows}),
        "gate_definition": (
            "severe=0,strong_rate<=1%,easy>=-0.02,hard>=0.25,mean>0,"
            "coverage>=10%,oracle_recovery>=0.15"
        ),
        "outputs": {
            "candidate_action_per_image": str(args.per_image_csv),
            "candidate_action_table": str(output_dir / "v04e_oof_candidate_action_table.csv"),
            "locked_threshold_by_fold": str(output_dir / "v04e_oof_locked_threshold_by_fold.csv"),
            "risk_feature_auc": str(output_dir / "v04e_oof_risk_feature_auc.csv"),
            "oof_calibration_curve": str(output_dir / "v04e_oof_calibration_curve.csv"),
            "accepted_vs_rejected_groups": str(output_dir / "v04e_oof_accepted_vs_rejected_groups.csv"),
            "strong_failure_signature": str(output_dir / "v04e_oof_strong_failure_signature.csv"),
        },
        "args": vars(args),
    }
    for row in summary["locked_rules"]:
        row["gate_pass_recomputed"] = e1_gate_pass(row, row["count"]) if row.get("status") != "missing_candidate" else False
    summary_path = output_dir / "v04e_oof_locked_threshold_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

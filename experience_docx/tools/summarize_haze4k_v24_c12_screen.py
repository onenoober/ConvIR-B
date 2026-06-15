#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    summaries = []
    for p in sorted(args.out_dir.glob("v24_c12_eval_*_summary.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        summaries.append(payload)
    summaries.sort(
        key=lambda r: (
            bool(r.get("gate_pass")),
            float(r.get("mean_dPSNR", 0.0)),
            float(r.get("hard_bottom25_dPSNR", 0.0)),
            -float(r.get("severe_loss_per_600", 999.0)),
        ),
        reverse=True,
    )
    write_csv(args.out_dir / "v24_c12_screen_leaderboard.csv", summaries)
    best = summaries[0] if summaries else {}
    decision = (
        "C12_SCREEN_PASS_FORMAL_DISTILLATION_REVIEW"
        if best.get("gate_pass")
        else "C12_SCREEN_FAIL_KEEP_WD0375_TEACHER"
    )
    (args.out_dir / "v24_c12_decision.md").write_text(
        "# C12 Screen Decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"Best variant: `{best.get('variant', 'none')}` checkpoint `{best.get('checkpoint', 'none')}`.\n\n"
        "Best mean/hard/easy/positive/severe: "
        f"`{float(best.get('mean_dPSNR', 0.0)):.6f}` / "
        f"`{float(best.get('hard_bottom25_dPSNR', 0.0)):.6f}` / "
        f"`{float(best.get('easy_top25_dPSNR', 0.0)):.6f}` / "
        f"`{float(best.get('positive_ratio', 0.0)):.6f}` / "
        f"`{float(best.get('severe_loss_per_600', 0.0)):.2f}/600`.\n\n"
        "Locked Haze4K remains untouched. C12 did not use locked outputs as targets.\n",
        encoding="utf-8",
    )
    readme = "# Haze4K v2.4 C12 WD0375 Distillation Evidence\n\n"
    readme += f"Decision: `{decision}`\n\n"
    if best:
        readme += (
            "Best screen result: "
            f"`{best.get('variant')}` `{best.get('checkpoint')}` with mean/hard/easy "
            f"`{float(best.get('mean_dPSNR', 0.0)):.6f}` / "
            f"`{float(best.get('hard_bottom25_dPSNR', 0.0)):.6f}` / "
            f"`{float(best.get('easy_top25_dPSNR', 0.0)):.6f}`, positive "
            f"`{float(best.get('positive_ratio', 0.0)):.6f}`, severe "
            f"`{float(best.get('severe_loss_per_600', 0.0)):.2f}/600`.\n\n"
        )
    readme += (
        "C12 is train-derived only. It uses WD0375 teacher caches generated from "
        "Haze4K train-core images and evaluates on the held-out C8 val_regular + "
        "val_hard names. Locked Haze4K remains untouched.\n"
    )
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print("C12_SCREEN_SUMMARY_OK", json.dumps({"decision": decision, "rows": len(summaries)}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""C2c learned tabular router screen from deployable C2 features.

This is a lightweight train-derived router audit: a small MLP predicts FullUDP
utility/risk from C2 deployable features, then a train-fold threshold is replayed
on held-out folds. It writes text evidence only and never touches locked data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from audit_haze4k_v20_c2b_multirule_router import (  # noqa: E402
    ABSTENTION_GATE,
    POLICY_FEATURES,
    STRICT_GATE,
    Table,
    abstention_gate_pass,
    fnum,
    fold_id,
    score as gate_score,
    strict_gate_pass,
    summarize_mask,
    write_csv,
)


SEEDS = [3407, 3411, 2026]
THRESHOLD_QUANTILES = [0.50, 0.60, 0.67, 0.75, 0.80, 0.85, 0.90, 0.95]
RISK_QUANTILES = [0.50, 0.60, 0.67, 0.75, 0.80, 0.85, 0.90]


class RouterMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 24),
            nn.GELU(),
        )
        self.head = nn.Linear(24, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bool_json(value: Any) -> bool:
    return bool(value) if not isinstance(value, np.bool_) else bool(value.item())


def make_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([[fnum(row[feature]) for feature in POLICY_FEATURES] for row in rows], dtype=np.float32)
    y = np.array([fnum(row["dPSNR"]) for row in rows], dtype=np.float32)
    y_pos = (y > 0.0).astype(np.float32)
    y_loss = (y < 0.0).astype(np.float32)
    y_severe = (y <= -0.20).astype(np.float32)
    return x, y, y_pos, y_loss, y_severe


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (other_x - mean) / std, np.concatenate([mean, std], axis=0)


def pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = float(labels.sum())
    neg = float(labels.shape[0] - pos)
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(min(20.0, max(1.0, neg / pos)), dtype=torch.float32)


def train_predict(
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    seed: int,
    epochs: int,
    lr: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    train_x, train_y, train_pos, train_loss, train_severe = make_xy(train_rows)
    heldout_x, _heldout_y, _heldout_pos, _heldout_loss, _heldout_severe = make_xy(heldout_rows)
    train_x, heldout_x, _stats = standardize(train_x, heldout_x)

    model = RouterMLP(train_x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y).view(-1, 1)
    y_pos = torch.from_numpy(train_pos).view(-1, 1)
    y_loss = torch.from_numpy(train_loss).view(-1, 1)
    y_severe = torch.from_numpy(train_severe).view(-1, 1)
    bce_pos = nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_pos))
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_loss))
    bce_severe = nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_severe))
    huber = nn.SmoothL1Loss(beta=0.25)

    model.train()
    for _epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        pred_gain = out[:, 0:1]
        loss = (
            huber(pred_gain, y)
            + 0.25 * bce_pos(out[:, 1:2], y_pos)
            + 0.35 * bce_loss(out[:, 2:3], y_loss)
            + 0.45 * bce_severe(out[:, 3:4], y_severe)
        )
        loss.backward()
        opt.step()

    def predict(arr: np.ndarray) -> dict[str, np.ndarray]:
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(arr)).float()
            pred_gain = out[:, 0].numpy()
            p_pos = torch.sigmoid(out[:, 1]).numpy()
            p_loss = torch.sigmoid(out[:, 2]).numpy()
            p_severe = torch.sigmoid(out[:, 3]).numpy()
        score = pred_gain + 0.15 * p_pos - 0.25 * p_loss - 0.55 * p_severe
        return {
            "pred_gain": pred_gain,
            "p_pos": p_pos,
            "p_loss": p_loss,
            "p_severe": p_severe,
            "router_score": score,
        }

    return predict(train_x), predict(heldout_x)


def average_predictions(preds: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = preds[0].keys()
    return {key: np.mean([pred[key] for pred in preds], axis=0) for key in keys}


def mask_from_policy(pred: dict[str, np.ndarray], policy: dict[str, float | str]) -> np.ndarray:
    score_name = str(policy["score_name"])
    threshold = float(policy["threshold"])
    mask = pred[score_name] >= threshold
    risk_name = str(policy.get("risk_name", "none"))
    if risk_name != "none":
        mask &= pred[risk_name] <= float(policy["risk_threshold"])
    return mask


def summarize_pred_policy(table: Table, pred: dict[str, np.ndarray], policy: dict[str, float | str]) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "policy_id": policy_id(policy),
        "score_name": policy["score_name"],
        "threshold": policy["threshold"],
        "risk_name": policy.get("risk_name", "none"),
        "risk_threshold": policy.get("risk_threshold", ""),
    }
    rec.update(summarize_mask(table, mask_from_policy(pred, policy)))
    rec["strict_gate_pass"] = strict_gate_pass(rec)
    rec["abstention_gate_pass"] = abstention_gate_pass(rec)
    rec["score"] = gate_score(rec)
    return rec


def policy_id(policy: dict[str, float | str]) -> str:
    base = f"{policy['score_name']}_ge_{float(policy['threshold']):.8g}"
    if str(policy.get("risk_name", "none")) != "none":
        return f"{base}_AND_{policy['risk_name']}_le_{float(policy['risk_threshold']):.8g}"
    return base


def candidate_policies(pred: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    policies: list[dict[str, float | str]] = []
    for score_name in ["router_score", "pred_gain", "p_pos"]:
        vals = pred[score_name]
        for q in THRESHOLD_QUANTILES:
            threshold = float(np.quantile(vals, q))
            policies.append({"score_name": score_name, "threshold": threshold, "risk_name": "none"})
            for risk_name in ["p_loss", "p_severe"]:
                risk_vals = pred[risk_name]
                for rq in RISK_QUANTILES:
                    policies.append(
                        {
                            "score_name": score_name,
                            "threshold": threshold,
                            "risk_name": risk_name,
                            "risk_threshold": float(np.quantile(risk_vals, rq)),
                        }
                    )
    return policies


def choose_policy(train_table: Table, train_pred: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = [summarize_pred_policy(train_table, train_pred, policy) for policy in candidate_policies(train_pred)]
    rows = [row for row in rows if fnum(row["coverage"]) >= 0.08]
    rows.sort(key=lambda row: (bool(row["strict_gate_pass"]), bool(row["abstention_gate_pass"]), fnum(row["score"])), reverse=True)
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_rows", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--lr", type=float, default=0.008)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.feature_rows)
    table = Table(rows, POLICY_FEATURES)
    fold_ids = np.array([fold_id(str(name)) for name in table.names], dtype=np.int64)

    pred_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selected_oof: set[str] = set()
    all_train_chosen: list[dict[str, Any]] = []

    for fold in range(5):
        train_mask = fold_ids != fold
        heldout_mask = fold_ids == fold
        train_rows = [row for row, keep in zip(rows, train_mask, strict=False) if keep]
        heldout_rows = [row for row, keep in zip(rows, heldout_mask, strict=False) if keep]
        train_table = Table(train_rows, POLICY_FEATURES)
        heldout_table = Table(heldout_rows, POLICY_FEATURES)

        train_preds = []
        heldout_preds = []
        for seed in SEEDS:
            train_pred, heldout_pred = train_predict(train_rows, heldout_rows, seed, args.epochs, args.lr)
            train_preds.append(train_pred)
            heldout_preds.append(heldout_pred)
        train_pred_avg = average_predictions(train_preds)
        heldout_pred_avg = average_predictions(heldout_preds)
        chosen = choose_policy(train_table, train_pred_avg)
        all_train_chosen.append(chosen)

        heldout_policy = {
            "score_name": chosen["score_name"],
            "threshold": float(chosen["threshold"]),
            "risk_name": chosen["risk_name"],
            "risk_threshold": float(chosen["risk_threshold"]) if chosen["risk_threshold"] != "" else 0.0,
        }
        heldout_selected = mask_from_policy(heldout_pred_avg, heldout_policy)
        eval_rec: dict[str, Any] = {
            "fold": fold,
            "train_policy_id": chosen["policy_id"],
            "train_strict_gate_pass": chosen["strict_gate_pass"],
            "train_abstention_gate_pass": chosen["abstention_gate_pass"],
        }
        eval_rec.update(summarize_mask(heldout_table, heldout_selected))
        eval_rec["strict_gate_pass"] = strict_gate_pass(eval_rec)
        eval_rec["abstention_gate_pass"] = abstention_gate_pass(eval_rec)
        eval_rec["score"] = gate_score(eval_rec)
        fold_rows.append(eval_rec)

        for idx, row in enumerate(heldout_rows):
            selected = bool(heldout_selected[idx])
            if selected:
                selected_oof.add(str(row["name"]))
            pred_rows.append(
                {
                    "name": row["name"],
                    "fold": fold,
                    "split": row["split"],
                    "A0_PSNR": row["A0_PSNR"],
                    "dPSNR": row["dPSNR"],
                    "dSSIM": row["dSSIM"],
                    "router_score": heldout_pred_avg["router_score"][idx],
                    "pred_gain": heldout_pred_avg["pred_gain"][idx],
                    "p_pos": heldout_pred_avg["p_pos"][idx],
                    "p_loss": heldout_pred_avg["p_loss"][idx],
                    "p_severe": heldout_pred_avg["p_severe"][idx],
                    "selected_oof": selected,
                }
            )

    pred_rows.sort(key=lambda row: str(row["name"]))
    write_csv(
        args.out_dir / "v20_c2c_mlp_oof_predictions.csv",
        pred_rows,
        ["name", "fold", "split", "A0_PSNR", "dPSNR", "dSSIM", "router_score", "pred_gain", "p_pos", "p_loss", "p_severe", "selected_oof"],
    )
    fold_fields = [
        "fold",
        "train_policy_id",
        "train_strict_gate_pass",
        "train_abstention_gate_pass",
        "count",
        "selected_count",
        "coverage",
        "mean_dPSNR",
        "hard_bottom25_dPSNR",
        "easy_top25_dPSNR",
        "dSSIM",
        "positive_ratio",
        "nonnegative_ratio",
        "severe_loss_count",
        "severe_loss_per_600",
        "strong_loss_count",
        "strong_loss_per_600",
        "selected_precision",
        "selected_nonnegative_ratio",
        "selected_severe_count",
        "strict_gate_pass",
        "abstention_gate_pass",
        "score",
    ]
    write_csv(args.out_dir / "v20_c2c_mlp_fold_metrics.csv", fold_rows, fold_fields)

    oof_mask = np.array([str(name) in selected_oof for name in table.names], dtype=bool)
    oof_summary = summarize_mask(table, oof_mask)
    oof_summary["strict_gate_pass"] = strict_gate_pass(oof_summary)
    oof_summary["abstention_gate_pass"] = abstention_gate_pass(oof_summary)
    oof_summary["score"] = gate_score(oof_summary)

    if oof_summary["strict_gate_pass"]:
        decision = "C2C_MLP_STRICT_SCREEN_PASS_START_C3_SHIFTED"
    elif oof_summary["abstention_gate_pass"]:
        decision = "C2C_MLP_ABSTENTION_SCREEN_PASS_START_C3_SHIFTED"
    else:
        decision = "C2C_MLP_ROUTER_SCREEN_FAIL_REASSESS_FEATURES_OR_EXPERT"

    summary = {
        "route": "Haze4K-v2.0 StrongExpert-GainMix",
        "phase": "C2c MLP OutputDiff Router Screen",
        "locked_test_touched": False,
        "feature_rows": str(args.feature_rows),
        "rows": len(rows),
        "policy_features": POLICY_FEATURES,
        "seeds": SEEDS,
        "epochs": args.epochs,
        "lr": args.lr,
        "strict_gate": STRICT_GATE,
        "abstention_gate": ABSTENTION_GATE,
        "fold_rows": fold_rows,
        "train_chosen_policies": all_train_chosen,
        "oof_summary": oof_summary,
        "decision": decision,
    }
    (args.out_dir / "v20_c2c_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=bool_json) + "\n", encoding="utf-8")

    lines = [
        "# Haze4K v2.0 C2c MLP OutputDiff Router Screen",
        "",
        f"Decision: `{decision}`",
        "",
        "C2c trains a small tabular MLP on train folds only and replays selected abstention thresholds on held-out folds.",
        "Only C2 deployable features are used for routing; A0_PSNR is used only for evaluation buckets.",
        "No raw images/tensors were read or written, and locked test data was not touched.",
        "",
        "## OOF Replay",
        "",
    ]
    for key, value in oof_summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Fold Policies", ""])
    for row in fold_rows:
        lines.append(
            f"- fold `{row['fold']}`: `{row['train_policy_id']}`, "
            f"mean `{row['mean_dPSNR']}`, hard `{row['hard_bottom25_dPSNR']}`, "
            f"easy `{row['easy_top25_dPSNR']}`, pass `{row['abstention_gate_pass']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- C3 shifted validation is authorized only if the OOF screen passes.",
            "- If OOF fails, the current FullUDP-A0 feature set is not stable enough for promotion; acquire stronger features/expert compatibility before locked test.",
        ]
    )
    (args.out_dir / "v20_c2c_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V20_C2C_MLP_ROUTER_OK decision={decision} rows={len(rows)} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

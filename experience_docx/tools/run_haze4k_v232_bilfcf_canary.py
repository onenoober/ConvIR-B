import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.data_load import DeblurDataset
from models.ConvIR import build_bilfcf_net, build_net


ALLOWED_NEW_PREFIXES = ("BILFCF_",)


def load_checkpoint_model(path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_haze4k_partial(model, checkpoint_path):
    state = load_checkpoint_model(checkpoint_path, "cpu")
    model_state = model.state_dict()
    loaded = {}
    shape_mismatch = []
    unexpected = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif model_state[key].shape != value.shape:
            shape_mismatch.append([key, list(value.shape), list(model_state[key].shape)])
        else:
            loaded[key] = value
    missing = [key for key in model_state if key not in loaded]
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in ALLOWED_NEW_PREFIXES)
    ]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            "partial-load failed: "
            f"unexpected={unexpected[:20]} shape_mismatch={shape_mismatch[:20]} "
            f"bad_missing={bad_missing[:20]}"
        )
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)


def build_models(args, device):
    official = build_net("base", "Haze4K", "original").to(device).eval()
    official.load_state_dict(load_checkpoint_model(args.checkpoint, device))
    candidate = build_bilfcf_net(
        "base",
        "Haze4K",
        "original",
        insertion="s5",
        alpha_max=args.alpha_max,
        gate_bias=args.gate_bias,
        hidden_channels=args.hidden_channels,
        lowpass_kernel=args.lowpass_kernel,
    ).to(device)
    load_haze4k_partial(candidate, args.checkpoint)
    return official, candidate


def set_adapter_only(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(ALLOWED_NEW_PREFIXES)
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No BILFCF trainable parameters found")
    return trainable


def ensure_crop_size(tensor, size):
    _, h, w = tensor.shape
    if h >= size and w >= size:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(max(size, h), max(size, w)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def crop_pair(input_img, label_img, size, rng=None):
    input_img = ensure_crop_size(input_img, size)
    label_img = ensure_crop_size(label_img, size)
    _, h, w = input_img.shape
    if rng is None:
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
    else:
        top = rng.randint(0, max(0, h - size))
        left = rng.randint(0, max(0, w - size))
    return (
        input_img[:, top : top + size, left : left + size],
        label_img[:, top : top + size, left : left + size],
    )


def make_batch(dataset, indices, crop_size, device, rng=None):
    inputs = []
    labels = []
    names = []
    for index in indices:
        input_img, label_img = dataset[index]
        input_img, label_img = crop_pair(input_img, label_img, crop_size, rng)
        inputs.append(input_img)
        labels.append(label_img)
        names.append(dataset.image_list[index])
    return torch.stack(inputs).to(device), torch.stack(labels).to(device), names


def iter_batches(indices, batch_size, seed, shuffle=True):
    ordered = list(indices)
    if shuffle:
        random.Random(seed).shuffle(ordered)
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


def psnr_per_sample(pred, label):
    mse = (pred - label).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def charbonnier_loss(pred, label, eps=1e-3):
    return torch.sqrt((pred - label).pow(2) + eps * eps).mean()


def multiscale_reconstruction_loss(outputs, label):
    label2 = F.interpolate(label, scale_factor=0.5, mode="bilinear", align_corners=False)
    label4 = F.interpolate(label, scale_factor=0.25, mode="bilinear", align_corners=False)
    loss_content = (
        charbonnier_loss(outputs[0], label4)
        + charbonnier_loss(outputs[1], label2)
        + charbonnier_loss(outputs[2], label)
    )
    fft_loss = 0.0
    for pred, target in ((outputs[0], label4), (outputs[1], label2), (outputs[2], label)):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        fft_loss = fft_loss + charbonnier_loss(pred_fft.real, target_fft.real)
        fft_loss = fft_loss + charbonnier_loss(pred_fft.imag, target_fft.imag)
    return loss_content + 0.05 * fft_loss


def lowpass(tensor, kernel=9):
    return F.avg_pool2d(tensor, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)


def route_loss(candidate, official, input_img, label_img, loss_variant, easy_threshold, amp_weight):
    outputs = candidate(input_img)
    loss = multiscale_reconstruction_loss(outputs, label_img)
    with torch.no_grad():
        official_final = official(input_img)[2]
        base_psnr = psnr_per_sample(torch.clamp(official_final, 0, 1), label_img)
        base_l1 = (official_final - label_img).abs().flatten(1).mean(dim=1)
    candidate_final = outputs[2]
    if loss_variant in ("B", "C", "D"):
        easy_mask = base_psnr >= easy_threshold
        if easy_mask.any():
            loss = loss + 0.25 * charbonnier_loss(candidate_final[easy_mask], official_final[easy_mask])
    if loss_variant in ("C", "D"):
        cand_l1 = (candidate_final - label_img).abs().flatten(1).mean(dim=1)
        tail_penalty = torch.relu(cand_l1 - base_l1 + 0.001)
        k = max(1, math.ceil(tail_penalty.numel() * 0.25))
        loss = loss + 0.5 * torch.topk(tail_penalty, k=k, largest=True).values.mean()
    if loss_variant == "D":
        loss = loss + 0.25 * charbonnier_loss(lowpass(candidate_final), lowpass(label_img))
    reg = candidate.bilfcf_regularization()
    if reg is not None and amp_weight > 0:
        loss = loss + amp_weight * reg
    return loss


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def cvar(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    k = max(1, math.ceil(len(ordered) * pct / 100.0))
    return statistics.mean(ordered[:k])


def summarize_rows(rows, label):
    deltas = [row["delta_psnr"] for row in rows]
    if not deltas:
        return {
            "bucket": label,
            "count": 0,
            "mean_delta": None,
            "hard_gain": None,
            "easy_gain": None,
            "p05": None,
            "cvar5": None,
            "severe_rate": None,
            "strong_reference_regression_rate": None,
        }
    return {
        "bucket": label,
        "count": len(rows),
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "p05": percentile(deltas, 5),
        "cvar5": cvar(deltas, 5),
        "severe_rate": sum(delta <= -0.20 for delta in deltas) / len(deltas),
        "strong_reference_regression_rate": sum(delta <= -0.05 for delta in deltas) / len(deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / len(deltas),
    }


def field_summary(rows, label):
    keys = [
        "field_energy_mean",
        "field_energy_p95",
        "gate_mean",
        "gate_std",
        "highfreq_leakage",
        "lowfreq_ratio",
    ]
    result = {"bucket": label, "count": len(rows)}
    for key in keys:
        vals = [row[key] for row in rows if row.get(key) is not None]
        result[key] = statistics.mean(vals) if vals else None
    return result


def evaluate_model(official, candidate, dataset, indices, args, device):
    rows = []
    candidate.eval()
    official.eval()
    with torch.no_grad():
        for index in indices:
            input_img, label_img, names = make_batch(dataset, [index], args.crop_size, device, rng=None)
            base_pred = torch.clamp(official(input_img)[2], 0, 1)
            cand_pred = torch.clamp(candidate(input_img)[2], 0, 1)
            base_psnr = psnr_per_sample(base_pred, label_img)[0].item()
            cand_psnr = psnr_per_sample(cand_pred, label_img)[0].item()
            stats = candidate.get_bilfcf_stats().get("BILFCF_s5", {})
            rows.append(
                {
                    "index": index,
                    "name": names[0],
                    "base_psnr": base_psnr,
                    "candidate_psnr": cand_psnr,
                    "delta_psnr": cand_psnr - base_psnr,
                    **stats,
                }
            )
    rows_sorted = sorted(rows, key=lambda row: row["base_psnr"])
    n = len(rows_sorted)
    hard = rows_sorted[: max(1, n // 4)]
    easy = rows_sorted[max(0, n - max(1, n // 4)) :]
    middle = rows_sorted[max(1, n // 4) : max(1, n - max(1, n // 4))]
    summaries = {
        "all": summarize_rows(rows_sorted, "all"),
        "hard": summarize_rows(hard, "hard"),
        "middle": summarize_rows(middle, "middle"),
        "easy": summarize_rows(easy, "easy"),
    }
    field_summaries = {
        "all": field_summary(rows_sorted, "all"),
        "hard": field_summary(hard, "hard"),
        "middle": field_summary(middle, "middle"),
        "easy": field_summary(easy, "easy"),
    }
    return rows_sorted, summaries, field_summaries


def identity_check(official, candidate, device, crop_size):
    x = torch.rand(1, 3, crop_size, crop_size, device=device)
    official.eval()
    candidate.eval()
    with torch.no_grad():
        base = official(x)
        cand = candidate(x)
    max_abs = max((a - b).abs().max().item() for a, b in zip(base, cand))
    return max_abs


def train_candidate(official, candidate, dataset, train_indices, args, device, easy_threshold):
    trainable = set_adapter_only(candidate)
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    candidate.train()
    step = 0
    for epoch in range(args.epochs):
        for batch_indices in iter_batches(train_indices, args.batch_size, args.seed + epoch, shuffle=True):
            rng = random.Random(args.seed + epoch * 100000 + step)
            input_img, label_img, _ = make_batch(dataset, batch_indices, args.crop_size, device, rng=rng)
            optimizer.zero_grad()
            loss = route_loss(
                candidate,
                official,
                input_img,
                label_img,
                args.loss_variant,
                easy_threshold,
                args.amplitude_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip_norm)
            optimizer.step()
            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                return step
    return step


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_report(phase, fold, summaries, identity_max_abs, train_steps, loss_variant):
    row = {
        "phase": phase,
        "fold": fold,
        "loss_variant": loss_variant,
        "identity_max_abs_vs_A0_start": identity_max_abs,
        "identity_start_pass": identity_max_abs <= 1e-6,
        "train_steps": train_steps,
        "mean_delta": summaries["all"]["mean_delta"],
        "hard_gain": summaries["hard"]["mean_delta"],
        "easy_gain": summaries["easy"]["mean_delta"],
        "p05": summaries["all"]["p05"],
        "cvar5": summaries["all"]["cvar5"],
        "severe_rate": summaries["all"]["severe_rate"],
        "strong_reference_regression_rate": summaries["easy"]["strong_reference_regression_rate"],
    }
    return row


def gate_canary32(row, field_row):
    return (
        bool(row["identity_start_pass"])
        and row["mean_delta"] is not None
        and row["mean_delta"] >= 0.50
        and field_row["field_energy_mean"] is not None
        and field_row["field_energy_mean"] > 0.0
        and field_row["gate_std"] is not None
    )


def gate_canary80(rows):
    mean_delta = statistics.mean(row["mean_delta"] for row in rows)
    hard_gain = statistics.mean(row["hard_gain"] for row in rows)
    easy_gain = statistics.mean(row["easy_gain"] for row in rows)
    p05 = min(row["p05"] for row in rows)
    cvar5_value = min(row["cvar5"] for row in rows)
    severe_rate = statistics.mean(row["severe_rate"] for row in rows)
    strong_reg = statistics.mean(row["strong_reference_regression_rate"] for row in rows)
    fold_tail_pass = sum(
        row["p05"] >= -0.20 and row["cvar5"] >= -0.50 and row["severe_rate"] <= 0.05
        for row in rows
    )
    return {
        "mean_delta": mean_delta,
        "hard_gain": hard_gain,
        "easy_gain": easy_gain,
        "p05_min_fold": p05,
        "cvar5_min_fold": cvar5_value,
        "severe_rate_mean": severe_rate,
        "strong_reference_regression_rate_mean": strong_reg,
        "fold_tail_pass": fold_tail_pass,
        "pass": (
            mean_delta >= 0.10
            and hard_gain >= 0.30
            and easy_gain >= -0.05
            and p05 >= -0.20
            and cvar5_value >= -0.50
            and severe_rate <= 0.05
            and strong_reg <= 0.05
            and fold_tail_pass >= 4
        ),
    }


def run_single_phase(args, device, dataset, indices, phase):
    official, candidate = build_models(args, device)
    identity = identity_check(official, candidate, device, args.crop_size)
    _, base_summaries, _ = evaluate_model(official, candidate, dataset, indices, args, device)
    easy_threshold = percentile(
        [row["base_psnr"] for row in evaluate_model(official, candidate, dataset, indices, args, device)[0]],
        75,
    )
    train_steps = train_candidate(official, candidate, dataset, indices, args, device, easy_threshold)
    rows, summaries, field_summaries = evaluate_model(official, candidate, dataset, indices, args, device)
    report = flatten_report(phase, "all", summaries, identity, train_steps, args.loss_variant)
    report["base_mean_delta_before_train"] = base_summaries["all"]["mean_delta"]
    return rows, summaries, field_summaries, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["p1_sanity", "canary32", "canary80_oof"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--sample_count", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=0.001)
    parser.add_argument("--loss_variant", default="C", choices=["A", "B", "C", "D"])
    parser.add_argument("--amplitude_loss_weight", type=float, default=0.01)
    parser.add_argument("--alpha_max", type=float, default=0.02)
    parser.add_argument("--gate_bias", type=float, default=-4.0)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--lowpass_kernel", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    dataset = DeblurDataset(os.path.join(args.data_dir, "train"), "Haze4K", transform=None)
    indices = list(range(min(args.sample_count, len(dataset))))
    if len(indices) < args.sample_count:
        raise RuntimeError(f"Requested {args.sample_count} samples but found {len(indices)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainability_rows = []
    field_rows = []
    preservation_rows = []
    closeout = {
        "phase": args.phase,
        "loss_variant": args.loss_variant,
        "sample_count": args.sample_count,
        "locked_test_touched": False,
        "p2b_selector_probe_launched": False,
        "rgb_output_output_residual": False,
        "learned_rgb_post_output_correction": False,
    }

    if args.phase in ("p1_sanity", "canary32"):
        rows, summaries, fields, report = run_single_phase(args, device, dataset, indices, args.phase)
        trainability_rows.append(report)
        for bucket, item in fields.items():
            field_rows.append({"phase": args.phase, "fold": "all", **item})
        for bucket, item in summaries.items():
            preservation_rows.append({"phase": args.phase, "fold": "all", **item})
        closeout["gate_pass"] = (
            fields["all"]["field_energy_mean"] is not None
            and fields["all"]["field_energy_mean"] >= 0.0
            and fields["all"]["highfreq_leakage"] is not None
            and fields["all"]["highfreq_leakage"] <= 0.85
            and bool(report["identity_start_pass"])
        )
        if args.phase == "canary32":
            closeout["gate_pass"] = gate_canary32(report, fields["all"])
        closeout["decision"] = (
            f"{args.phase.upper()}_PASS" if closeout["gate_pass"] else f"{args.phase.upper()}_FAIL_NORMAL_PAUSE"
        )
    else:
        fold_count = 5
        fold_size = math.ceil(len(indices) / fold_count)
        for fold in range(fold_count):
            val_indices = indices[fold * fold_size : (fold + 1) * fold_size]
            train_indices = [index for index in indices if index not in val_indices]
            official, candidate = build_models(args, device)
            identity = identity_check(official, candidate, device, args.crop_size)
            base_rows, _, _ = evaluate_model(official, candidate, dataset, train_indices, args, device)
            easy_threshold = percentile([row["base_psnr"] for row in base_rows], 75)
            train_steps = train_candidate(
                official,
                candidate,
                dataset,
                train_indices,
                args,
                device,
                easy_threshold,
            )
            rows, summaries, fields = evaluate_model(official, candidate, dataset, val_indices, args, device)
            trainability_rows.append(
                flatten_report(args.phase, str(fold), summaries, identity, train_steps, args.loss_variant)
            )
            for bucket, item in fields.items():
                field_rows.append({"phase": args.phase, "fold": str(fold), **item})
            for bucket, item in summaries.items():
                preservation_rows.append({"phase": args.phase, "fold": str(fold), **item})
        gate = gate_canary80(trainability_rows)
        closeout.update(gate)
        closeout["gate_pass"] = gate["pass"]
        closeout["decision"] = (
            "CANARY80_OOF_PASS" if gate["pass"] else "CANARY80_OOF_FAIL_NORMAL_PAUSE"
        )

    if args.phase == "p1_sanity":
        trainability_name = "v232_p1_field_sanity_report.csv"
        field_name = "v232_p1_field_energy_by_bucket.csv"
        preservation_name = "v232_p1_easy_strong_reference_preservation.csv"
        closeout_name = "v232_p1_field_sanity_closeout.json"
        write_csv(output_dir / "v232_p1_highfreq_leakage_report.csv", field_rows)
    elif args.phase == "canary32":
        trainability_name = "v232_p2_canary32_trainability_report.csv"
        field_name = "v232_p2_canary32_field_energy_by_bucket.csv"
        preservation_name = "v232_p2_canary32_easy_strong_reference_preservation.csv"
        closeout_name = "v232_p2_canary32_closeout.json"
    else:
        trainability_name = "v232_p2_canary80_oof_tail_report.csv"
        field_name = "v232_p2_field_energy_by_bucket.csv"
        preservation_name = "v232_p2_easy_strong_reference_preservation.csv"
        closeout_name = "v232_p2_canary80_oof_closeout.json"

    write_csv(output_dir / trainability_name, trainability_rows)
    write_csv(output_dir / field_name, field_rows)
    write_csv(output_dir / preservation_name, preservation_rows)
    (output_dir / closeout_name).write_text(
        json.dumps(closeout, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(closeout, indent=2))
    if not closeout["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

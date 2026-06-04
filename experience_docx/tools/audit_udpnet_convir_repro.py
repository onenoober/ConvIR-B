#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def maybe_clone(repo_dir, clone_url, clone_if_missing, clone_depth):
    repo_dir = Path(repo_dir)
    if repo_dir.exists():
        return repo_dir
    if not clone_if_missing:
        raise FileNotFoundError(f"UDPNet repo_dir does not exist: {repo_dir}")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if clone_depth > 0:
        cmd.extend(["--depth", str(clone_depth)])
    cmd.extend([clone_url, str(repo_dir)])
    subprocess.run(cmd, check=True)
    return repo_dir


def read_text_if_exists(path):
    path = Path(path)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def grep_lines(text, patterns):
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        if any(pattern in lower for pattern in patterns):
            rows.append({"line": line_no, "text": ascii_safe(line.strip())})
    return rows


def find_checkpoint_like_urls(text):
    urls = re.findall(r"https?://[^\s)>\"]+", text)
    return [url for url in urls if any(token in url.lower() for token in ("checkpoint", "pretrain", "model", "drive", "huggingface", "baidu"))]


def ascii_safe(text):
    return text.encode("ascii", errors="replace").decode("ascii")


def write_markdown(path, payload):
    lines = [
        "# UDPNet ConvIR Reproduction Audit",
        "",
        f"- Repo dir: `{payload['repo_dir']}`",
        f"- ConvIR UDP model file: `{payload['convir_udp_model']}`",
        f"- ConvIR UDP model exists: `{payload['convir_udp_model_exists']}`",
        f"- README exists: `{payload['readme_exists']}`",
        "",
        "## README Lines",
        "",
    ]
    if payload["readme_relevant_lines"]:
        for row in payload["readme_relevant_lines"]:
            lines.append(f"- L{row['line']}: {row['text']}")
    else:
        lines.append("- No Haze4K/ConvIR/UDP lines found by the text audit.")
    lines.extend(["", "## Checkpoint-Like URLs", ""])
    if payload["checkpoint_like_urls"]:
        for url in payload["checkpoint_like_urls"]:
            lines.append(f"- {url}")
    else:
        lines.append("- No checkpoint-like URLs found by the text audit.")
    lines.extend(
        [
            "",
            "## Next Manual Eval Checks",
            "",
            "- Confirm official checkpoint availability and license.",
            "- Run the official UDPNet Haze4K eval on the same data root, RGB order, crop/padding, and metrics used by ConvIR-B.",
            "- Compare official UDPNet ConvIR per-image rows against the local A0 baseline and write bucket deltas.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--clone_url", default="https://github.com/Harbinzzy/UDPNet.git")
    parser.add_argument("--clone_if_missing", action="store_true")
    parser.add_argument("--clone_depth", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    repo_dir = maybe_clone(args.repo_dir, args.clone_url, args.clone_if_missing, args.clone_depth)
    readme_path = repo_dir / "README.md"
    convir_udp_model = repo_dir / "Dehazing" / "ITS" / "models" / "ConvIR_UDPNet.py"
    readme = read_text_if_exists(readme_path)
    payload = {
        "repo_dir": str(repo_dir),
        "clone_url": args.clone_url,
        "readme_exists": readme_path.is_file(),
        "convir_udp_model": str(convir_udp_model),
        "convir_udp_model_exists": convir_udp_model.is_file(),
        "readme_relevant_lines": grep_lines(readme, ["haze4k", "convir", "udp", "depth"]),
        "checkpoint_like_urls": find_checkpoint_like_urls(readme),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "udpnet_convir_repro_audit.json"
    md_path = output_dir / "v14_udpnet_repro_audit.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps(payload, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

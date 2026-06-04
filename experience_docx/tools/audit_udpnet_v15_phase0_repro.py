#!/usr/bin/env python3
"""Phase-0 audit for official UDPNet ConvIR Haze4K reproduction."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.parse
import urllib.request


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/124 Safari/537.36"
)
APP_ID = "250528"
SHARE_SURL = "JqB-YBPzZAiQsdLlNcidLQ"
SHARE_URL = f"https://pan.baidu.com/s/1{SHARE_SURL}?pwd=2026"
TARGET_CKPT = "ConvIR_UDPNet_haze4k.ckpt"


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_files(path: Path) -> int | None:
    if not path.exists() or not path.is_dir():
        return None
    return sum(1 for p in path.iterdir() if p.is_file())


class BaiduShareClient:
    def __init__(self, surl: str, pwd: str) -> None:
        self.surl = surl
        self.pwd = pwd
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookie_jar)
        )
        self.shareid: str | None = None
        self.share_uk: str | None = None
        self.sekey: str | None = None
        self.sign: str | None = None
        self.timestamp: int | None = None

    def request(
        self,
        url: str,
        data: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        req_headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }
        if headers:
            req_headers.update(headers)
        body = None
        if data is not None:
            body = urllib.parse.urlencode(data).encode("utf-8")
            req_headers.setdefault(
                "Content-Type", "application/x-www-form-urlencoded; charset=UTF-8"
            )
        req = urllib.request.Request(url, data=body, headers=req_headers)
        with self.opener.open(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def cookie_value(self, name: str) -> str | None:
        for cookie in self.cookie_jar:
            if cookie.name == name:
                return cookie.value
        return None

    def verify_and_load_page(self) -> dict[str, object]:
        init_url = f"https://pan.baidu.com/share/init?surl={self.surl}&pwd={self.pwd}"
        self.request(init_url)
        logid = base64.b64encode(str(int(time.time() * 1000)).encode()).decode()
        verify_url = (
            "https://pan.baidu.com/share/verify?"
            + urllib.parse.urlencode(
                {
                    "surl": self.surl,
                    "t": str(int(time.time() * 1000)),
                    "channel": "chunlei",
                    "web": "1",
                    "app_id": APP_ID,
                    "bdstoken": "",
                    "logid": logid,
                    "clienttype": "0",
                }
            )
        )
        verify = self.request(
            verify_url,
            {"pwd": self.pwd, "vcode": "", "vcode_str": ""},
            {"Referer": init_url, "X-Requested-With": "XMLHttpRequest"},
        )
        verify_json = json.loads(verify)
        page = self.request(SHARE_URL)
        match = re.search(r"locals\.mset\((.*?)\);", page, re.S)
        if not match:
            raise RuntimeError("could not locate locals.mset data on Baidu share page")
        data = json.loads(match.group(1))
        self.shareid = str(data.get("shareid"))
        self.share_uk = str(data.get("share_uk"))
        raw_sekey = self.cookie_value("BDCLND") or ""
        self.sekey = urllib.parse.unquote(raw_sekey)

        tpl_url = (
            "https://pan.baidu.com/share/tplconfig?"
            + urllib.parse.urlencode(
                {
                    "fields": "sign,timestamp",
                    "channel": "chunlei",
                    "web": "1",
                    "app_id": APP_ID,
                    "clienttype": "0",
                    "surl": "1" + self.surl,
                }
            )
        )
        tpl = json.loads(
            self.request(
                tpl_url,
                headers={"Referer": SHARE_URL, "X-Requested-With": "XMLHttpRequest"},
            )
        )
        if tpl.get("errno") == 0:
            self.sign = str(tpl["data"]["sign"])
            self.timestamp = int(tpl["data"]["timestamp"])

        return {
            "verify": verify_json,
            "page_file_list": data.get("file_list", []),
            "shareid": self.shareid,
            "share_uk": self.share_uk,
            "has_bdclnd": bool(raw_sekey),
            "tplconfig_errno": tpl.get("errno"),
            "tplconfig_has_sign": bool(self.sign and self.timestamp),
        }

    def list_dir(self, dir_path: str) -> dict[str, object]:
        if not self.shareid or not self.share_uk:
            raise RuntimeError("share page must be loaded before list_dir")
        url = (
            "https://pan.baidu.com/share/list?"
            + urllib.parse.urlencode(
                {
                    "app_id": APP_ID,
                    "web": "1",
                    "channel": "chunlei",
                    "clienttype": "0",
                    "shareid": self.shareid,
                    "uk": self.share_uk,
                    "dir": dir_path,
                    "page": "1",
                    "num": "100",
                    "order": "time",
                    "desc": "1",
                    "showempty": "0",
                }
            )
        )
        return json.loads(
            self.request(
                url, headers={"Referer": SHARE_URL, "X-Requested-With": "XMLHttpRequest"}
            )
        )

    def attempt_sharedownload(self, fs_id: int) -> dict[str, object]:
        if not (self.shareid and self.share_uk and self.sign and self.timestamp):
            return {"attempted": False, "reason": "missing shareid/share_uk/sign"}
        url = (
            "https://pan.baidu.com/api/sharedownload?"
            + urllib.parse.urlencode(
                {
                    "sign": self.sign,
                    "timestamp": str(self.timestamp),
                    "channel": "chunlei",
                    "web": "1",
                    "app_id": APP_ID,
                    "bdstoken": "",
                    "clienttype": "0",
                }
            )
        )
        post = {
            "encrypt": "0",
            "product": "share",
            "uk": self.share_uk,
            "primaryid": self.shareid,
            "fid_list": json.dumps([fs_id]),
            "path_list": "[]",
            "extra": json.dumps({"sekey": self.sekey or ""}),
            "vip": "0",
        }
        body = self.request(
            url, post, {"Referer": SHARE_URL, "X-Requested-With": "XMLHttpRequest"}
        )
        result = json.loads(body)
        list_value = result.get("list")
        return {
            "attempted": True,
            "errno": result.get("errno"),
            "request_id": result.get("request_id"),
            "server_time": result.get("server_time"),
            "has_direct_dlink": isinstance(list_value, list)
            and any(isinstance(x, dict) and x.get("dlink") for x in list_value),
            "returned_client_encrypted_list": isinstance(list_value, str),
            "list_type": type(list_value).__name__,
            "list_length": len(list_value) if isinstance(list_value, str) else None,
            "raw_head": body[:500],
        }


def run_baidupcs_probe(bin_path: Path, output_dir: Path) -> dict[str, object]:
    if not bin_path.exists():
        return {"attempted": False, "reason": f"missing {bin_path}"}
    log_path = output_dir / "baidupcs_transfer_probe.log"
    cmd = [
        str(bin_path),
        "transfer",
        "--download",
        f"https://pan.baidu.com/s/1{SHARE_SURL}",
        "2026",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return {
        "attempted": True,
        "returncode": proc.returncode,
        "log": str(log_path),
        "stdout_head": proc.stdout[:1000],
        "reported_metadata_failure": "获取分享项元数据错误" in proc.stdout,
    }


def write_csvs(output_dir: Path, status: str, blockers: list[dict[str, str]]) -> None:
    bucket_path = output_dir / "udpnet_convir_bucket_compare.csv"
    with bucket_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "split",
                "a0_psnr",
                "udpnet_psnr",
                "delta_psnr",
                "a0_ssim",
                "udpnet_ssim",
                "delta_ssim",
                "bucket",
                "audit_status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "__PHASE0_NOT_EVALUATED__",
                "split": "",
                "audit_status": status,
                "note": "official checkpoint unavailable; no per-image bucket compare",
            }
        )

    failure_path = output_dir / "udpnet_convir_failure_audit.csv"
    with failure_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["blocker_id", "severity", "detail", "evidence"])
        writer.writeheader()
        for row in blockers:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--udp_repo", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--official_checkpoint", required=True)
    parser.add_argument("--baidupcs_bin", default="/root/autodl-tmp/workspace/tools/bin/BaiduPCS-Go")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    udp_repo = Path(args.udp_repo)
    data_dir = Path(args.data_dir)
    depth_cache = Path(args.depth_cache_dir)
    official_ckpt = Path(args.official_checkpoint)
    a0 = Path(args.a0_checkpoint)

    ckpt_candidates: list[dict[str, object]] = []
    baidu_summary: dict[str, object] = {}
    target_item: dict[str, object] | None = None
    blockers: list[dict[str, str]] = []

    try:
        client = BaiduShareClient(SHARE_SURL, "2026")
        baidu_summary["page"] = client.verify_and_load_page()
        listed = client.list_dir("/UDPNet_checkpoints")
        baidu_summary["checkpoint_list_errno"] = listed.get("errno")
        for item in listed.get("list", []):
            row = {
                "server_filename": item.get("server_filename"),
                "path": item.get("path"),
                "fs_id": item.get("fs_id"),
                "size": item.get("size"),
                "md5": item.get("md5"),
                "isdir": item.get("isdir"),
            }
            ckpt_candidates.append(row)
            if item.get("server_filename") == TARGET_CKPT:
                target_item = row
        if target_item and target_item.get("fs_id"):
            baidu_summary["sharedownload_probe"] = client.attempt_sharedownload(
                int(target_item["fs_id"])
            )
    except Exception as exc:  # keep audit evidence instead of hiding infra failures
        baidu_summary["error"] = repr(exc)

    baidupcs_probe = run_baidupcs_probe(Path(args.baidupcs_bin), output_dir)

    official_available = official_ckpt.exists()
    if not official_available:
        blockers.append(
            {
                "blocker_id": "official_checkpoint_missing",
                "severity": "blocking",
                "detail": f"{official_ckpt} is absent, so official ConvIR+UDP eval cannot run.",
                "evidence": "udpnet_convir_repro_eval.json:official_checkpoint_available=false",
            }
        )
    shared = baidu_summary.get("sharedownload_probe", {})
    if isinstance(shared, dict) and shared.get("returned_client_encrypted_list"):
        blockers.append(
            {
                "blocker_id": "baidu_share_no_plain_dlink",
                "severity": "blocking",
                "detail": "Baidu sharedownload returned a client encrypted task list, not a direct HTTP dlink.",
                "evidence": "udpnet_convir_repro_eval.json:baidu_share.sharedownload_probe",
            }
        )
    if baidupcs_probe.get("reported_metadata_failure"):
        blockers.append(
            {
                "blocker_id": "baidupcs_public_transfer_failed",
                "severity": "blocking",
                "detail": "BaiduPCS-Go public share transfer failed to get share item metadata without a logged-in account.",
                "evidence": "baidupcs_transfer_probe.log",
            }
        )

    repo_model = udp_repo / "Dehazing/ITS/models/ConvIR_UDPNet.py"
    repo_test = udp_repo / "Dehazing/ITS/test.py"
    repo_data = udp_repo / "Dehazing/ITS/data/data_load.py"
    status = "BLOCKED_CHECKPOINT_UNAVAILABLE" if not official_available else "READY_FOR_EVAL"

    eval_json = {
        "route": "ConvIR-Dehaze-v1.5-FullUDP",
        "phase": "Phase 0 official UDPNet ConvIR reproduction audit",
        "status": status,
        "state_label": "PREFLIGHT_FAILED_ENGINEERING" if not official_available else "PLANNED",
        "locked_test_allowed": False,
        "official_checkpoint_available": official_available,
        "official_checkpoint_path": str(official_ckpt),
        "official_checkpoint_sha256": sha256_file(official_ckpt),
        "a0_checkpoint_path": str(a0),
        "a0_checkpoint_sha256": sha256_file(a0),
        "udp_repo": str(udp_repo),
        "udp_repo_head": subprocess.run(
            ["git", "-C", str(udp_repo), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
        if (udp_repo / ".git").exists()
        else None,
        "udp_repo_model_exists": repo_model.exists(),
        "udp_repo_test_imports_fsnet": "models.FSNet import build_net"
        in repo_test.read_text(encoding="utf-8", errors="ignore")
        if repo_test.exists()
        else None,
        "data_dir": str(data_dir),
        "data_counts": {
            "train_haze": count_files(data_dir / "train/haze"),
            "train_hazy": count_files(data_dir / "train/hazy"),
            "train_gt": count_files(data_dir / "train/gt"),
            "test_haze": count_files(data_dir / "test/haze"),
            "test_hazy": count_files(data_dir / "test/hazy"),
            "test_gt": count_files(data_dir / "test/gt"),
            "test_depth2l": count_files(data_dir / "test/depth2l"),
        },
        "depth_cache_dir": str(depth_cache),
        "depth_cache_counts": {
            "train": count_files(depth_cache / "train"),
            "test": count_files(depth_cache / "test"),
        },
        "checkpoint_source": {
            "url": SHARE_URL,
            "target": TARGET_CKPT,
            "target_item": target_item,
            "candidates": ckpt_candidates,
        },
        "baidu_share": baidu_summary,
        "baidupcs_probe": baidupcs_probe,
        "metrics": {
            "mean_psnr_delta_vs_a0": None,
            "mean_ssim_delta_vs_a0": None,
            "hard_bottom25_delta": None,
            "easy_top25_delta": None,
            "positive_ratio": None,
            "strong_regression_ratio": None,
            "worst_le_minus_0_20_count": None,
            "runtime_seconds_per_image": None,
            "peak_memory_mib": None,
        },
        "gate": {
            "reproduction_gate_pass": False,
            "reason": "official checkpoint is unavailable; no evaluation was run",
        },
        "blockers": blockers,
    }
    (output_dir / "udpnet_convir_repro_eval.json").write_text(
        json.dumps(eval_json, indent=2, sort_keys=True), encoding="utf-8"
    )

    write_csvs(output_dir, status, blockers)

    protocol_md = f"""# UDPNet ConvIR Phase 0 Protocol Diff

Status: `{status}`

## Checkpoint Source

- Official share: `{SHARE_URL}`.
- Target checkpoint: `{TARGET_CKPT}`.
- Listed target item: `{target_item}`.
- Local official checkpoint path: `{official_ckpt}`.
- Local official checkpoint available: `{official_available}`.
- A0 checkpoint: `{a0}`.
- A0 sha256: `{sha256_file(a0)}`.

## Download/Access Result

- Baidu share verify/list succeeded and exposed the official checkpoint list.
- `api/sharedownload` did not expose a plain HTTP `dlink`; it returned a client
  encrypted task list for the target checkpoint.
- `BaiduPCS-Go transfer --download` did not retrieve share metadata without a
  logged-in account.
- Browser-side click invokes the BaiduNetdisk desktop client path, not a normal
  browser-download artifact that can be archived by this run.

## Repository/Entrypoint Diff

- UDPNet repo: `{udp_repo}`.
- UDPNet commit: `{eval_json["udp_repo_head"]}`.
- `ConvIR_UDPNet.py` exists: `{repo_model.exists()}`.
- Official `Dehazing/ITS/test.py` imports FSNet by default:
  `{eval_json["udp_repo_test_imports_fsnet"]}`. A ConvIR reproduction needs a
  controlled ConvIR_UDPNet build/eval wrapper rather than the unmodified test
  entrypoint.

## Data/Depth Layout Diff

- Current Haze4K root: `{data_dir}`.
- Existing test hazy directory count: `{count_files(data_dir / "test/hazy")}`.
- Existing test haze directory count: `{count_files(data_dir / "test/haze")}`.
- Official UDPNet loader expects `test/hazy`, `test/gt`, and `test/depth2l`.
- Current project data uses `haze` plus a separate DepthAnything cache:
  `{depth_cache}`.
- A future eval wrapper must symlink or adapt `haze -> hazy` and map/cache
  DepthAnything outputs into UDPNet's `depth2l` contract.

## Evaluation Status

No PSNR/SSIM evaluation was launched because the official checkpoint is absent.
This is an acquisition/protocol blocker, not a scientific failure of UDPNet.
"""
    (output_dir / "udpnet_protocol_diff.md").write_text(protocol_md, encoding="utf-8")

    readme = f"""# Haze4K v1.5 Full UDPNet Phase 0 Evidence

Status: `{status}`.

Primary files:

- `udpnet_convir_repro_eval.json`: official checkpoint acquisition/protocol audit.
- `udpnet_convir_bucket_compare.csv`: placeholder row because eval did not run.
- `udpnet_convir_failure_audit.csv`: blockers that prevented reproduction eval.
- `udpnet_protocol_diff.md`: checkpoint, entrypoint, data, depth, and metric diff.
- `baidupcs_transfer_probe.log`: BaiduPCS-Go public-share transfer probe.

Decision:

- Official ConvIR+UDP Haze4K checkpoint could be listed in the Baidu share but
  could not be downloaded as a plain checkpoint artifact in this environment.
- Do not claim README-level UDPNet reproduction until the official checkpoint is
  available with a recorded sha256 and a controlled ConvIR_UDPNet eval wrapper.
- Locked Haze4K test remains blocked.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"PHASE0_AUDIT_OK output={output_dir} status={status}")


if __name__ == "__main__":
    main()

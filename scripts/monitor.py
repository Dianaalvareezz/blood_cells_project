"""Monitor a Kaggle training kernel and feed the HTML dashboard.

Polls the kernel status and log, extracts the per-epoch `METRICS {...}` lines written by the
training script, and writes the job's live JSON for the dashboard. When the kernel finishes it
downloads the outputs and builds the reports.

Jobs (--job):
  detector (default)  kaggle/train/train.py          -> dashboard/data/live.json,     runs/kaggle/
  wbc                 kaggle/train_wbc/train_wbc.py  -> dashboard/data/live_wbc.json, runs/kaggle_wbc/

Usage:
  .venv/bin/python scripts/monitor.py              # poll + serve dashboard on http://localhost:8000
  .venv/bin/python scripts/monitor.py --job wbc    # same, for the white-blood-cell classifier
  .venv/bin/python scripts/monitor.py --once       # single poll, no server
"""
import argparse
import functools
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
JOBS = {
    "detector": dict(kernel="dianalvareezz/blood-cells-yolo26-train", live="live.json", out="kaggle",
                     report="make_reports.py", metric=("metrics/mAP50-95", "mAP50-95")),
    "wbc": dict(kernel="dianalvareezz/blood-cells-wbc-classifier-train", live="live_wbc.json",
                out="kaggle_wbc", report="make_wbc_reports.py", metric=("metrics/accuracy_top1", "top-1 acc")),
}
# Set by select_job(); the detector job is the default.
KERNEL = LIVE = OUT = REPORT = METRIC = None


def select_job(name):
    global KERNEL, LIVE, OUT, REPORT, METRIC
    job = JOBS[name]
    KERNEL, METRIC = job["kernel"], job["metric"]
    LIVE = ROOT / "dashboard" / "data" / job["live"]
    OUT = ROOT / "runs" / job["out"]
    REPORT = ROOT / "scripts" / job["report"]


select_job("detector")
TAG_RE = re.compile(r"^(START|METRICS|FINAL) (\{.*\})\s*$")


def load_token():
    """Accept either a bare token or KEY=VALUE lines in .env."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    text = (ROOT / ".env").read_text().strip()
    for line in text.splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip() in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_TOKEN"):
                os.environ["KAGGLE_API_TOKEN"] = v.strip().strip("'\"")
                return
        elif line:
            os.environ["KAGGLE_API_TOKEN"] = line
            return
    sys.exit("No Kaggle token found in .env")


def kaggle(*args):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    return r.stdout + r.stderr


def get_status():
    out = kaggle("kernels", "status", KERNEL)
    m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', out)
    return (m.group(1).upper() if m else "UNKNOWN"), out.strip()


def get_events():
    """Return (start, epochs, final) parsed from the kernel log."""
    raw = kaggle("kernels", "logs", KERNEL)
    text = ""
    try:
        start = raw.index("[")
        for item in json.loads(raw[start:]):
            if item.get("stream_name") == "stdout":
                text += item.get("data", "")
    except (ValueError, json.JSONDecodeError):
        text = raw
    start, epochs, final = None, {}, None
    for line in text.replace("\r", "\n").splitlines():
        m = TAG_RE.match(line.strip())
        if not m:
            continue
        payload = json.loads(m.group(2))
        if m.group(1) == "START":
            start = payload
        elif m.group(1) == "METRICS":
            # Ultralytics fires the callback once more during final validation; skip it.
            if payload["epoch"] <= payload["epochs"]:
                epochs[payload["epoch"]] = payload
        else:
            final = payload
    return start, [epochs[k] for k in sorted(epochs)], final


def write_live(state):
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LIVE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(LIVE)


def download_and_report():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Downloading kernel outputs ...", flush=True)
    kaggle("kernels", "output", KERNEL, "-p", str(OUT), "-o", "-q")
    subprocess.run([sys.executable, str(REPORT)], check=True)


def poll(prev=None):
    status, raw = get_status()
    start, epochs, final = get_events()
    prev = prev or {}
    state = {
        "kernel": KERNEL,
        "kernel_url": f"https://www.kaggle.com/code/{KERNEL}",
        "status": status,
        "status_raw": raw,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "first_seen_running": prev.get("first_seen_running"),
        "start": start or prev.get("start"),
        # keep what we already had if the log is temporarily unavailable
        "epochs": epochs or prev.get("epochs", []),
        "final": final or prev.get("final"),
        "live_log_available": bool(epochs),
    }
    if status == "RUNNING" and not state["first_seen_running"]:
        state["first_seen_running"] = state["updated_at"]
    write_live(state)
    last = state["epochs"][-1] if state["epochs"] else None
    msg = f"[{state['updated_at']}] {status}"
    if last:
        msg += f" · epoch {last['epoch']}/{last['epochs']} · {METRIC[1]} {last.get(METRIC[0], 0):.3f}"
    print(msg, flush=True)
    return state


def serve(port):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    handler = functools.partial(Quiet, directory=str(ROOT))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Dashboard: http://localhost:{port}/dashboard/", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=sorted(JOBS), default="detector")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()
    select_job(args.job)
    load_token()

    prev = json.loads(LIVE.read_text()) if LIVE.exists() else None
    if args.once:
        state = poll(prev)
        if state["status"] == "COMPLETE":
            download_and_report()
        return

    if not args.no_serve:
        serve(args.port)
    while True:
        prev = poll(prev)
        if prev["status"] in ("COMPLETE", "ERROR", "CANCELACKNOWLEDGED", "CANCELREQUESTED"):
            if prev["status"] == "COMPLETE":
                download_and_report()
            print(f"Kernel finished with status {prev['status']}.", flush=True)
            if args.no_serve:
                return
            print("Still serving the dashboard; Ctrl+C to stop.", flush=True)
            while True:
                time.sleep(3600)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

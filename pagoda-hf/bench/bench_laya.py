#!/usr/bin/env python3
"""Reference-implementation benchmark for Laya (PyTorch), mirroring
pagoda-hf/examples/bench_laya.rs run-for-run on the same machine.

The Laya repo ships its own reference implementation (`rl_agent_api.py` +
`rl_common.py`, Apache-2.0). We do not vendor them; on first run they are
downloaded next to this script from the model repo (same snapshot family as
the weights under test).

Usage:
    python bench_laya.py --snapshot <path-to-laya-snapshot> [--device cpu|cuda] [--iters 50] [--warmup 5]

Prints one JSON line: load time, latency percentiles, peak RSS.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

_REF_FILES = ("rl_agent_api.py", "rl_common.py")
_REF_BASE = "https://huggingface.co/convaiinnovations/laya/resolve/main/"

for _name in _REF_FILES:
    _dst = os.path.join(_HERE, _name)
    if not os.path.exists(_dst):
        print(f"fetching reference implementation: {_name}", file=sys.stderr)
        urllib.request.urlretrieve(_REF_BASE + _name, _dst)

import torch  # noqa: E402

from rl_agent_api import RLAgent  # noqa: E402

STATE = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "other": "everything else"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
}


def peak_rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def pct(sorted_vals, p):
    idx = round((len(sorted_vals) - 1) * p)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    t0 = time.perf_counter()
    agent = RLAgent(args.snapshot, device=args.device)
    load_s = time.perf_counter() - t0

    for _ in range(args.warmup):
        agent.system_one(STATE, QUESTIONS)

    lat = []
    dept = None
    probs = None
    for _ in range(args.iters):
        t = time.perf_counter()
        out = agent.system_one(STATE, QUESTIONS)
        lat.append((time.perf_counter() - t) * 1000.0)
        dept = out["answers"]["department"]["choice"]
        probs = out["answers"]["department"]["probabilities"]
    assert dept == "billing", f"sanity: department must route to billing, got {dept}"

    lat.sort()
    print(json.dumps({
        "impl": "laya-python",
        "device": args.device,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "load_s": round(load_s, 3),
        "mean_ms": round(statistics.fmean(lat), 2),
        "p50_ms": round(pct(lat, 0.50), 2),
        "p95_ms": round(pct(lat, 0.95), 2),
        "min_ms": round(lat[0], 2),
        "max_ms": round(lat[-1], 2),
        "peak_rss_mb": round(peak_rss_kb() / 1024.0),
        "iters": args.iters,
        "dept_probs": probs,
    }))


if __name__ == "__main__":
    main()
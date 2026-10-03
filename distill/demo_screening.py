#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screening factory demo: A/B (base vs tuned Laya in Rust) + extraction pipeline.

Phase A: screen all seed abstracts through BOTH laya_server instances
         (base :31180, tuned :31181) and score accuracy + latency.
Phase B: for abstracts the tuned model includes, run a structured-extraction
         request through the pagoda gateway (:31100) with a long SHARED PICO
         prefix, so the radix tree reuses the prefix KV across requests.
Phase C: read /stats from the gateway and emit results-screening.json.
"""
import json, sys, time, urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:31180"
TUNED = "http://127.0.0.1:31181"
GATEWAY = "http://127.0.0.1:31100"
SEEDS = Path(__file__).parent / "seeds" / "abstracts.jsonl"
OUT = Path(__file__).parent / "results-screening.json"

NOUL_INSTR = ("Answer whether the statement about the scientific abstract is true. "
              "Statement: the study is a randomized controlled trial in adults with type 2 "
              "diabetes comparing a structured exercise intervention (aerobic, resistance, "
              "or combined training) against usual care or no exercise, and reports HbA1c "
              "(glycated hemoglobin) as an outcome.")
CHOICE_CRITERIA = {
    "rct": "randomized controlled trial with random allocation of participants",
    "cohort": "observational cohort or case-control study without randomization",
    "review": "narrative or systematic review, meta-analysis, or pooled analysis",
    "invitro": "in-vitro, animal, or mechanistic laboratory study",
    "other": "other design: cross-sectional, case report, quasi-randomized, conference abstract",
}

EXTRACT_PREFIX = """You are a data-extraction assistant for a systematic review.
PICO: adults with type 2 diabetes; structured exercise vs usual care; outcome HbA1c; RCTs only.
From the abstract below, extract a JSON object with exactly these fields:
  "n_participants": integer total randomized,
  "exercise_type": one of "aerobic" | "resistance" | "combined" | "unclear",
  "duration_weeks": integer program length,
  "hba1c_change_exercise": float change in the exercise arm (percent points),
  "hba1c_change_control": float change in the control arm (percent points).
If a field is not reported, use null. Reply with the JSON object only.

Abstract:
"""


def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), (time.time() - t0) * 1000


def screen(server, state):
    payload = {
        "state": state,
        "questions": {
            "include": {"type": "noul", "instructions": NOUL_INSTR},
            "design": {"type": "choice", "instructions":
                       "Classify the study design of the scientific abstract described in the text.",
                       "criteria": CHOICE_CRITERIA},
        },
    }
    resp, ms = post(server + "/decide", payload)
    a = resp["answers"]
    p_inc = a["include"]["noul"]
    return dict(include=bool(p_inc is not None and p_inc > 0.5),
                include_p=p_inc,
                include_conf=a["include"]["confidence"],
                design=a["design"]["choice"],
                design_conf=a["design"]["confidence"],
                input_tokens=resp["usage"]["input_tokens"], ms=ms)


def acc(rows, key, borderline=None):
    sel = [r for r in rows if borderline is None or r["borderline"] == borderline]
    if not sel:
        return float("nan")
    return sum(r["pred_" + key] == r["gold_" + key] for r in sel) / len(sel)


def main():
    seeds = [json.loads(l) for l in SEEDS.read_text().splitlines() if l.strip()]
    pico = next(s["pico"] for s in seeds if "pico" in s)
    seeds = [s for s in seeds if "id" in s]
    print(f"[A] screening {len(seeds)} abstracts on base(:31180) vs tuned(:31181)")

    rows = []
    t0 = time.time()
    for i, s in enumerate(seeds):
        state = f"Title: {s['title']}\nAbstract: {s['abstract']}"
        b = screen(BASE, state)
        t = screen(TUNED, state)
        rows.append(dict(id=s["id"], borderline=bool(s.get("borderline")),
                         gold_include=s["include"], gold_design=s["design"],
                         pred_include=t["include"], pred_design=t["design"],
                         base_include=b["include"], base_design=b["design"],
                         tuned_conf=t["include_conf"], base_conf=b["include_conf"],
                         tuned_ms=t["ms"], base_ms=b["ms"],
                         input_tokens=t["input_tokens"]))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(seeds)} done ({time.time()-t0:.0f}s)")
    wall = time.time() - t0

    base_rows = [dict(r, pred_include=r["base_include"], pred_design=r["base_design"]) for r in rows]
    res = {
        "n": len(rows),
        "n_borderline": sum(r["borderline"] for r in rows),
        "wall_seconds": round(wall, 1),
        "per_item_ms_tuned": round(sum(r["tuned_ms"] for r in rows) / len(rows), 1),
        "per_item_ms_base": round(sum(r["base_ms"] for r in rows) / len(rows), 1),
        "base": dict(noul=round(acc(base_rows, "include"), 3),
                     borderline=round(acc(base_rows, "include", True), 3),
                     design=round(acc(base_rows, "design"), 3)),
        "tuned": dict(noul=round(acc(rows, "include"), 3),
                      borderline=round(acc(rows, "include", True), 3),
                      design=round(acc(rows, "design"), 3)),
    }
    print(f"[A] base   : {res['base']}")
    print(f"[A] tuned  : {res['tuned']}")
    print(f"[A] wall {wall:.1f}s, tuned {res['per_item_ms_tuned']} ms/abstract "
          f"(two questions)")

    # ---- Phase B: extraction through the gateway with a shared PICO prefix ----
    included = [r for r in rows if r["pred_include"]]
    print(f"[B] {len(included)} included -> structured extraction via gateway :31100")
    ok, failed = 0, 0
    lat, usage = [], []
    t0 = time.time()
    for r in included:
        s = next(x for x in seeds if x["id"] == r["id"])
        prompt = EXTRACT_PREFIX + f"Title: {s['title']}\n{s['abstract']}"
        try:
            resp, ms = post(GATEWAY + "/v1/chat/completions", {
                "model": "student",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 160, "temperature": 0.0,
            })
            txt = resp["choices"][0]["message"]["content"]
            json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            ok += 1
            lat.append(ms)
            usage.append(resp.get("usage", {}))
        except Exception as e:
            failed += 1
            print(f"    [B] {r['id']}: {type(e).__name__}: {e}")
    extract_wall = time.time() - t0
    print(f"[B] extraction ok={ok} failed={failed} in {extract_wall:.1f}s")
    if lat:
        print(f"[B] latency: first={lat[0]:.0f}ms  rest_avg={sum(lat[1:])/max(1,len(lat)-1):.0f}ms"
              f"  min={min(lat):.0f}ms")
        print(f"[B] usage[0]={usage[0]}")

    # ---- Phase C: gateway stats ----
    try:
        with urllib.request.urlopen(GATEWAY + "/stats", timeout=10) as r:
            stats = json.loads(r.read())
        print(f"[C] gateway /stats: {json.dumps(stats)}")
    except Exception as e:
        stats = {"error": str(e)}
        print(f"[C] /stats unavailable: {e}")

    res["extraction"] = dict(included=len(included), ok=ok, failed=failed,
                             wall_seconds=round(extract_wall, 1),
                             latencies_ms=[round(x) for x in lat])
    res["gateway_stats"] = stats
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()

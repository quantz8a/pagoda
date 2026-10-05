#!/usr/bin/env python3
# Real-world validation: screen real PubMed abstracts (Umpierre 2011 pool)
# through base (:31180) and tuned (:31181) Laya, score vs human gold standard.
import json, time, urllib.request, urllib.parse, random, sys
import xml.etree.ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BASE = "http://127.0.0.1:31180"
TUNED = "http://127.0.0.1:31181"
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

def get(url, retries=4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception as e:
            print("retry", i, str(e)[:80], flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError("fetch failed")

def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), (time.time() - t0) * 1000

def efetch(pmids):
    u = (EUTILS + "efetch.fcgi?db=pubmed&retmode=xml&id=" + ",".join(pmids))
    return ET.fromstring(get(u))

def pubtypes_of(pmids):
    out = {}
    for i in range(0, len(pmids), 100):
        root = efetch(pmids[i:i+100])
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            pts = [pt.text for pt in art.iter("PublicationType") if pt.text]
            out[pmid] = pts
        time.sleep(0.5)
    return out

def abstracts_of(pmids):
    out = {}
    for i in range(0, len(pmids), 100):
        root = efetch(pmids[i:i+100])
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            parts = []
            for ab in art.iter("AbstractText"):
                label = ab.get("Label")
                txt = "".join(ab.itertext())
                parts.append((label + ": " if label else "") + txt)
            out[pmid] = {"title": title, "abstract": "\n".join(parts)}
        time.sleep(0.5)
    return out

def screen(server, abstract):
    payload = {"state": abstract,
               "questions": {"include": {"type": "noul", "instructions": NOUL_INSTR},
                             "design": {"type": "choice",
                                        "instructions": "Classify the study design of the scientific abstract described in the text.",
                                        "criteria": CHOICE_CRITERIA}}}
    resp, ms = post(server + "/decide", payload)
    a = resp["answers"]
    p = a["include"]["noul"]
    return {"include": bool(p is not None and p > 0.5), "p": p,
            "design": a["design"]["choice"], "ms": round(ms)}

data = json.load(open("/tmp/umpierre_pool.json"))
pool, candidates = data["pool"], data["gold_positives"]
print("pool:", len(pool), "candidate included:", len(candidates), flush=True)

print("== fetching publication types for candidates ==", flush=True)
pts = pubtypes_of(candidates)
RCT_TYPES = {"Randomized Controlled Trial", "Controlled Clinical Trial", "Clinical Trial"}
positives = [p for p in candidates if pts.get(p) and set(pts[p]) & RCT_TYPES]
excluded_by_type = [(p, pts.get(p)) for p in candidates if p not in positives]
print("gold positives (RCT pubtype):", len(positives))
print("dropped (not RCT pubtype):")
for p, t in excluded_by_type:
    print("   ", p, t)

random.seed(42)
negs = random.sample(sorted(set(pool) - set(positives)), 250)
print("negatives sampled:", len(negs), flush=True)

allids = positives + negs
gold = {p: 1 for p in positives}
gold.update({p: 0 for p in negs})
print("== fetching abstracts ==", flush=True)
abs_ = abstracts_of(allids)
noabs = [p for p in allids if not abs_.get(p, {}).get("abstract")]
print("missing abstracts (dropped):", len(noabs), noabs[:10])
eval_ids = [p for p in allids if p not in noabs]
print("eval set:", len(eval_ids), "= pos", sum(gold[p] for p in eval_ids), "+ neg", sum(1-gold[p] for p in eval_ids), flush=True)

rows = []
t_start = time.time()
for idx, pmid in enumerate(eval_ids):
    text = abs_[pmid]["title"] + "\n" + abs_[pmid]["abstract"]
    tb = screen(BASE, text)
    tt = screen(TUNED, text)
    rows.append({"pmid": pmid, "gold": gold[pmid], "title": abs_[pmid]["title"][:120],
                 "base": tb, "tuned": tt})
    if (idx + 1) % 20 == 0:
        el = time.time() - t_start
        print("  progress %d/%d  elapsed %.0fs" % (idx + 1, len(eval_ids), el), flush=True)

def metrics(rows, key):
    tp = sum(1 for r in rows if r["gold"] == 1 and r[key]["include"])
    fn = sum(1 for r in rows if r["gold"] == 1 and not r[key]["include"])
    tn = sum(1 for r in rows if r["gold"] == 0 and not r[key]["include"])
    fp = sum(1 for r in rows if r["gold"] == 0 and r[key]["include"])
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    acc = (tp + tn) / len(rows)
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "accuracy": round(acc, 4), "ppv": round(ppv, 4)}

result = {"review": "Umpierre 2011 JAMA (PMID 21540423)", "pool_size": len(pool),
          "n_eval": len(eval_ids), "n_pos": sum(gold[p] for p in eval_ids),
          "n_neg": sum(1 - gold[p] for p in eval_ids),
          "missing_abstracts": len(noabs),
          "base": metrics(rows, "base"), "tuned": metrics(rows, "tuned"),
          "rows": rows}
json.dump(result, open("/tmp/umpierre_validation.json", "w"), ensure_ascii=False, indent=1)
print("\n==== RESULTS (real PubMed gold standard) ====")
print("base :", json.dumps(result["base"]))
print("tuned:", json.dumps(result["tuned"]))
print("saved /tmp/umpierre_validation.json")

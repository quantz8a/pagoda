#!/usr/bin/env python3
# Re-validate the v2 checkpoint on the SAME untouched eval set (from
# umpierre_validation.json), then print v1-base / v1-tuned / v2 comparison
# plus a threshold sweep (pick best t for sensitivity>=0.95).
import json, time, urllib.request, sys

V2 = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:31182"
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

def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

# We stored only titles in v1 results -> refetch abstracts for eval PMIDs
d = json.load(open("/tmp/umpierre_validation.json"))
evals = [(r["pmid"], r["gold"]) for r in d["rows"]]
import xml.etree.ElementTree as ET
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
def abstracts_of(pmids):
    out = {}
    for i in range(0, len(pmids), 100):
        for _try in range(4):
            try:
                with urllib.request.urlopen(EUTILS + "efetch.fcgi?db=pubmed&retmode=xml&id=" + ",".join(pmids[i:i+100]), timeout=90) as r:
                    root = ET.fromstring(r.read())
                break
            except Exception as e:
                print("retry", str(e)[:60], flush=True); time.sleep(4)
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            t = art.find(".//ArticleTitle")
            title = "".join(t.itertext()) if t is not None else ""
            parts = []
            for ab in art.iter("AbstractText"):
                label = ab.get("Label"); txt = "".join(ab.itertext())
                parts.append((label + ": " if label else "") + txt)
            out[pmid] = title + "\n" + "\n".join(parts)
        time.sleep(0.4)
    return out

absmap = abstracts_of([p for p, _ in evals])
rows = []
t0 = time.time()
for idx, (pmid, gold) in enumerate(evals):
    payload = {"state": absmap[pmid],
               "questions": {"include": {"type": "noul", "instructions": NOUL_INSTR},
                             "design": {"type": "choice",
                                        "instructions": "Classify the study design of the scientific abstract described in the text.",
                                        "criteria": CHOICE_CRITERIA}}}
    resp = post(V2 + "/decide", payload)
    a = resp["answers"]
    rows.append({"pmid": pmid, "gold": gold, "p": a["include"]["noul"],
                 "design": a["design"]["choice"]})
    if (idx + 1) % 20 == 0:
        print("progress %d/%d (%.0fs)" % (idx + 1, len(evals), time.time() - t0), flush=True)

json.dump(rows, open("/tmp/umpierre_validation_v2.json", "w"))

def metrics(ps):
    tp = sum(1 for g, p in ps if g == 1 and p > 0.5); fn = sum(1 for g, p in ps if g == 1 and p <= 0.5)
    tn = sum(1 for g, p in ps if g == 0 and p <= 0.5); fp = sum(1 for g, p in ps if g == 0 and p > 0.5)
    return tp, fn, tn, fp

ps = [(r["gold"], r["p"] or 0) for r in rows]
tp, fn, tn, fp = metrics(ps)
print("\n== v2 @ t=0.5 ==")
print("sens %.3f  spec %.3f  acc %.3f  (tp %d fn %d tn %d fp %d)" % (
    tp/(tp+fn), tn/(tn+fp), (tp+tn)/len(ps), tp, fn, tn, fp))

best = None
for i in range(2, 99, 2):
    t = i / 100
    tp2 = sum(1 for g, p in ps if g == 1 and p > t); fn2 = sum(1 for g, p in ps if g == 1 and p <= t)
    tn2 = sum(1 for g, p in ps if g == 0 and p <= t); fp2 = sum(1 for g, p in ps if g == 0 and p > t)
    sens = tp2 / (tp2 + fn2); spec = tn2 / (tn2 + fp2)
    if sens >= 0.95 and (best is None or spec > best[2]):
        best = (t, sens, spec)
print("best t for sens>=0.95:", best)
print("\n== v1 reference @ t=0.5 ==")
print("base : sens 0.941 spec 0.278 acc 0.358")
print("tuned: sens 0.471 spec 0.890 acc 0.839")

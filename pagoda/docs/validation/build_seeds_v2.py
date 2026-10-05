#!/usr/bin/env python3
# Build real-distribution training seeds from the Umpierre pool (1975),
# EXCLUDING all 279 eval PMIDs (no leakage). Silver labels via PublicationType.
import json, os, time, urllib.request, random
import xml.etree.ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
random.seed(7)

def get(url, retries=4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception as e:
            print("retry", i, str(e)[:80], flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError("fetch failed")

def efetch(pmids):
    return ET.fromstring(get(EUTILS + "efetch.fcgi?db=pubmed&retmode=xml&id=" + ",".join(pmids)))

pool = json.load(open("/tmp/umpierre_pool.json"))["pool"]
val_rows = json.load(open("/tmp/umpierre_validation.json"))["rows"]
eval_ids = set(r["pmid"] for r in val_rows)
cand = [p for p in pool if p not in eval_ids]
print("pool:", len(pool), "eval excluded:", len(eval_ids), "candidates:", len(cand), flush=True)

# fetch pubtypes + abstracts for ALL candidates (17 batches)
info = {}
for i in range(0, len(cand), 100):
    root = efetch(cand[i:i+100])
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//PMID")
        pts = [pt.text for pt in art.iter("PublicationType") if pt.text]
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""
        parts = []
        for ab in art.iter("AbstractText"):
            label = ab.get("Label")
            txt = "".join(ab.itertext())
            parts.append((label + ": " if label else "") + txt)
        info[pmid] = {"pubtypes": pts, "title": title, "abstract": "\n".join(parts)}
    print("fetched", min(i + 100, len(cand)), "/", len(cand), flush=True)
    time.sleep(0.4)

RCT = {"Randomized Controlled Trial", "Controlled Clinical Trial", "Clinical Trial"}
REV = {"Review", "Systematic Review", "Meta-Analysis"}
def design_of(pts):
    s = set(pts)
    if s & RCT: return "rct"
    if s & REV: return "review"
    if s & {"Observational Study", "Cohort Study", "Case-Control Study"}: return "cohort"
    return "other"

have_abs = [p for p in cand if info.get(p, {}).get("abstract")]
rcts  = [p for p in have_abs if design_of(info[p]["pubtypes"]) == "rct"]
revs  = [p for p in have_abs if design_of(info[p]["pubtypes"]) == "review"]
cohs  = [p for p in have_abs if design_of(info[p]["pubtypes"]) == "cohort"]
oths  = [p for p in have_abs if design_of(info[p]["pubtypes"]) == "other"]
print("with abstracts:", len(have_abs), "| rct:", len(rcts), "review:", len(revs),
      "cohort:", len(cohs), "other:", len(oths))

# sample: 96 positives (rct) + 64 negatives (24 review + 20 cohort + 20 other)
random.shuffle(rcts); random.shuffle(revs); random.shuffle(cohs); random.shuffle(oths)
sel = ([(p, 1) for p in rcts[:96]] + [(p, 0) for p in revs[:24]]
       + [(p, 0) for p in cohs[:20]] + [(p, 0) for p in oths[:20]])
print("selected:", len(sel), "= pos 96 + neg 64")

PICO = "PICO: adults with type 2 diabetes; structured exercise vs usual care; outcome HbA1c; RCTs only. (REAL-DISTRIBUTION v2: PubMed silver labels via PublicationType; eval set excluded)"
lines = [json.dumps({"pico": PICO})]
for pmid, inc in sel:
    d = design_of(info[pmid]["pubtypes"])
    lines.append(json.dumps({"id": "pm" + pmid, "include": bool(inc), "design": d,
                             "title": info[pmid]["title"], "abstract": info[pmid]["abstract"]}))
open(os.path.join(os.environ.get("PAGODA_HOME", os.path.expanduser("~/pagoda")), "distill/seeds/abstracts-real-v2.jsonl"), "w").write("\n".join(lines) + "\n")
print("saved seeds/abstracts-real-v2.jsonl,", len(lines) - 1, "seeds")

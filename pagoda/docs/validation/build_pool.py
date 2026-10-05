#!/usr/bin/env python3
# Reconstruct Umpierre 2011 search pool on PubMed (through March 2011),
# intersect with references -> gold positives; sample negatives.
import json, time, urllib.request, urllib.parse, random

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def get_json(url, retries=4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            print("retry", i, e)
            time.sleep(3 * (i + 1))
    raise RuntimeError("failed")

QUERY = ('("diabetes mellitus, type 2"[MeSH Terms]) AND '
         '("exercise"[MeSH Terms] OR "exercise therapy"[MeSH Terms] OR "resistance training"[MeSH Terms] '
         'OR exercise[Title/Abstract] OR aerobic[Title/Abstract] OR resistance[Title/Abstract] '
         'OR "physical activity"[Title/Abstract] OR training[Title/Abstract]) AND '
         '("glycated hemoglobin"[MeSH Terms] OR hba1c[Title/Abstract] OR "glycated hemoglobin"[Title/Abstract] '
         'OR glycosylated[Title/Abstract] OR "hemoglobin a1c"[Title/Abstract]) AND '
         '("1900/01/01"[Date - Publication] : "2011/03/31"[Date - Publication])')

u = EUTILS + "esearch.fcgi?db=pubmed&retmode=json&retmax=0&term=" + urllib.parse.quote(QUERY)
res = get_json(u)
total = int(res["esearchresult"]["count"])
print("pool size:", total)

u2 = EUTILS + "esearch.fcgi?db=pubmed&retmode=json&retmax=100000&term=" + urllib.parse.quote(QUERY)
res2 = get_json(u2)
pool = set(res2["esearchresult"]["idlist"])
print("fetched pmids:", len(pool))

refs = json.load(open("/tmp/umpierre_refs.json"))["refs_with_pmid"]
ref_pmids = {v["pmid"]: v for v in refs.values()}
inter = sorted(set(ref_pmids) & pool)
print("refs in pool (candidate included trials):", len(inter))
for p in inter:
    print("  ", p, ref_pmids[p]["year"], (ref_pmids[p]["title"] or "")[:70])

random.seed(42)
pool_list = sorted(pool)
json.dump({"query": QUERY, "pool": pool_list, "gold_positives": inter},
          open("/tmp/umpierre_pool.json", "w"))
print("saved /tmp/umpierre_pool.json")

#!/usr/bin/env python3
# Fetch Umpierre 2011 (PMID 21540423) metadata + references from OpenAlex,
# resolve referenced works to PMIDs, save to /tmp/umpierre_refs.json
import json, time, urllib.request

def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pagoda-validation (mailto:dev@example.org)"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read())
        except Exception as e:
            print("retry", i, url[:90], e)
            time.sleep(3 * (i + 1))
    raise RuntimeError("failed: " + url)

w = get("https://api.openalex.org/works/pmid:21540423")
print("title:", w["title"])
print("year:", w["publication_year"])
refs = w.get("referenced_works", [])
print("referenced_works:", len(refs))

# resolve each openalex id -> pmid (batch by filter:ids.openalex:...)
pmids = {}
short = [r.rsplit("/", 1)[-1] for r in refs]
for i in range(0, len(short), 50):
    chunk = short[i:i+50]
    q = "|".join(chunk)
    res = get("https://api.openalex.org/works?filter=ids.openalex:" + q + "&per_page=50&select=id,ids,title,publication_year")
    for it in res.get("results", []):
        pmid = (it.get("ids") or {}).get("pmid")
        if pmid:
            pmids[it["id"].rsplit("/",1)[-1]] = {
                "pmid": pmid.replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip("/"),
                "title": it.get("title"), "year": it.get("publication_year")}
    time.sleep(1)

out = {"review_title": w["title"], "review_pmid": "21540423",
       "n_refs": len(refs), "refs_with_pmid": pmids}
json.dump(out, open("/tmp/umpierre_refs.json", "w"), ensure_ascii=False, indent=1)
print("resolved PMIDs:", len(pmids))

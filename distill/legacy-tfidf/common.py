"""Shared constants and tiny text features for the distill demo.

The distilled student is deliberately a *non-autoregressive* decision head:
one forward over the full text -> probabilities. There is no prefill/decode
phase — that concept belongs to autoregressive LLMs (see README).
"""
import math
import re

DEPARTMENTS = ["billing", "technical", "account", "other"]

def normalize(text):
    return re.sub(r"\s+", " ", (text or "")).strip()

def tokenize(text):
    """Words + adjacent bigrams. English-focused; swap for char n-grams
    if your tickets are Chinese."""
    words = re.findall(r"[a-z0-9']+", normalize(text).lower())
    toks = list(words)
    toks += [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]
    return toks

def vectorize(text, vocab_index, idf):
    """TF-IDF (sublinear TF + l2 norm) into a dense float list."""
    counts = {}
    for t in tokenize(text):
        counts[t] = counts.get(t, 0) + 1
    v = [0.0] * len(idf)
    for t, c in counts.items():
        idx = vocab_index.get(t)
        if idx is None:
            continue
        v[idx] = (1.0 + math.log(c)) * idf[idx]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]
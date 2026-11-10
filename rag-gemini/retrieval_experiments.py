#!/usr/bin/env python3
"""Retrieval-only experiments that reuse cached Gemini embeddings (no API calls).

1. Passages per question: k = 1, 3, 5, 10
2. Embedding size: Gemini Embedding 2 truncated to 3072 / 1536 / 768 / 256 dimensions
   (the model is trained so that a shortened, re-normalised vector stays usable)
3. Keyword search (BM25) vs embeddings vs a hybrid (reciprocal rank fusion)

    python3 retrieval_experiments.py --data data --cache results/embed-cache --out results/experiments.json
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import re

EMBED_MODEL = "gemini-embedding-2"


def load_cached(cache, texts):
    out = []
    for t in texts:
        p = cache / (hashlib.sha256((EMBED_MODEL + t).encode()).hexdigest() + ".json")
        if not p.exists():
            raise SystemExit(f"Not cached: run run.py first ({t[:60]!r})")
        out.append(json.loads(p.read_text()))
    return out


def normalise(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def ranked(qv, index):
    return [i for _, i in sorted(((sum(a * b for a, b in zip(qv, v)), i) for i, v in enumerate(index)), reverse=True)]


def metrics(rankings, questions, docs, k):
    p = r = mrr = 0.0
    by_need = collections.defaultdict(list)
    for order, q in zip(rankings, questions):
        got = [docs[i]["chunk_link"] for i in order[:k]]
        want = set(q["correct_chunks"])
        hits = [g for g in got if g in want]
        p += len(hits) / k
        r += len(hits) / len(want)
        mrr += next((1 / (n + 1) for n, g in enumerate(got) if g in want), 0.0)
        by_need[min(len(want), 3)].append(len(hits) / len(want))
    n = len(questions)
    return {"k": k, "precision": round(p / n, 4), "recall": round(r / n, 4), "mrr": round(mrr / n, 4),
            "recall_one_passage": round(sum(by_need[1]) / len(by_need[1]), 4),
            "recall_two_passages": round(sum(by_need[2]) / len(by_need[2]), 4)}


# --------------------------------------------------------------- BM25

TOKEN = re.compile(r"[a-z0-9_]+")


def tokens(text):
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [collections.Counter(tokens(t)) for t in texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = collections.Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def ranked(self, query):
        q = tokens(query)
        scores = []
        for i, (d, ln) in enumerate(zip(self.docs, self.lens)):
            s = 0.0
            for w in q:
                if w in d:
                    tf = d[w]
                    s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * ln / self.avg))
            scores.append((s, i))
        return [i for _, i in sorted(scores, reverse=True)]


def rrf(*rankings, c=60):
    """Reciprocal rank fusion: combine rankings without tuning score scales."""
    score = collections.defaultdict(float)
    for order in rankings:
        for pos, i in enumerate(order):
            score[i] += 1 / (c + pos + 1)
    return [i for _, i in sorted(((s, i) for i, s in score.items()), reverse=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--cache", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    docs = json.loads((a.data / "anthropic_summary_indexed_docs.json").read_text())
    questions = json.loads((a.data / "docs_evaluation_dataset.json").read_text())

    plain_texts = [f"Heading: {d['chunk_heading']}\n\n Chunk Text:{d['text']}" for d in docs]
    idx = load_cached(a.cache, plain_texts)
    qv = load_cached(a.cache, [q["question"] for q in questions])
    emb_rank = [ranked(v, idx) for v in qv]
    out = {}

    print("1. Passages per question (plain Gemini search)")
    out["k_sweep"] = [metrics(emb_rank, questions, docs, k) for k in (1, 3, 5, 10)]
    for m in out["k_sweep"]:
        print(f"   k={m['k']:<3} precision {m['precision']:.2f}  recall {m['recall']:.2f}  "
              f"(one-passage {m['recall_one_passage']:.2f}, two-passage {m['recall_two_passages']:.2f})  MRR {m['mrr']:.2f}")

    print("2. Embedding size (k=3)")
    out["dimensions"] = []
    for dim in (3072, 1536, 768, 256):
        ix = [normalise(v[:dim]) for v in idx]
        qs = [normalise(v[:dim]) for v in qv]
        m = metrics([ranked(v, ix) for v in qs], questions, docs, 3) | {"dimensions": dim}
        out["dimensions"].append(m)
        print(f"   {dim:>4} dims  recall {m['recall']:.2f}  MRR {m['mrr']:.2f}  (storage {dim / 3072:.0%})")

    print("3. Keyword vs embedding vs hybrid (k=3)")
    bm = BM25(plain_texts)
    bm_rank = [bm.ranked(q["question"]) for q in questions]
    hybrid = [rrf(e, b) for e, b in zip(emb_rank, bm_rank)]
    out["methods"] = []
    for name, rk in (("BM25 keyword search", bm_rank), ("Gemini Embedding 2", emb_rank), ("Hybrid (RRF)", hybrid)):
        m = metrics(rk, questions, docs, 3) | {"method": name}
        out["methods"].append(m)
        print(f"   {name:<20} recall {m['recall']:.2f}  MRR {m['mrr']:.2f}  two-passage recall {m['recall_two_passages']:.2f}")
    out["methods_k5"] = [metrics(rk, questions, docs, 5) | {"method": name}
                         for name, rk in (("BM25 keyword search", bm_rank), ("Gemini Embedding 2", emb_rank), ("Hybrid (RRF)", hybrid))]

    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Three-level RAG evaluation with Gemini Embedding 2.

Level 1  plain search: embed heading + text, take the top 3 by cosine similarity
Level 2  summary indexing: embed heading + text + summary, top 3
Level 3  re-ranking: take the top 20 from the level-2 index, a Claude model picks the best 3

Retrieval metrics: precision, recall, F1, MRR (first correct passage).
End-to-end metric: answer accuracy, judged by a second Claude model against
the reference answer.

Data (MIT-licensed public evaluation set): pass its folder with --data.
    anthropic_docs.json, anthropic_summary_indexed_docs.json, docs_evaluation_dataset.json

Keys: GEMINI_API_KEY and ANTHROPIC_API_KEY from the environment, or from
~/.config/calinalabs/keys.env (KEY=value lines, chmod 600).

    python3 run.py --data DIR --out DIR             # full run (~$5 of API calls)
    python3 run.py --data DIR --out DIR --limit 10  # quick check on 10 questions
    python3 run.py --data DIR --out DIR --dry-run   # no API calls: fake models, tests the pipeline
    python3 run.py --data DIR --out DIR --search-only  # levels 1-2 retrieval, Gemini key only
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.request

EMBED_MODEL = "gemini-embedding-2"
ANSWER_MODEL = "claude-haiku-5-5"
JUDGE_MODEL = "claude-sonnet-5-5"
K, CANDIDATES = 3, 20
WORKERS = 8
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "8"))  # small batches stay under free-tier per-minute limits


# ----------------------------------------------------------------- plumbing

def load_keys():
    f = pathlib.Path.home() / ".config" / "calinalabs" / "keys.env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def post(url, body, headers, tries=20):
    data = json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read()
            if e.code in (429, 500, 502, 503, 529) and attempt < tries - 1:
                # Honour the server's requested delay (Gemini sends retryDelay, e.g. "37s").
                m = re.search(rb'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', detail)
                wait = float(m.group(1)) + 1 if m else min(60, 2 ** attempt) + random.random()
                print(f"    rate limited (HTTP {e.code}), waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"{url.split('?')[0]} -> HTTP {e.code}: {detail[:300]!r}") from None
        except urllib.error.URLError:
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def normalise(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class Gemini:
    URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents"

    def __init__(self, cache):
        self.key = os.environ["GEMINI_API_KEY"]
        self.cache = cache

    def embed(self, texts):
        out, todo = [None] * len(texts), []
        for i, t in enumerate(texts):
            h = hashlib.sha256((EMBED_MODEL + t).encode()).hexdigest()
            p = self.cache / f"{h}.json"
            if p.exists():
                out[i] = json.loads(p.read_text())
            else:
                todo.append((i, t, p))
        for start in range(0, len(todo), EMBED_BATCH):
            batch = todo[start:start + EMBED_BATCH]
            body = {"requests": [{"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": t}]}} for _, t, _ in batch]}
            res = post(self.URL, body, {"x-goog-api-key": self.key})
            for (i, _, p), e in zip(batch, res["embeddings"]):
                v = normalise(e["values"])
                p.write_text(json.dumps(v))
                out[i] = v
        return out


class Claude:
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self):
        self.key = os.environ["ANTHROPIC_API_KEY"]

    def ask(self, model, prompt, max_tokens=1500):
        res = post(self.URL, {"model": model, "max_tokens": max_tokens, "temperature": 0,
                              "messages": [{"role": "user", "content": prompt}]},
                   {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in res["content"])


class FakeEmbed:
    """Deterministic bag-of-words vectors: lets --dry-run exercise every step."""
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 256
            for w in re.findall(r"[a-z]{3,}", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
            out.append(normalise(v))
        return out


class FakeClaude:
    def ask(self, model, prompt, max_tokens=0):
        if "<relevant_indices>" in prompt:
            n = len(re.findall(r"<document index=", prompt))
            return "<relevant_indices>" + ",".join(str(i) for i in range(min(K, n))) + "</relevant_indices>"
        if "<correct>" in prompt:
            return "<explanation>dry run</explanation><correct>" + random.choice(["true", "false"]) + "</correct>"
        return "Dry-run answer."


# ------------------------------------------------------------------- levels

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def search(qv, index, k):
    scored = sorted(((dot(qv, v), i) for i, v in enumerate(index)), reverse=True)
    return [i for _, i in scored[:k]]


def rerank(llm, question, docs, cand):
    listing = "\n".join(f'<document index="{n}">\n{docs[i]["chunk_heading"]}\n{docs[i]["summary"]}\n</document>' for n, i in enumerate(cand))
    prompt = (f"Query: {question}\n\nYou are about to be given a group of documents, each preceded by its index number in square brackets. "
              f"Your task is to select only the {K} most relevant documents from the list to help us answer the query.\n\n{listing}\n\n"
              f"Output only the indices of {K} most relevant documents in order of relevance, separated by commas, enclosed in XML tags here:\n"
              f"<relevant_indices>put the numbers of your indices here, separated by commas</relevant_indices>")
    raw = llm.ask(ANSWER_MODEL, prompt, 50)
    m = re.search(r"<relevant_indices>(.*?)</relevant_indices>", raw, re.S)
    picks = []
    for tok in (m.group(1).split(",") if m else []):
        tok = tok.strip()
        if tok.isdigit() and int(tok) < len(cand) and cand[int(tok)] not in picks:
            picks.append(cand[int(tok)])
    for i in cand:  # fallback: keep embedding order for any missing slots
        if len(picks) >= K:
            break
        if i not in picks:
            picks.append(i)
    return picks[:K]


def answer(llm, question, docs, ids, with_summary):
    ctx = "\n".join(
        f"<document>\n{docs[i]['chunk_heading']}\n\nText\n{docs[i]['text']}" + (f"\n\nSummary:\n{docs[i]['summary']}" if with_summary else "") + "\n</document>"
        for i in ids)
    prompt = (f"You have been tasked with helping us to answer the following query:\n<query>\n{question}\n</query>\n"
              f"You have access to the following documents which are meant to provide context as you answer the query:\n<documents>\n{ctx}\n</documents>\n"
              "Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already.\n"
              "Answer the question now, and avoid providing preamble such as 'Here is the answer', etc")
    return llm.ask(ANSWER_MODEL, prompt, 2500)


def judge(llm, question, reference, generated):
    prompt = (f"You are an AI assistant tasked with evaluating the correctness of answers to questions about Anthropic's documentation.\n\n"
              f"Question: {question}\n\nCorrect Answer: {reference}\n\nGenerated Answer: {generated}\n\n"
              "Is the Generated Answer correct based on the Correct Answer? You should pay attention to the substance of the answer, and ignore minute details that may differ.\n\n"
              "Small differences or changes in wording don't matter. If the generated answer and correct answer are saying essentially the same thing then that generated answer should be marked correct.\n\n"
              "However, if there is any critical piece of information which is missing from the generated answer in comparison to the correct answer, then we should mark this as incorrect.\n\n"
              "Finally, if there are any direct contradictions between the correct answer and generated answer, we should deem the generated answer to be incorrect.\n\n"
              "Respond in the following XML format:\n<evaluation>\n<content>\n<explanation>Your reasoning here</explanation>\n<is_correct>true/false</is_correct>\n</content>\n</evaluation>\n\n"
              "Reply with <correct>true</correct> or <correct>false</correct> on the last line.")
    raw = llm.ask(JUDGE_MODEL, prompt, 1500)
    m = re.findall(r"<(?:correct|is_correct)>\s*(true|false)\s*</(?:correct|is_correct)>", raw, re.I)
    return (m[-1].lower() == "true") if m else None, raw


def retrieval_metrics(rows):
    p = r = mrr = f1q = 0.0
    for row in rows:
        got, want = row["retrieved"], set(row["correct_chunks"])
        hits = [g for g in got if g in want]
        pi, ri = len(hits) / len(got) if got else 0.0, len(hits) / len(want) if want else 0.0
        p, r = p + pi, r + ri
        f1q += 2 * pi * ri / (pi + ri) if pi + ri else 0.0
        mrr += next((1 / (n + 1) for n, g in enumerate(got) if g in want), 0.0)
    n = len(rows)
    P, R = p / n, r / n
    return {"precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0.0, "f1_per_query": f1q / n, "mrr": mrr / n}


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-answers", action="store_true", help="retrieval metrics only (cheapest)")
    ap.add_argument("--search-only", action="store_true", help="levels 1-2 retrieval only: needs just GEMINI_API_KEY")
    a = ap.parse_args()
    if a.search_only:
        a.no_answers = True

    docs = json.loads((a.data / "anthropic_summary_indexed_docs.json").read_text())
    questions = json.loads((a.data / "docs_evaluation_dataset.json").read_text())
    if a.limit:
        questions = questions[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    random.seed(7)

    if a.dry_run:
        emb, llm = FakeEmbed(), FakeClaude()
    else:
        load_keys()
        needed = ("GEMINI_API_KEY",) if a.search_only else ("GEMINI_API_KEY", "ANTHROPIC_API_KEY")
        missing = [k for k in needed if not os.environ.get(k)]
        if missing:
            sys.exit(f"Missing {', '.join(missing)}: set it in the environment or ~/.config/calinalabs/keys.env")
        (a.out / "embed-cache").mkdir(exist_ok=True)
        emb, llm = Gemini(a.out / "embed-cache"), (None if a.search_only else Claude())

    t0 = time.time()
    print(f"Embedding {len(docs)} passages twice and {len(questions)} questions with {EMBED_MODEL if not a.dry_run else 'a fake model'}...")
    idx1 = emb.embed([f"Heading: {d['chunk_heading']}\n\n Chunk Text:{d['text']}" for d in docs])
    idx2 = emb.embed([f"{d['chunk_heading']}\n\n{d['text']}\n\n{d['summary']}" for d in docs])
    qvec = emb.embed([q["question"] for q in questions])

    def run_question(n):
        q, qv = questions[n], qvec[n]
        l1, cand = search(qv, idx1, K), search(qv, idx2, CANDIDATES)
        l2 = cand[:K]
        l3 = None if a.search_only else rerank(llm, q["question"], docs, cand)
        out = {"id": q["id"], "question": q["question"], "correct_chunks": q["correct_chunks"], "levels": {}}
        for name, ids, summ in (("level1", l1, False), ("level2", l2, True), ("level3", l3, True)):
            if ids is None:
                continue
            row: dict = {"retrieved": [docs[i]["chunk_link"] for i in ids]}
            if not a.no_answers:
                gen = answer(llm, q["question"], docs, ids, summ)
                ok, raw = judge(llm, q["question"], q["correct_answer"], gen)
                row.update(answer=gen, correct=ok, judge=raw)
            out["levels"][name] = row
        return out

    results = []
    with cf.ThreadPoolExecutor(WORKERS) as pool:
        for i, res in enumerate(pool.map(run_question, range(len(questions))), 1):
            results.append(res)
            if i % 10 == 0 or i == len(questions):
                print(f"  {i}/{len(questions)} questions done")

    summary = {"embedding_model": EMBED_MODEL if not a.dry_run else "fake", "answer_model": ANSWER_MODEL, "judge_model": JUDGE_MODEL,
               "questions": len(questions), "k": K, "rerank_candidates": CANDIDATES, "dry_run": a.dry_run, "levels": {}}
    for name in [n for n in ("level1", "level2", "level3") if n in results[0]["levels"]]:
        rows = [{"retrieved": r["levels"][name]["retrieved"], "correct_chunks": r["correct_chunks"]} for r in results]
        m = retrieval_metrics(rows)
        if not a.no_answers:
            judged = [r["levels"][name]["correct"] for r in results]
            m["accuracy"] = sum(1 for j in judged if j) / len(judged)
            m["judge_unparsed"] = sum(1 for j in judged if j is None)
        summary["levels"][name] = m
    summary["seconds"] = round(time.time() - t0)

    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    names = {"level1": "Plain search", "level2": "+ summary indexing", "level3": "+ re-ranking"}
    print(f"\n{'Level':22} {'Precision':>9} {'Recall':>7} {'F1':>6} {'MRR':>6} {'Accuracy':>9}")
    for k, v in summary["levels"].items():
        acc = f"{v['accuracy']:.0%}" if "accuracy" in v else "-"
        print(f"{names[k]:22} {v['precision']:9.2f} {v['recall']:7.2f} {v['f1']:6.2f} {v['mrr']:6.2f} {acc:>9}")
    print(f"\nSaved {a.out / 'summary.json'} and results.json ({summary['seconds']}s)")


if __name__ == "__main__":
    main()

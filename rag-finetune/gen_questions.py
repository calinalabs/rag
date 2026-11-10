#!/usr/bin/env python3
"""Generate synthetic training questions for each passage (Claude Haiku).

The 100 evaluation questions are never read here: training data comes only
from the passages themselves.

    python3 gen_questions.py --data ../rag-gemini/data --out data/train_questions.json
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "rag-gemini"))
from run import Claude, load_keys  # noqa: E402  (same HTTP client and key loading)

MODEL = "claude-haiku-5-5"
PER_PASSAGE = 3
PROMPT = """Here is a passage from a software product's developer documentation.

<heading>{heading}</heading>
<passage>
{text}
</passage>

Write {n} different questions that a developer might ask, which this passage answers.
- Vary them: one short and direct, one about how to do something, one that needs a detail from the passage.
- Phrase them the way a user would ask, not by copying the heading or the passage's wording.
- Each question must be answerable from this passage alone.

Return only a JSON list of {n} strings."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    load_keys()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY")
    docs = json.loads((a.data / "anthropic_summary_indexed_docs.json").read_text())
    llm = Claude()

    def one(d):
        raw = llm.ask(MODEL, PROMPT.format(heading=d["chunk_heading"], text=d["text"][:6000], n=PER_PASSAGE), 600)
        m = re.search(r"\[.*\]", raw, re.S)
        qs = json.loads(m.group(0)) if m else []
        return [{"chunk_link": d["chunk_link"], "question": q.strip()} for q in qs if isinstance(q, str) and q.strip()][:PER_PASSAGE]

    rows = []
    with cf.ThreadPoolExecutor(8) as pool:
        for n, r in enumerate(pool.map(one, docs), 1):
            rows += r
            if n % 50 == 0 or n == len(docs):
                print(f"  {n}/{len(docs)} passages")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1))
    print(f"Saved {len(rows)} questions for {len({r['chunk_link'] for r in rows})} passages to {a.out}")


if __name__ == "__main__":
    main()

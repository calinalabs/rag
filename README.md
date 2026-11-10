# Calina Labs: RAG experiments

Code and full results behind the retrieval experiments published by [Calina Labs](https://calinalabs.com), an independent UK consultancy that builds agentic processes, private knowledge bases and process automation.

- Website: [calinalabs.com](https://calinalabs.com)
- Write-ups: [calinalabs.com/blog](https://calinalabs.com/blog/)
- Knowledge bases on your own documents: [calinalabs.com/services/knowledge-base](https://calinalabs.com/services/knowledge-base/)
- Contact: [calinalabs.com/contact](https://calinalabs.com/contact/)

| Folder | Experiment | Write-up |
|---|---|---|
| [`rag-gemini/`](rag-gemini/) | Three-level RAG evaluation with Gemini Embedding 2, plus k, embedding-size and BM25/hybrid sweeps | [Testing RAG with Gemini Embedding 2](https://calinalabs.com/blog/gemini-embedding-rag/) |
| [`rag-finetune/`](rag-finetune/) | Fine-tuning an open-weight embedding model (bge-small) on synthetic questions | [Fine-tuning a small open embedding model on a laptop](https://calinalabs.com/blog/fine-tune-embedding-model/) |

## rag-gemini: three-level RAG evaluation with Gemini Embedding 2

Measures how much two common changes improve document search for an AI agent, using Google's `gemini-embedding-2` model on a public evaluation set of 100 questions over 232 documentation passages.

| Level | What changes |
|---|---|
| 1. Plain search | Embed each passage with its heading; return the 3 most similar passages |
| 2. Summary indexing | Embed heading + text + a short summary of each passage |
| 3. Re-ranking | Take the top 20 from level 2; a fast LLM picks the best 3, in order |

Metrics: precision, recall, F1 and MRR for search; answer accuracy (an LLM judge against reference answers) end to end.

### Results (10 November 2026)

Gemini Embedding 2, 100 questions, top 3 passages; answers by Claude Haiku 5.5, judged by Claude Sonnet 5.5.

| Level | Precision | Recall | F1 | MRR | Answer accuracy |
|---|---|---|---|---|---|
| Plain search | 0.45 | 0.74 | 0.56 | 0.90 | 86% |
| + summary indexing | 0.46 | 0.74 | 0.57 | 0.89 | 91% |
| + re-ranking | 0.46 | 0.75 | 0.57 | 0.88 | 88% |

Every answer, retrieved passage and judge verdict is in `rag-gemini/results/full/results.json`. Write-up: [Testing RAG with Gemini Embedding 2](https://calinalabs.com/blog/gemini-embedding-rag/).

### Run it

Requires Python 3.10+ and no third-party packages.

```bash
cd rag-gemini
./get_data.sh                                     # download the evaluation set into ./data
export GEMINI_API_KEY=...                         # or put it in ~/.config/calinalabs/keys.env
python3 run.py --data data --out results --search-only     # levels 1-2, Gemini key only
export ANTHROPIC_API_KEY=...
python3 run.py --data data --out results                   # all three levels + answer accuracy
python3 run.py --data data --out results --dry-run         # no API calls: tests the pipeline
```

Embeddings are cached in `results/embed-cache`, so an interrupted run resumes where it stopped. On Gemini's free tier the script waits whenever the per-minute limit is reached.

### Notes

- Gemini Embedding 2 does not accept `task_type`, so questions and passages are embedded the same way.
- The script always takes the top 3 passages. It does not drop results below a fixed similarity score, because such thresholds are specific to one embedding model.
- The evaluation set is downloaded from its original source (MIT licence) by `get_data.sh`; it is not redistributed here.

## rag-finetune: post-training an open embedding model

Fine-tunes `BAAI/bge-small-en-v1.5` on synthetic questions for the same passages and scores it on the same 100 test questions. Recall@3 rises from 0.62 to 0.71 and MRR from 0.73 to 0.87. Details and commands: [`rag-finetune/README.md`](rag-finetune/README.md). Write-up: [Fine-tuning a small open embedding model on a laptop](https://calinalabs.com/blog/fine-tune-embedding-model/).

---

Built by [Calina Labs](https://calinalabs.com). Questions about applying this to your own documents? [Get in touch](https://calinalabs.com/contact/).

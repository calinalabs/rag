# rag-finetune: post-training an open embedding model

Fine-tunes `BAAI/bge-small-en-v1.5` (33M parameters, open weights) on synthetic questions
for the same 232 passages used in `rag-gemini`, then scores it on the same 100 test questions.

```bash
python3 gen_questions.py --data ../rag-gemini/data --out data/train_questions.json   # Claude Haiku, ~700 questions
../.venv/bin/python finetune.py --data ../rag-gemini/data --train data/train_questions.json --out results
```

Setup: `python3 -m venv ../.venv && ../.venv/bin/pip install torch sentence-transformers datasets accelerate`.
Runs on an Apple M3 with 16 GB (about 15 minutes for both training runs). Results: `results/finetune_results.json`.

| Model | Recall@3 | MRR |
|---|---|---|
| BM25 keyword search | 0.60 | 0.75 |
| bge-small, untuned | 0.62 | 0.73 |
| bge-small, fine-tuned | 0.71 | 0.87 |
| Gemini Embedding 2 | 0.74 | 0.90 |

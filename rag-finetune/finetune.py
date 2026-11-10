#!/usr/bin/env python3
"""Post-train an open-weight embedding model on synthetic questions and measure it.

Experiments (all scored on the same 100 human-written test questions):
  A. base model, untuned
  B. fine-tuned on questions for all 232 passages
  C. held-out check: fine-tuned on questions for half the passages only, then
     scored separately on test questions about seen vs unseen passages
Plus a leakage check: how close any training question is to any test question.

    ../.venv/bin/python finetune.py --data ../rag-gemini/data --train data/train_questions.json --out results
"""
import argparse
import hashlib
import json
import pathlib
import random

import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers

BASE = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge's recommended query prompt
K = 3
SEED = 7


def passage_text(d):
    return f"{d['chunk_heading']}\n\n{d['text']}"


def evaluate(model, docs, questions, k=K):
    links = [d["chunk_link"] for d in docs]
    P = model.encode([passage_text(d) for d in docs], normalize_embeddings=True, convert_to_tensor=True, batch_size=16)
    Q = model.encode([QUERY_PREFIX + q["question"] for q in questions], normalize_embeddings=True, convert_to_tensor=True)
    order = torch.argsort(Q @ P.T, dim=1, descending=True)[:, :k].tolist()
    p = r = mrr = 0.0
    two = []
    for row, q in zip(order, questions):
        got, want = [links[i] for i in row], set(q["correct_chunks"])
        hits = [g for g in got if g in want]
        p += len(hits) / k
        r += len(hits) / len(want)
        mrr += next((1 / (n + 1) for n, g in enumerate(got) if g in want), 0.0)
        if len(want) == 2:
            two.append(len(hits) / 2)
    n = len(questions)
    return {"n": n, "precision": round(p / n, 4), "recall": round(r / n, 4), "mrr": round(mrr / n, 4),
            "recall_two_passages": round(sum(two) / len(two), 4) if two else None}


def train(train_rows, docs_by_link, out_dir, epochs, batch):
    # "eager" attention: the Mac GPU's fused attention has no dropout support in training mode.
    model = SentenceTransformer(BASE, device="mps" if torch.backends.mps.is_available() else "cpu",
                                model_kwargs={"attn_implementation": "eager"})
    model.max_seq_length = 512
    ds = Dataset.from_dict({
        "anchor": [QUERY_PREFIX + r["question"] for r in train_rows],
        "positive": [passage_text(docs_by_link[r["chunk_link"]]) for r in train_rows],
    })
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs, per_device_train_batch_size=batch,
        learning_rate=2e-5, warmup_ratio=0.1, seed=SEED,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # no two questions about the same passage in a batch
        save_strategy="no", logging_steps=10, report_to="none")
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                               # Cached MNRL (GradCache): same 32-example batches, computed 8 at a time to fit in 16 GB.
                               loss=losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)).train()
    return model


def half(link):
    return int(hashlib.sha256(link.encode()).hexdigest(), 16) % 2  # deterministic 50/50 split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--train", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    docs = json.loads((a.data / "anthropic_summary_indexed_docs.json").read_text())
    tests = json.loads((a.data / "docs_evaluation_dataset.json").read_text())
    train_rows = json.loads(a.train.read_text())
    by_link = {d["chunk_link"]: d for d in docs}
    train_rows = [r for r in train_rows if r["chunk_link"] in by_link]
    a.out.mkdir(parents=True, exist_ok=True)
    res = {"base_model": BASE, "train_questions": len(train_rows), "epochs": a.epochs, "batch": a.batch}

    base = SentenceTransformer(BASE, device="mps" if torch.backends.mps.is_available() else "cpu")
    base.max_seq_length = 512

    # Leakage check: nearest training question for each test question (untuned model).
    T = base.encode([r["question"] for r in train_rows], normalize_embeddings=True, convert_to_tensor=True)
    E = base.encode([q["question"] for q in tests], normalize_embeddings=True, convert_to_tensor=True)
    best, idx = (E @ T.T).max(dim=1)
    pairs = sorted(zip(best.tolist(), idx.tolist(), range(len(tests))), reverse=True)
    res["leakage"] = {"max_similarity": round(pairs[0][0], 4), "over_0_9": sum(1 for s, _, _ in pairs if s > 0.9),
                      "exact_matches": sum(1 for q in tests for r in train_rows if q["question"].strip().lower() == r["question"].lower()),
                      "closest": [{"similarity": round(s, 3), "test": tests[t]["question"], "train": train_rows[i]["question"]} for s, i, t in pairs[:3]]}
    print("Leakage check:", {k: v for k, v in res["leakage"].items() if k != "closest"})

    print("A. Untuned", BASE)
    res["A_base"] = evaluate(base, docs, tests)
    print("  ", res["A_base"])

    print("B. Fine-tuned on all passages")
    model_b = train(train_rows, by_link, a.out / "ckpt-b", a.epochs, a.batch)
    res["B_finetuned_all"] = evaluate(model_b, docs, tests)
    print("  ", res["B_finetuned_all"])
    # Robustness: drop test questions with a near-duplicate training question.
    clean = [q for q, s in zip(tests, best.tolist()) if s <= 0.9]
    res["B_no_close_match"] = {"untuned": evaluate(base, docs, clean), "finetuned": evaluate(model_b, docs, clean)}
    print("   without near-duplicates:", res["B_no_close_match"])
    model_b.save(str(a.out.parent / "models" / "bge-small-finetuned"))

    print("C. Held-out check: train on half the passages")
    seen = {d["chunk_link"] for d in docs if half(d["chunk_link"]) == 0}
    model_c = train([r for r in train_rows if r["chunk_link"] in seen], by_link, a.out / "ckpt-c", a.epochs, a.batch)
    t_seen = [q for q in tests if all(c in seen for c in q["correct_chunks"])]
    t_unseen = [q for q in tests if not any(c in seen for c in q["correct_chunks"])]
    res["C_heldout"] = {
        "seen_passages": {"untuned": evaluate(base, docs, t_seen), "finetuned_half": evaluate(model_c, docs, t_seen)},
        "unseen_passages": {"untuned": evaluate(base, docs, t_unseen), "finetuned_half": evaluate(model_c, docs, t_unseen)},
    }
    for part, v in res["C_heldout"].items():
        print(f"   {part}: n={v['untuned']['n']}  untuned recall {v['untuned']['recall']:.2f} -> {v['finetuned_half']['recall']:.2f}, "
              f"MRR {v['untuned']['mrr']:.2f} -> {v['finetuned_half']['mrr']:.2f}")

    (a.out / "finetune_results.json").write_text(json.dumps(res, indent=2))
    print("Saved", a.out / "finetune_results.json")


if __name__ == "__main__":
    main()

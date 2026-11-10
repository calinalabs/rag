#!/usr/bin/env bash
# Download the public RAG evaluation set used by run.py into ./data.
# Source: an open-source RAG guide, MIT licence (see the repository below).
set -euo pipefail
cd "$(dirname "$0")"
SRC=https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/capabilities/retrieval_augmented_generation
mkdir -p data
for f in data/anthropic_summary_indexed_docs.json evaluation/docs_evaluation_dataset.json; do
  curl -fsSL "$SRC/$f" -o "data/$(basename "$f")"
done
echo "Saved to ./data:"; ls -l data

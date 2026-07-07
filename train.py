"""train.py — Build the BM25 spec RAG index and evaluate."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.data import make_corpus
from src.model import build_bm25_retriever
from src.evaluate import evaluate_spec_rag, save_metrics

def main():
    corpus = make_corpus()
    print(f"Corpus: {corpus['n_documents']} chip spec docs, {corpus['n_queries']} queries")
    retriever = build_bm25_retriever(corpus)
    print("Built BM25 retriever.")
    metrics = evaluate_spec_rag(retriever, corpus["queries"])
    agg = metrics["aggregate"]
    print(f"  Recall@5: {agg['recall@5']:.4f}")
    print(f"  MRR:      {agg['mrr']:.4f}")
    print(f"  Spec Acc: {agg['spec_accuracy']:.4f}")
    save_metrics(metrics)
    print("Saved metrics -> models/metrics.json")

if __name__ == "__main__": main()

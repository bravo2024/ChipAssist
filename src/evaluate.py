"""evaluate.py — Batch evaluation of the ChipAssist BM25 spec RAG."""
from __future__ import annotations
import json
from pathlib import Path
from src.core import recall_at_k, mrr, spec_extraction_accuracy


def evaluate_spec_rag(retriever, queries, k_values=(1, 3, 5)):
    from src.model import answer_spec_query
    per_query = []
    for q in queries:
        result = answer_spec_query(retriever, q["query"], k=5)
        retrieved_ids = [r["doc_id"] for r in result.get("retrieved", [])]
        spec_acc = spec_extraction_accuracy(result.get("extracted_specs", {}), q.get("expected_spec", {}))
        per_query.append({
            "query_id": q["id"], "query": q["query"],
            "answer": result["answer"], "source_doc_id": result["source_doc_id"],
            "extracted_specs": result.get("extracted_specs", {}),
            "spec_accuracy": spec_acc,
            "recall@5": recall_at_k(retrieved_ids, q["relevant_doc_ids"], 5),
            "mrr": mrr(retrieved_ids, q["relevant_doc_ids"]),
        })
    n = max(len(per_query), 1)
    aggregate = {
        "n_queries": len(per_query),
        "recall@5": sum(p["recall@5"] for p in per_query) / n,
        "mrr": sum(p["mrr"] for p in per_query) / n,
        "spec_accuracy": sum(p["spec_accuracy"] for p in per_query) / n,
    }
    return {"aggregate": aggregate, "per_query": per_query}


def save_metrics(metrics, path="models/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def _clean(v):
        if isinstance(v, (int, float)): return float(v)
        if isinstance(v, dict): return {k: _clean(vv) for k, vv in v.items()}
        if isinstance(v, list): return [_clean(x) for x in v]
        return v
    with open(path, "w") as f: json.dump(_clean(metrics), f, indent=2)
    return metrics

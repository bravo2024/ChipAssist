"""Smoke tests for ChipAssist — BM25 semiconductor spec RAG."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_corpus
from src.model import BM25Retriever, build_bm25_retriever, answer_spec_query, extract_specs
from src.core import recall_at_k, mrr, spec_extraction_accuracy


def test_corpus():
    c = make_corpus()
    assert c["n_documents"] >= 10
    assert c["n_queries"] >= 8


def test_bm25_retriever():
    c = make_corpus()
    r = build_bm25_retriever(c)
    results = r.retrieve("supply voltage PHY", k=3)
    assert len(results) > 0
    assert results[0]["score"] >= results[-1]["score"]


def test_spec_extraction():
    """Spec extractor pulls voltage and pin count from text."""
    text = "Supply voltage is 3.3V. The device has 48 pins in a QFN package."
    specs = extract_specs(text)
    assert specs.get("voltage") == "3.3V"
    assert specs.get("pins") == "48"


def test_answer_spec_query():
    c = make_corpus()
    r = build_bm25_retriever(c)
    result = answer_spec_query(r, "What is the supply voltage of the 88E1510 PHY?", k=3)
    assert result["source_doc_id"] is not None
    assert len(result["answer"]) > 0


def test_retrieval_metrics():
    c = make_corpus()
    r = build_bm25_retriever(c)
    recalls = []
    for q in c["queries"][:8]:
        retrieved = [x["doc_id"] for x in r.retrieve(q["query"], k=5)]
        recalls.append(recall_at_k(retrieved, q["relevant_doc_ids"], 5))
    assert sum(recalls) / len(recalls) > 0.2


def test_spec_accuracy_metric():
    extracted = {"voltage": "3.3V", "pins": "48"}
    expected = {"voltage": "3.3V", "pins": "48", "clock": "125 MHz"}
    acc = spec_extraction_accuracy(extracted, expected)
    assert abs(acc - 2/3) < 0.01


if __name__ == "__main__":
    test_corpus()
    test_bm25_retriever()
    test_spec_extraction()
    test_answer_spec_query()
    test_retrieval_metrics()
    test_spec_accuracy_metric()
    print("All ChipAssist smoke tests passed!")

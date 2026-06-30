"""core.py — BM25 retrieval and spec-extraction metrics for ChipAssist.

Unlike KnowledgeBot (TF-IDF) and ChatAssist (TF-IDF + guardrails),
ChipAssist uses a BM25 retriever and adds spec extraction accuracy.

Reference: Robertson & Zaragoza (2009), "The Probabilistic Relevance
Framework: BM25 and Beyond."
"""
from __future__ import annotations
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("a an the and or of to in on for is are was were be with as at by it its from this that".split())


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOPWORDS]


def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    return sum(1 for r in list(retrieved_ids)[:k] if r in relevant) / len(relevant)


def mrr(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def spec_extraction_accuracy(extracted, expected) -> float:
    """Fraction of expected spec keys correctly extracted."""
    if not expected:
        return 1.0
    return sum(1 for k, v in expected.items() if extracted.get(k) == v) / len(expected)


def faithfulness(answer, context) -> float:
    """Token-F1 overlap between answer and context (SQuAD-style)."""
    a = Counter(_tokenize(answer))
    c = Counter(_tokenize(context))
    common = sum((a & c).values())
    total = sum(a.values()) + sum(c.values())
    return 2 * common / total if total > 0 else 0.0
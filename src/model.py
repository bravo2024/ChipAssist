"""model.py — BM25 retriever and spec extractor for ChipAssist.

Uses Okapi BM25 (Robertson & Zaragoza 2009) instead of TF-IDF.
Also includes a spec extraction function using regex patterns.

BM25(q, d) = sum_t IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*|d|/avgdl))
"""
from __future__ import annotations
import re
from collections import Counter
from typing import Any

import numpy as np

from src.core import _tokenize


class BM25Retriever:
    """Okapi BM25 retriever over a document corpus."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, Any]] = []
        self.doc_freqs: list[dict[str, int]] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[dict[str, Any]]) -> "BM25Retriever":
        self.documents = list(documents)
        self.doc_freqs = []
        self.doc_lens = []
        vocab = set()
        for d in self.documents:
            text = f"{d.get('title', '')} {d.get('content', '')}"
            tokens = _tokenize(text)
            self.doc_freqs.append(dict(Counter(tokens)))
            self.doc_lens.append(len(tokens))
            vocab.update(tokens)
        self.avgdl = float(np.mean(self.doc_lens)) if self.doc_lens else 1.0
        n = len(self.documents)
        for term in vocab:
            df = sum(1 for df_doc in self.doc_freqs if term in df_doc)
            self.idf[term] = float(np.log((n - df + 0.5) / (df + 0.5) + 1))
        return self

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        q_tokens = _tokenize(query)
        scores = []
        for i, doc in enumerate(self.documents):
            score = 0.0
            freq = self.doc_freqs[i]
            dl = self.doc_lens[i]
            for term in q_tokens:
                if term not in freq:
                    continue
                tf = freq[term]
                idf = self.idf.get(term, 0.0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                score += idf * num / den
            scores.append((score, i))
        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:k]:
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append({
                "doc_id": doc["id"], "title": doc.get("title", ""),
                "content": doc.get("content", ""), "category": doc.get("category", ""),
                "score": float(score),
            })
        return results
        return results


# ── Spec extraction ──────────────────────────────────────────────────────────

_SPEC_PATTERNS = {
    "voltage": re.compile(r"(\d+\.?\d*\s*V)", re.I),
    "pins": re.compile(r"(\d+)\s+pins?", re.I),
    "clock": re.compile(r"(\d+\.?\d*\s*(?:GHz|MHz))", re.I),
    "cores": re.compile(r"(\d+)\s*(?:core|cores)", re.I),
    "capacity": re.compile(r"(\d+\s*Gbps)", re.I),
    "pcie": re.compile(r"(Gen\d)", re.I),
    "ports": re.compile(r"(\d+)\s*port", re.I),
    "rate": re.compile(r"(\d+\.?\d*\s*Gbps)", re.I),
    "encryption": re.compile(r"(AES-\d+)", re.I),
    "wifi": re.compile(r"802\.11\s*\w*\s*(\w+)", re.I),
}


def extract_specs(text: str) -> dict:
    """Extract key-value spec pairs from a text passage using regex patterns."""
    specs = {}
    for key, pattern in _SPEC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            specs[key] = match.group(1).strip()
    return specs


def build_bm25_retriever(corpus: dict) -> BM25Retriever:
    """Convenience: fit a BM25Retriever on the corpus."""
    return BM25Retriever().fit(corpus["documents"])


def answer_spec_query(retriever: BM25Retriever, query: str, k: int = 3) -> dict:
    """Retrieve relevant passages and extract specs from the top result."""
    retrieved = retriever.retrieve(query, k=k)
    if not retrieved:
        return {"answer": "", "source_doc_id": None, "extracted_specs": {}, "retrieved": []}
    top = retrieved[0]
    specs = extract_specs(top["content"])
    if specs:
        answer = "; ".join(f"{k}: {v}" for k, v in specs.items())
    else:
        sentences = re.split(r"(?<=[.!?])\s+", top["content"])
        q_tokens = set(_tokenize(query))
        best = max(sentences, key=lambda s: len(q_tokens & set(_tokenize(s))), default=sentences[0] if sentences else "")
        answer = best.strip() if best else top["content"][:200]
    return {"answer": answer, "source_doc_id": top["doc_id"],
            "extracted_specs": specs, "retrieved": retrieved}
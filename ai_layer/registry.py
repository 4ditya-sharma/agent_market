"""Agent registry + semantic search.

Uses sentence-transformers when installed; falls back to TF-IDF so it still works on a
laptop without torch. Ranking blends meaning-match, trust score and price.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import config
from .schemas import AgentCard

log = logging.getLogger("ai_layer.registry")


class _Embedder:
    def __init__(self):
        self.kind = "tfidf"
        self._st = None
        self._tfidf = None
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(config.EMBED_MODEL)
            self.kind = "sentence-transformers"
        except Exception as e:  # not installed / no internet for weights
            log.info("sentence-transformers unavailable (%s); using TF-IDF", e)

    def fit(self, corpus: list[str]):
        if self._st is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(texts, normalize_embeddings=True))
        m = self._tfidf.transform(texts).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


class AgentRegistry:
    def __init__(self):
        self.agents: dict[str, AgentCard] = {}
        self.trust: dict[str, float] = {}  # updated by audits / judge history
        self._embedder: Optional[_Embedder] = None
        self._matrix: Optional[np.ndarray] = None
        self._ids: list[str] = []

    @staticmethod
    def _text(a: AgentCard) -> str:
        return f"{a.name}. {a.description}. Skills: {', '.join(a.capabilities)}"

    def register(self, card: AgentCard, trust: float = 0.7):
        self.agents[card.agent_id] = card
        self.trust.setdefault(card.agent_id, trust)
        self._matrix = None  # rebuild index lazily

    def _index(self):
        if self._matrix is not None:
            return
        if self._embedder is None:
            self._embedder = _Embedder()
        self._ids = list(self.agents)
        corpus = [self._text(self.agents[i]) for i in self._ids]
        self._embedder.fit(corpus)
        self._matrix = self._embedder.encode(corpus)

    def search(
        self,
        query: str,
        top_k: int = 3,
        capability: Optional[str] = None,
        max_price: Optional[float] = None,
    ) -> list[dict]:
        """Return agents ranked by 0.6*similarity + 0.3*trust + 0.1*cheapness."""
        if not self.agents:
            return []
        self._index()
        q = self._embedder.encode([query])[0]
        sims = self._matrix @ q
        prices = np.array([self.agents[i].price_per_task for i in self._ids])
        cheap = 1 - (prices / prices.max() if prices.max() > 0 else prices)
        out = []
        for idx, aid in enumerate(self._ids):
            a = self.agents[aid]
            if capability and capability not in a.capabilities:
                continue
            if max_price is not None and a.price_per_task > max_price:
                continue
            score = 0.6 * float(sims[idx]) + 0.3 * self.trust[aid] + 0.1 * float(cheap[idx])
            out.append({
                "agent": a.model_dump(),
                "similarity": round(float(sims[idx]), 4),
                "trust": round(self.trust[aid], 3),
                "score": round(score, 4),
            })
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:top_k]

    @property
    def backend(self) -> str:
        self._index()
        return self._embedder.kind

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*")

# A small, curated stop-word list. Kept conservative on purpose: the matcher
# already discounts very common terms through the inverse-document-frequency
# factor, so the stop list mostly removes filler that adds noise to short
# queries.
_STOP = frozenset(
    """
    a an and are as at be by for from has have he his i in is it its of on or
    she that the their them they this to was we were will with you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOP]


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class LexicalIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, Counter] = {}
        self._doc_len: dict[str, int] = {}
        self._df: Counter = Counter()
        self._avgdl: float = 0.0

    def add(self, doc_id: str, text: str, *, aliases: list[str] | None = None) -> None:
        if doc_id in self._docs:
            self.remove(doc_id)
        searchable = text
        if aliases:
            searchable = text + " " + " ".join(aliases)
        tokens = tokenize(searchable)
        ctr = Counter(tokens)
        self._docs[doc_id] = ctr
        self._doc_len[doc_id] = len(tokens)
        for term in ctr:
            self._df[term] += 1
        self._recompute_avg()

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            return
        ctr = self._docs.pop(doc_id)
        self._doc_len.pop(doc_id)
        for term in ctr:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        self._recompute_avg()

    def update(self, doc_id: str, text: str, *, aliases: list[str] | None = None) -> None:
        self.remove(doc_id)
        self.add(doc_id, text, aliases=aliases)

    @property
    def N(self) -> int:
        return len(self._docs)

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0 or self.N == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: Iterable[str], doc_id: str) -> float:
        if self.N == 0 or doc_id not in self._docs:
            return 0.0
        ctr = self._docs[doc_id]
        dl = self._doc_len[doc_id]
        norm = (1 - self.b) + self.b * (dl / max(self._avgdl, 1e-9))
        s = 0.0
        for t in query_tokens:
            tf = ctr.get(t, 0)
            if tf == 0:
                continue
            idf = self._idf(t)
            if idf <= 0:
                continue
            s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        return s

    def score_all(self, query: str) -> dict[str, float]:
        q_tokens = list(tokenize(query))
        if not q_tokens or self.N == 0:
            return {doc_id: 0.0 for doc_id in self._docs}
        return {doc_id: self.score(q_tokens, doc_id) for doc_id in self._docs}

    def _recompute_avg(self) -> None:
        if not self._doc_len:
            self._avgdl = 0.0
        else:
            self._avgdl = sum(self._doc_len.values()) / len(self._doc_len)

"""Shared types for the search engine."""

from __future__ import annotations

from typing import TypedDict


class Hit(TypedDict):
    """Single search result hit."""
    id: int
    score: float


class SearchResult(TypedDict):
    """
    Fields
    ------
    max_score
        Cosine similarity of the top-1 hit. 0.0 if the result set is empty.
    results
        List of hits ordered by descending score. Length ≤ ``top_k``.
    """

    max_score: float
    results: list[Hit]

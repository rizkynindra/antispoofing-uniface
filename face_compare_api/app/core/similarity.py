"""Cosine-similarity decision tiers shared by the API and batch CLIs."""

from typing import Literal

from app.config import settings

MatchLabel = Literal["TRUE", "Need Confirmation", "Different"]
ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


def classify_similarity(
    similarity: float | None,
    match_threshold: float = settings.SIMILARITY_MATCH_THRESHOLD,
    need_confirm_threshold: float = settings.NEED_CONFIRMATION_THRESHOLD,
) -> tuple[MatchLabel, ConfidenceLevel] | None:
    """Map a cosine similarity score onto match label and confidence.

    Uses inclusive lower bounds (>=) so API and CLI verdicts stay aligned.
    """
    if similarity is None:
        return None
    if similarity >= match_threshold:
        return "TRUE", "HIGH"
    if similarity >= need_confirm_threshold:
        return "Need Confirmation", "MEDIUM"
    return "Different", "LOW"


def classify_similarity_label(
    similarity: float | None,
    match_threshold: float = settings.SIMILARITY_MATCH_THRESHOLD,
    need_confirm_threshold: float = settings.NEED_CONFIRMATION_THRESHOLD,
) -> MatchLabel | None:
    """Return only the match label used by batch Excel output."""
    classified = classify_similarity(
        similarity,
        match_threshold=match_threshold,
        need_confirm_threshold=need_confirm_threshold,
    )
    return None if classified is None else classified[0]

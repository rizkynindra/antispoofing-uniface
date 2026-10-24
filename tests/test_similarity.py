"""Tests for cosine similarity decision tiers."""

from app.config import settings
from app.core.similarity import classify_similarity, classify_similarity_label


def test_exact_match_threshold_is_true() -> None:
    label, confidence = classify_similarity(settings.SIMILARITY_MATCH_THRESHOLD)
    assert label == "TRUE"
    assert confidence == "HIGH"


def test_just_below_match_threshold_needs_confirmation() -> None:
    score = settings.SIMILARITY_MATCH_THRESHOLD - 0.01
    assert classify_similarity_label(score) == "Need Confirmation"


def test_exact_need_confirmation_threshold() -> None:
    assert classify_similarity_label(settings.NEED_CONFIRMATION_THRESHOLD) == "Need Confirmation"


def test_below_need_confirmation_is_different() -> None:
    score = settings.NEED_CONFIRMATION_THRESHOLD - 0.01
    assert classify_similarity_label(score) == "Different"


def test_none_similarity_returns_none() -> None:
    assert classify_similarity(None) is None
    assert classify_similarity_label(None) is None


def test_custom_thresholds() -> None:
    assert classify_similarity_label(0.5, match_threshold=0.6, need_confirm_threshold=0.3) == "Need Confirmation"
    assert classify_similarity_label(0.6, match_threshold=0.6, need_confirm_threshold=0.3) == "TRUE"

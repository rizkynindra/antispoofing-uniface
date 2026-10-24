"""Tests for URL building and image-input heuristics."""

from app.config import settings
from app.core.url_utils import build_full_url, is_likely_url, is_missing_url_value


def test_build_full_url_keeps_absolute_http() -> None:
    url = "https://cdn.example.com/a.jpg"
    assert build_full_url(url) == url


def test_build_full_url_prefixes_relative_path() -> None:
    assert build_full_url("/bucket/photo.jpg", "http://s3.example") == "http://s3.example/bucket/photo.jpg"


def test_build_full_url_adds_slash_when_missing() -> None:
    assert build_full_url("bucket/photo.jpg", "http://s3.example/") == "http://s3.example/bucket/photo.jpg"


def test_build_full_url_returns_none_for_empty_and_nan() -> None:
    assert build_full_url(None) is None
    assert build_full_url("  ") is None
    assert build_full_url(float("nan")) is None
    assert is_missing_url_value("nan") is True


def test_is_likely_url_detects_http_and_paths() -> None:
    assert is_likely_url("https://example.com/x.png")
    assert is_likely_url("/var/data/face.jpg")
    assert is_likely_url("portrait.webp")
    assert not is_likely_url("not-an-image")


def test_long_payload_is_not_treated_as_url() -> None:
    bulky = "a" * (settings.URL_HEURISTIC_MAX_LENGTH + 10)
    assert is_likely_url(bulky) is False

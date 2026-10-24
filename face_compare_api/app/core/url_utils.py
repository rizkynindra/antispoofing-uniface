"""URL normalization and input-type heuristics for image sources."""

import math
from typing import Any

from app.config import settings

_MISSING_URL_TOKENS = frozenset({"", "nan", "<na>", "none", "nat"})


def is_missing_url_value(url_val: Any) -> bool:
    """Return True when a spreadsheet or payload value has no usable URL."""
    if url_val is None:
        return True
    if isinstance(url_val, float) and math.isnan(url_val):
        return True
    text = str(url_val).strip()
    return text.lower() in _MISSING_URL_TOKENS


def build_full_url(url_val: Any, base_url: str = "") -> str | None:
    """Build a full HTTP URL, prepending S3 base URL for relative paths.

    Returns None when the input is empty or missing so callers can treat
    blank Excel cells and empty strings the same way.
    """
    if is_missing_url_value(url_val):
        return None

    url_str = str(url_val).strip()
    if url_str.startswith(settings.HTTP_URL_SCHEMES):
        return url_str

    resolved_base = base_url or settings.S3_BASE_URL
    if not resolved_base:
        return url_str

    if not url_str.startswith("/"):
        url_str = f"/{url_str}"
    return resolved_base.rstrip("/") + url_str


def is_likely_url(data_str: str) -> bool:
    """Heuristic to distinguish an HTTP/S3 URL or file path from base64."""
    candidate = data_str.strip()
    if candidate.startswith(settings.URL_OR_PATH_SCHEMES):
        return True
    if len(candidate) >= settings.URL_HEURISTIC_MAX_LENGTH or "\n" in candidate:
        return False
    lower = candidate.lower()
    return candidate.startswith("/") or lower.endswith(settings.IMAGE_FILE_EXTENSIONS)

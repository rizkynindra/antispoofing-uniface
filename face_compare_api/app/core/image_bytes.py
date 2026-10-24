"""Decode raw image bytes into OpenCV BGR arrays."""

import time

import cv2
import numpy as np
import requests

from app.config import settings

_RETRYABLE_HTTP_ERRORS = (
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ConnectionError,
)


def decode_image_bytes(content: bytes) -> tuple[np.ndarray | None, str | None]:
    """Decode encoded image bytes (JPEG/PNG/etc.) into a BGR NumPy array."""
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None, "Failed to decode image"
    return image, None


def create_image_session() -> requests.Session:
    """Build a requests session with browser-like headers for object storage."""
    session = requests.Session()
    session.headers.update(settings.IMAGE_DOWNLOAD_HEADERS)
    return session


def download_and_decode_image(
    url: str,
    timeout: int = settings.REQUEST_TIMEOUT,
    max_retries: int = settings.MAX_RETRIES,
    retry_delay: float = settings.RETRY_BACKOFF_BASE_SECONDS,
    session: requests.Session | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Download an image URL and decode it in memory, with exponential backoff."""
    owns_session = session is None
    http = session or create_image_session()
    last_error: str | None = None

    try:
        for attempt in range(max_retries + 1):
            try:
                response = http.get(url, timeout=timeout)
                response.raise_for_status()
                return decode_image_bytes(response.content)
            except _RETRYABLE_HTTP_ERRORS as exc:
                last_error = f"HTTP Error: {type(exc).__name__}"
                if attempt < max_retries:
                    time.sleep(retry_delay * (2**attempt))
                    continue
                return None, f"{last_error} (after {max_retries} retries)"
            except requests.exceptions.RequestException as exc:
                return None, f"HTTP Error: {type(exc).__name__}"
            except Exception as exc:
                return None, f"Error: {type(exc).__name__}"
    finally:
        if owns_session:
            http.close()

    return None, last_error or "Unknown download error"

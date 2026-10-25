"""Universal image loader and validator supporting URL, S3, and Base64 inputs."""

import base64

import numpy as np

from app.config import settings
from app.core.image_bytes import decode_image_bytes, download_and_decode_image
from app.core.url_utils import build_full_url, is_likely_url
from app.schemas import ImageInputObject, ImageInputType

BASE64_PADDING_MODULUS = 4


def decode_base64_image(b64_str: str) -> tuple[np.ndarray | None, str | None]:
    """Decode a base64 string to an OpenCV BGR numpy array."""
    try:
        raw = b64_str.strip()
        if "," in raw:
            # Strip data-URI scheme prefix (e.g. data:image/jpeg;base64,...)
            raw = raw.split(",", 1)[1]

        missing_padding = len(raw) % BASE64_PADDING_MODULUS
        if missing_padding:
            raw += "=" * (BASE64_PADDING_MODULUS - missing_padding)

        img_bytes = base64.b64decode(raw)
        image, decode_error = decode_image_bytes(img_bytes)
        if image is None:
            return None, "Failed to decode base64 bytes into image matrix."
        return image, decode_error
    except Exception as exc:
        return None, f"Invalid base64 payload: {exc}"


def download_image_url(
    url: str,
    timeout: int = settings.REQUEST_TIMEOUT,
    max_retries: int = settings.MAX_RETRIES,
) -> tuple[np.ndarray | None, str | None]:
    """Download an image from HTTP/S3 and decode it into a BGR numpy array."""
    return download_and_decode_image(url, timeout=timeout, max_retries=max_retries)


def load_image(
    input_item: ImageInputType,
    s3_base_url: str = settings.S3_BASE_URL,
    timeout: int = settings.REQUEST_TIMEOUT,
    max_retries: int = settings.MAX_RETRIES,
) -> tuple[np.ndarray | None, str | None]:
    """Resolve and load an image from either an ImageInputObject or raw string."""
    if isinstance(input_item, ImageInputObject):
        data_type = input_item.type.lower()
        raw_data = input_item.data.strip()
    elif isinstance(input_item, str):
        raw_data = input_item.strip()
        data_type = "auto"
    else:
        return None, f"Unsupported image input type: {type(input_item)}"

    if not raw_data:
        return None, "Image data is empty."

    if data_type == "auto":
        data_type = "url" if is_likely_url(raw_data) else "base64"

    if settings.HF_SPACE and data_type == "url":
        return None, "Upload a photo file. This demo does not fetch a web address."

    if data_type == "url":
        full_url = build_full_url(raw_data, s3_base_url)
        if not full_url:
            return None, "Image URL is empty."
        return download_image_url(full_url, timeout=timeout, max_retries=max_retries)
    if data_type == "base64":
        return decode_base64_image(raw_data)
    return None, f"Unknown image data type: {data_type}"

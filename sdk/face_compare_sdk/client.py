"""HTTP client for POST /v1/compare and GET /health."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from face_compare_sdk.errors import FaceCompareError
from face_compare_sdk.models import CompareResult, HealthStatus, ImageMetadata

DEFAULT_BASE_URL = "http://127.0.0.1:2112"
DEFAULT_TIMEOUT_SECONDS = 60.0
USER_AGENT = "face-compare-sdk/1.0.0"
REMOTE_PREFIXES = ("http://", "https://", "s3://", "data:")

ImageSource = str | bytes | os.PathLike[str]


class FaceCompareClient:
    """Call a running face comparison service.

    Pass a URL, a base64 string, raw image bytes, or a path to a local file.
    Local files are sent as base64. URLs and S3 keys are forwarded as-is so
    the service downloads them.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        channel_id: str = "UNKNOWN",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.channel_id = channel_id
        self.timeout = timeout

    def __enter__(self) -> FaceCompareClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def health(self) -> HealthStatus:
        """Return service status, including whether it is on GPU or CPU."""
        payload, _trace, _transaction = self._request("GET", "/health")
        return HealthStatus(
            status=str(payload.get("status", "")),
            service=str(payload.get("service", "")),
            version=str(payload.get("version", "")),
            device=str(payload.get("device", "")),
            environment=str(payload.get("environment", "")),
        )

    def compare(
        self,
        image1: ImageSource,
        image2: ImageSource,
        *,
        channel_id: str | None = None,
        check_liveness_image1: bool = False,
        check_liveness_image2: bool = False,
        similarity_threshold: float | None = None,
        need_confirmation_threshold: float | None = None,
        trace_id: str | None = None,
        prev_transaction_id: str | None = None,
    ) -> CompareResult:
        """Compare image1 (reference) with image2 (photo being checked)."""
        resolved_channel = channel_id or self.channel_id
        body: dict[str, Any] = {
            "ChId": resolved_channel,
            "image1": encode_image(image1),
            "image2": encode_image(image2),
        }
        options = _options_payload(
            check_liveness_image1=check_liveness_image1,
            check_liveness_image2=check_liveness_image2,
            similarity_threshold=similarity_threshold,
            need_confirmation_threshold=need_confirmation_threshold,
        )
        if options:
            body["options"] = options

        headers: dict[str, str] = {"X-ChId": resolved_channel}
        if trace_id:
            headers["X-TRACE-ID"] = trace_id
        if prev_transaction_id:
            headers["X-PREV-TRANS-ID"] = prev_transaction_id

        payload, response_trace, transaction_id = self._request(
            "POST",
            "/v1/compare",
            body=body,
            headers=headers,
        )
        return _parse_compare_result(payload, response_trace, transaction_id)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], str, str]:
        request_headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                status = response.status
                trace_id = response.headers.get("X-TRACE-ID", "")
                transaction_id = response.headers.get("X-TRANSACTION-ID", "")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
            trace_id = exc.headers.get("X-TRACE-ID", "") if exc.headers else ""
            transaction_id = exc.headers.get("X-TRANSACTION-ID", "") if exc.headers else ""
        except urllib.error.URLError as exc:
            raise FaceCompareError(
                f"Could not reach {self.base_url}{path}: {exc.reason}",
                error_code="TRANSPORT",
            ) from exc

        payload = _decode_json(raw, status, trace_id, transaction_id)
        if status >= 400 or payload.get("status") == "FAILED":
            raise _error_from_payload(payload, status, trace_id, transaction_id)
        return payload, trace_id, transaction_id


def encode_image(source: ImageSource) -> str | dict[str, str]:
    """Turn a caller image into the JSON value POST /v1/compare expects."""
    if isinstance(source, bytes):
        return _base64_payload(source)
    if isinstance(source, os.PathLike):
        return _file_payload(Path(source))
    if not isinstance(source, str):
        raise FaceCompareError(
            f"Unsupported image input type: {type(source).__name__}",
            error_code="INVALID_IMAGE",
        )

    text = source.strip()
    if not text:
        raise FaceCompareError("Image data is empty.", error_code="EMPTY_IMAGE")
    if text.startswith(REMOTE_PREFIXES):
        return text

    path = Path(text)
    if path.is_file():
        return _file_payload(path)
    return text


def _file_payload(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FaceCompareError(
            f"Image file does not exist: {path}",
            error_code="IMAGE_NOT_FOUND",
        )
    return _base64_payload(path.read_bytes())


def _base64_payload(raw: bytes) -> dict[str, str]:
    if not raw:
        raise FaceCompareError("Image data is empty.", error_code="EMPTY_IMAGE")
    return {"type": "base64", "data": base64.b64encode(raw).decode("ascii")}


def _options_payload(
    *,
    check_liveness_image1: bool,
    check_liveness_image2: bool,
    similarity_threshold: float | None,
    need_confirmation_threshold: float | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if check_liveness_image1:
        options["check_liveness_image1"] = True
    if check_liveness_image2:
        options["check_liveness_image2"] = True
    if similarity_threshold is not None:
        options["similarity_threshold"] = similarity_threshold
    if need_confirmation_threshold is not None:
        options["need_confirmation_threshold"] = need_confirmation_threshold
    return options


def _decode_json(
    raw: bytes,
    status: int,
    trace_id: str,
    transaction_id: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FaceCompareError(
            f"Response was not JSON (HTTP {status}).",
            error_code="INVALID_RESPONSE",
            http_status=status,
            trace_id=trace_id,
            transaction_id=transaction_id,
        ) from exc
    if not isinstance(payload, dict):
        raise FaceCompareError(
            f"Response JSON was not an object (HTTP {status}).",
            error_code="INVALID_RESPONSE",
            http_status=status,
            trace_id=trace_id,
            transaction_id=transaction_id,
        )
    return payload


def _error_from_payload(
    payload: dict[str, Any],
    status: int,
    trace_id: str,
    transaction_id: str,
) -> FaceCompareError:
    message = str(payload.get("message") or f"Request failed with HTTP {status}.")
    return FaceCompareError(
        message,
        error_code=str(payload.get("error_code") or f"HTTP_{status}"),
        http_status=status,
        channel_id=str(payload.get("ChId") or "UNKNOWN"),
        image1=_parse_metadata(payload.get("image1_metadata")),
        image2=_parse_metadata(payload.get("image2_metadata")),
        execution_time_ms=_optional_float(payload.get("execution_time_ms")),
        trace_id=trace_id,
        transaction_id=transaction_id,
    )


def _parse_compare_result(
    payload: dict[str, Any],
    trace_id: str,
    transaction_id: str,
) -> CompareResult:
    result = payload.get("result")
    image1 = _parse_metadata(payload.get("image1_metadata"))
    image2 = _parse_metadata(payload.get("image2_metadata"))
    if not isinstance(result, dict) or image1 is None or image2 is None:
        raise FaceCompareError(
            "Comparison response is missing the verdict or image metadata.",
            error_code="INVALID_RESPONSE",
            http_status=200,
            channel_id=str(payload.get("ChId") or "UNKNOWN"),
            image1=image1,
            image2=image2,
            trace_id=trace_id,
            transaction_id=transaction_id,
        )
    return CompareResult(
        channel_id=str(payload.get("ChId") or "UNKNOWN"),
        is_match=str(result.get("is_match") or ""),
        similarity_score=float(result.get("similarity_score")),
        similarity_percentage=str(result.get("similarity_percentage") or ""),
        confidence_level=str(result.get("confidence_level") or ""),
        image1=image1,
        image2=image2,
        execution_time_ms=float(payload.get("execution_time_ms") or 0.0),
        message=str(payload.get("message") or ""),
        trace_id=trace_id,
        transaction_id=transaction_id,
    )


def _parse_metadata(value: Any) -> ImageMetadata | None:
    if not isinstance(value, dict):
        return None
    bbox = value.get("bbox")
    return ImageMetadata(
        faces_detected=int(value.get("faces_detected") or 0),
        bbox=tuple(float(item) for item in bbox) if isinstance(bbox, list) else None,
        is_live=value.get("is_live"),
        liveness_score=_optional_float(value.get("liveness_score")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

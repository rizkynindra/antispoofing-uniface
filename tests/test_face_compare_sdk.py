"""The face comparison client speaks the HTTP API without loading models."""

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from face_compare_sdk import FaceCompareClient, FaceCompareError
from face_compare_sdk.client import encode_image


class _ApiHandler(BaseHTTPRequestHandler):
    routes: dict

    def do_GET(self) -> None:
        self._reply(self.routes.get(("GET", self.path)))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        self.server.last_body = json.loads(raw.decode("utf-8"))  # type: ignore[attr-defined]
        self.server.last_headers = dict(self.headers)  # type: ignore[attr-defined]
        self._reply(self.routes.get(("POST", self.path)))

    def _reply(self, route) -> None:
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        status, payload = route
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-TRACE-ID", "TC-test")
        self.send_header("X-TRANSACTION-ID", "TR-test")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


@pytest.fixture
def api_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ApiHandler)
    server.routes = {}  # type: ignore[attr-defined]
    _ApiHandler.routes = server.routes  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield server, base_url
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_encode_image_keeps_urls_and_encodes_local_files(tmp_path: Path) -> None:
    photo = tmp_path / "live.jpg"
    photo.write_bytes(b"jpeg-bytes")

    assert encode_image("https://example.test/a.jpg") == "https://example.test/a.jpg"
    assert encode_image("kyc/ref.jpg") == "kyc/ref.jpg"
    encoded = encode_image(photo)
    assert encoded == {
        "type": "base64",
        "data": base64.b64encode(b"jpeg-bytes").decode("ascii"),
    }
    with pytest.raises(FaceCompareError) as missing:
        encode_image(tmp_path / "missing.jpg")
    assert missing.value.error_code == "IMAGE_NOT_FOUND"


def test_health_reads_device(api_server) -> None:
    server, base_url = api_server
    server.routes[("GET", "/health")] = (
        200,
        {
            "status": "UP",
            "service": "svc-face-compare",
            "version": "1.0.0",
            "device": "GPU (CUDA)",
            "environment": "production",
        },
    )

    health = FaceCompareClient(base_url).health()

    assert health.device == "GPU (CUDA)"
    assert health.status == "UP"


def test_compare_posts_channel_liveness_and_trace(api_server, tmp_path: Path) -> None:
    server, base_url = api_server
    photo = tmp_path / "live.jpg"
    photo.write_bytes(b"jpeg-bytes")
    server.routes[("POST", "/v1/compare")] = (
        200,
        {
            "status": "SUCCESS",
            "ChId": "PORTAL_KLAIM",
            "result": {
                "is_match": "TRUE",
                "similarity_score": 0.8642,
                "similarity_percentage": "86.42%",
                "confidence_level": "HIGH",
            },
            "image1_metadata": {"faces_detected": 1, "bbox": [1, 2, 3, 4], "is_live": None},
            "image2_metadata": {
                "faces_detected": 1,
                "bbox": [5, 6, 7, 8],
                "is_live": True,
                "liveness_score": 0.99,
            },
            "execution_time_ms": 68.4,
            "message": "done",
        },
    )

    verdict = FaceCompareClient(base_url, channel_id="PORTAL_KLAIM").compare(
        "kyc/ref.jpg",
        photo,
        check_liveness_image2=True,
        similarity_threshold=0.8,
        trace_id="caller-trace",
    )

    assert verdict.is_match == "TRUE"
    assert verdict.similarity_score == 0.8642
    assert verdict.image2.is_live is True
    assert verdict.trace_id == "TC-test"
    assert verdict.transaction_id == "TR-test"
    assert server.last_body["ChId"] == "PORTAL_KLAIM"
    assert server.last_body["image1"] == "kyc/ref.jpg"
    assert server.last_body["image2"]["type"] == "base64"
    assert server.last_body["options"] == {
        "check_liveness_image2": True,
        "similarity_threshold": 0.8,
    }
    assert server.last_headers["X-Trace-Id"] == "caller-trace"
    assert server.last_headers["X-Chid"] == "PORTAL_KLAIM"


def test_compare_raises_api_error_with_metadata(api_server) -> None:
    server, base_url = api_server
    server.routes[("POST", "/v1/compare")] = (
        400,
        {
            "status": "FAILED",
            "ChId": "PORTAL_KLAIM",
            "error_code": "NO_FACE_IN_IMAGE_1",
            "message": "No face detected in Image 1 (Reference photo).",
            "image1_metadata": {"faces_detected": 0},
            "image2_metadata": {"faces_detected": 1, "bbox": [1, 2, 3, 4]},
            "execution_time_ms": 12.5,
        },
    )

    with pytest.raises(FaceCompareError) as caught:
        FaceCompareClient(base_url).compare("http://example.test/a.jpg", "http://example.test/b.jpg")

    error = caught.value
    assert error.error_code == "NO_FACE_IN_IMAGE_1"
    assert error.http_status == 400
    assert error.image1 is not None
    assert error.image1.faces_detected == 0
    assert error.trace_id == "TC-test"


def test_compare_reports_a_closed_port() -> None:
    client = FaceCompareClient("http://127.0.0.1:1")
    with pytest.raises(FaceCompareError) as caught:
        client.health()
    assert caught.value.error_code == "TRANSPORT"

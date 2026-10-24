"""Tests for compare error mapping used by the HTTP layer."""

from fastapi import status

from app.compare_errors import internal_failure, invalid_image_failure, no_face_failure
from app.schemas import SingleImageMetadata


def test_invalid_image_failure_codes() -> None:
    first = invalid_image_failure(1, "timeout")
    assert first.error_code == "INVALID_IMAGE_1"
    assert first.http_status == status.HTTP_400_BAD_REQUEST
    assert "Image 1" in first.message
    assert first.log_level == "ERROR"

    second = invalid_image_failure(2, "decode failed")
    assert second.error_code == "INVALID_IMAGE_2"
    assert "Image 2" in second.message


def test_no_face_failure_includes_metadata() -> None:
    meta1 = SingleImageMetadata(faces_detected=0)
    meta2 = SingleImageMetadata(faces_detected=1, bbox=[1.0, 2.0, 3.0, 4.0])
    failure = no_face_failure(1, meta1, meta2)
    assert failure.error_code == "NO_FACE_IN_IMAGE_1"
    assert failure.res_desc == "NO_DETECTION"
    assert failure.log_level == "WARN"
    assert failure.image1_metadata is meta1
    assert failure.image2_metadata is meta2


def test_internal_failure_is_server_error() -> None:
    failure = internal_failure("boom")
    assert failure.error_code == "INTERNAL_SERVER_ERROR"
    assert failure.http_status == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "boom" in failure.message

"""Typed comparison failures used by the service layer and HTTP mapping."""

from dataclasses import dataclass

from fastapi import status

from app.schemas import SingleImageMetadata


@dataclass(frozen=True)
class CompareFailure:
    """Business-level comparison failure that maps to a JSON error response."""

    error_code: str
    message: str
    http_status: int
    log_level: str
    res_desc: str
    image1_metadata: SingleImageMetadata | None = None
    image2_metadata: SingleImageMetadata | None = None


def invalid_image_failure(image_index: int, detail: str | None) -> CompareFailure:
    """Build a failure when an input image cannot be loaded or decoded."""
    label = "Image 1 (Reference)" if image_index == 1 else "Image 2 (Target)"
    code = f"INVALID_IMAGE_{image_index}"
    return CompareFailure(
        error_code=code,
        message=f"Failed to load {label}: {detail}",
        http_status=status.HTTP_400_BAD_REQUEST,
        log_level="ERROR",
        res_desc="FAILED",
    )


def no_face_failure(
    image_index: int,
    image1_metadata: SingleImageMetadata,
    image2_metadata: SingleImageMetadata,
) -> CompareFailure:
    """Build a failure when detection finds no face in one of the images."""
    if image_index == 1:
        message = "No face detected in Image 1 (Reference photo)."
        code = "NO_FACE_IN_IMAGE_1"
    else:
        message = "No face detected in Image 2 (Target/Live photo)."
        code = "NO_FACE_IN_IMAGE_2"
    return CompareFailure(
        error_code=code,
        message=message,
        http_status=status.HTTP_400_BAD_REQUEST,
        log_level="WARN",
        res_desc="NO_DETECTION",
        image1_metadata=image1_metadata,
        image2_metadata=image2_metadata,
    )


def internal_failure(detail: str) -> CompareFailure:
    """Build a failure for unexpected processing errors."""
    return CompareFailure(
        error_code="INTERNAL_SERVER_ERROR",
        message=f"Internal server error during face comparison: {detail}",
        http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        log_level="ERROR",
        res_desc="ERROR",
    )

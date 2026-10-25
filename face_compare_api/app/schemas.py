"""Pydantic schemas for Face Comparison API."""

from typing import Literal, Union

from pydantic import BaseModel, Field

from app.config import settings


class ImageInputObject(BaseModel):
    """Flexible image input specification."""
    type: Literal["url", "base64", "auto"] = Field(
        default="auto",
        description="Type of input: 'url' (HTTP/S3), 'base64' (raw base64 string), or 'auto' (detected automatically)."
    )
    data: str = Field(
        ...,
        description="Image URL or base64 encoded string."
    )


# Allow image1/image2 to be either ImageInputObject or directly a URL/base64 string
ImageInputType = Union[ImageInputObject, str]


class CompareOptions(BaseModel):
    """Execution options for face comparison."""
    check_liveness_image1: bool = Field(
        default=False,
        description="Whether to perform anti-spoofing / liveness check on Image 1 (Reference)."
    )
    check_liveness_image2: bool = Field(
        default=False,
        description="Whether to perform anti-spoofing / liveness check on Image 2 (Live/Target)."
    )
    similarity_threshold: float = Field(
        default_factory=lambda: settings.SIMILARITY_MATCH_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Threshold score above which result is classified as 'TRUE'.",
    )
    need_confirmation_threshold: float = Field(
        default_factory=lambda: settings.NEED_CONFIRMATION_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Threshold score above which result is classified as 'Need Confirmation'.",
    )


class FaceCompareRequest(BaseModel):
    """Request payload for /v1/compare."""
    ChId: str = Field(
        default="UNKNOWN",
        description="Caller channel id. Echoed on the response."
    )
    image1: ImageInputType = Field(
        ...,
        description="First image (Reference photo). Can be URL string, Base64 string, or ImageInputObject."
    )
    image2: ImageInputType = Field(
        ...,
        description="Second image (Live / Target photo). Can be URL string, Base64 string, or ImageInputObject."
    )
    options: CompareOptions = Field(
        default_factory=CompareOptions,
        description="Optional comparison configuration."
    )


class SingleImageMetadata(BaseModel):
    """Metadata extracted from a single image."""
    faces_detected: int = Field(..., description="Number of faces detected in the image.")
    bbox: list[float] | None = Field(default=None, description="Bounding box [x1, y1, x2, y2] of the primary face.")
    is_live: bool | None = Field(default=None, description="Anti-spoofing verdict (True if real, False if spoof).")
    liveness_score: float | None = Field(default=None, description="Anti-spoofing confidence score.")


class CompareResultData(BaseModel):
    """Comparison evaluation result."""
    is_match: Literal["TRUE", "Need Confirmation", "Different"] = Field(
        ...,
        description="Match classification verdict based on thresholds."
    )
    similarity_score: float = Field(
        ...,
        description="Calculated cosine similarity score between 0.0 and 1.0."
    )
    similarity_percentage: str = Field(
        ...,
        description="Formatted percentage representation of similarity score (e.g. '86.42%')."
    )
    confidence_level: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ...,
        description="Confidence category based on similarity magnitude."
    )


class FaceCompareResponse(BaseModel):
    """Standard success response payload for /v1/compare."""
    status: Literal["SUCCESS", "FAILED"] = Field(default="SUCCESS")
    ChId: str = Field(..., description="Channel ID passed in the request.")
    result: CompareResultData | None = Field(default=None, description="Comparison score and classification.")
    image1_metadata: SingleImageMetadata = Field(..., description="Metadata for image 1 (Reference).")
    image2_metadata: SingleImageMetadata = Field(..., description="Metadata for image 2 (Target/Live).")
    execution_time_ms: float = Field(..., description="Service processing duration in milliseconds.")
    message: str = Field(default="", description="Summary message of the operation.")


class ErrorResponse(BaseModel):
    """Standard error response payload."""
    status: Literal["FAILED"] = Field(default="FAILED")
    ChId: str = Field(default="UNKNOWN")
    error_code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error explanation.")
    image1_metadata: SingleImageMetadata | None = Field(default=None)
    image2_metadata: SingleImageMetadata | None = Field(default=None)
    execution_time_ms: float | None = Field(default=None)

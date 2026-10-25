"""Response objects returned by the face comparison API."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageMetadata:
    """Detection and liveness details for one input photo."""

    faces_detected: int
    bbox: tuple[float, ...] | None = None
    is_live: bool | None = None
    liveness_score: float | None = None


@dataclass(frozen=True)
class CompareResult:
    """Successful 1:1 comparison."""

    channel_id: str
    is_match: str
    similarity_score: float
    similarity_percentage: str
    confidence_level: str
    image1: ImageMetadata
    image2: ImageMetadata
    execution_time_ms: float
    message: str
    trace_id: str
    transaction_id: str


@dataclass(frozen=True)
class HealthStatus:
    """GET /health payload."""

    status: str
    service: str
    version: str
    device: str
    environment: str

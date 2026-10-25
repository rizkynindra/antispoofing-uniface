"""Client for the face comparison HTTP API."""

from face_compare_sdk.client import FaceCompareClient
from face_compare_sdk.errors import FaceCompareError
from face_compare_sdk.models import CompareResult, HealthStatus, ImageMetadata

__all__ = [
    "CompareResult",
    "FaceCompareClient",
    "FaceCompareError",
    "HealthStatus",
    "ImageMetadata",
]

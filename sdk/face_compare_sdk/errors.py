"""Errors raised when a comparison call does not return a verdict."""

from face_compare_sdk.models import ImageMetadata


class FaceCompareError(Exception):
    """The API rejected the call, or the response could not be read.

    Client-side problems, such as an empty image, use http_status None.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        http_status: int | None = None,
        channel_id: str = "UNKNOWN",
        image1: ImageMetadata | None = None,
        image2: ImageMetadata | None = None,
        execution_time_ms: float | None = None,
        trace_id: str = "",
        transaction_id: str = "",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.http_status = http_status
        self.channel_id = channel_id
        self.image1 = image1
        self.image2 = image2
        self.execution_time_ms = execution_time_ms
        self.trace_id = trace_id
        self.transaction_id = transaction_id

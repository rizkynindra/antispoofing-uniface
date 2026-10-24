"""HTTP request identifiers and BPJS START/STOP log helpers."""

import time
import uuid
from dataclasses import dataclass

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.compare_errors import CompareFailure
from app.config import MILLISECONDS_PER_SECOND
from app.schemas import ErrorResponse, FaceCompareResponse
from app.utils.standard_logger import StandardLogger

TRACE_ID_PREFIX = "TC-"
TRANSACTION_ID_PREFIX = "TR-"
DEFAULT_CLIENT_IP = "127.0.0.1"
UNKNOWN_CHANNEL = "UNKNOWN"
DEFAULT_CALLER = "FastAPI"
COMPARE_ENDPOINT = "/v1/compare"
COMPARE_FUNCTION_NAME = "compare"


def get_client_ip(request: Request) -> str:
    """Extract client IP: X-Forwarded-For first hop, then X-Real-IP, then peer."""
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()

    x_real_ip = request.headers.get("X-Real-IP")
    if x_real_ip:
        return x_real_ip.strip()

    if request.client and request.client.host:
        return request.client.host
    return DEFAULT_CLIENT_IP


def get_or_generate_trace_id(request: Request) -> str:
    """Return TC-[UUID] from header or generate a new UUID v4."""
    trace_header = request.headers.get("X-TRACE-ID") or request.headers.get("x-trace-id")
    if trace_header:
        trace_str = trace_header.strip()
        if trace_str.upper().startswith(TRACE_ID_PREFIX):
            return trace_str
        return f"{TRACE_ID_PREFIX}{trace_str}"
    return f"{TRACE_ID_PREFIX}{uuid.uuid4()}"


def get_prev_transaction_id(request: Request) -> str:
    """Return TR-[UUID] from X-PREV-TRANS-ID when present."""
    prev_header = request.headers.get("X-PREV-TRANS-ID") or request.headers.get("x-prev-trans-id")
    if not prev_header:
        return ""
    prev_str = prev_header.strip()
    if prev_str.upper().startswith(TRANSACTION_ID_PREFIX):
        return prev_str
    return f"{TRANSACTION_ID_PREFIX}{prev_str}"


def resolve_channel_id(payload_ch_id: str | None, request: Request) -> str:
    """Resolve channel ID from payload, then headers, then UNKNOWN."""
    return (
        payload_ch_id
        or request.headers.get("X-ChId")
        or request.headers.get("X-Channel-Id")
        or UNKNOWN_CHANNEL
    )


def trace_response_headers(trace_id: str, transaction_id: str) -> dict[str, str]:
    """Headers echoed back so callers can correlate logs."""
    return {"X-TRACE-ID": trace_id, "X-TRANSACTION-ID": transaction_id}


@dataclass
class CompareLogContext:
    """Request-scoped values needed for START/STOP system logs."""

    logger: StandardLogger
    transaction_id: str
    trace_id: str
    prev_trans_id: str
    ch_id: str
    caller_info: str
    content_type: str
    client_ip: str
    start_time: float

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * MILLISECONDS_PER_SECOND

    def log_start(self) -> None:
        self.logger.write_system_log(
            log_level="INFO",
            transaction_id=self.transaction_id,
            trace_id=self.trace_id,
            endpoint=COMPARE_ENDPOINT,
            method_type="POST",
            content_type=self.content_type,
            function_name=COMPARE_FUNCTION_NAME,
            caller_info=self.caller_info,
            execution_time=0.0,
            client_ip=self.client_ip,
            event_name="Comparison Request",
            activity="Start Prediction",
            obj_name="Image",
            res_desc="PENDING",
            short_desc=f"Processing request for ChId: {self.ch_id}",
            flag_start_stop="[START]",
            error="",
            prev_transaction_id=self.prev_trans_id,
        )

    def log_stop(
        self,
        *,
        log_level: str,
        res_desc: str,
        short_desc: str,
        error: str = "",
        extra_detail: str = "",
        execution_time_ms: float | None = None,
    ) -> float:
        elapsed = self.elapsed_ms() if execution_time_ms is None else execution_time_ms
        self.logger.write_system_log(
            log_level=log_level,
            transaction_id=self.transaction_id,
            trace_id=self.trace_id,
            endpoint=COMPARE_ENDPOINT,
            method_type="POST",
            content_type=self.content_type,
            function_name=COMPARE_FUNCTION_NAME,
            caller_info=self.caller_info,
            execution_time=elapsed,
            client_ip=self.client_ip,
            event_name="Comparison Result",
            activity="Finish Prediction",
            obj_name="Image",
            res_desc=res_desc,
            short_desc=short_desc,
            flag_start_stop="[STOP]",
            extra_detail=extra_detail,
            error=error,
            prev_transaction_id=self.prev_trans_id,
        )
        return elapsed

    def failure_response(self, failure: CompareFailure) -> JSONResponse:
        elapsed = self.elapsed_ms()
        http_prefix = "500" if failure.http_status >= status.HTTP_500_INTERNAL_SERVER_ERROR else "400"
        self.log_stop(
            log_level=failure.log_level,
            res_desc=failure.res_desc,
            short_desc=failure.message,
            error=f"{http_prefix}: {failure.message}",
            execution_time_ms=elapsed,
        )
        payload = ErrorResponse(
            status="FAILED",
            ChId=self.ch_id,
            error_code=failure.error_code,
            message=failure.message,
            image1_metadata=failure.image1_metadata,
            image2_metadata=failure.image2_metadata,
            execution_time_ms=round(elapsed, 2),
        )
        return JSONResponse(
            status_code=failure.http_status,
            content=payload.model_dump(),
            headers=trace_response_headers(self.trace_id, self.transaction_id),
        )

    def success_response(self, response: FaceCompareResponse) -> JSONResponse:
        result = response.result
        match_label = result.is_match if result else "SUCCESS"
        similarity = result.similarity_score if result else 0.0
        self.log_stop(
            log_level="INFO",
            res_desc=match_label,
            short_desc=(
                f"Comparison result: {match_label}. Similarity: {similarity:.4f}. "
                f"Match: {match_label == 'TRUE'}"
            ),
            extra_detail=f"Max Confidence: {similarity:.4f}",
            execution_time_ms=response.execution_time_ms,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response.model_dump(),
            headers=trace_response_headers(self.trace_id, self.transaction_id),
        )


def build_compare_log_context(
    request: Request,
    logger: StandardLogger,
    ch_id: str,
) -> CompareLogContext:
    """Collect identifiers used for one /v1/compare invocation."""
    caller_info = ch_id if ch_id and ch_id != UNKNOWN_CHANNEL else DEFAULT_CALLER
    return CompareLogContext(
        logger=logger,
        transaction_id=f"{TRANSACTION_ID_PREFIX}{uuid.uuid4()}",
        trace_id=get_or_generate_trace_id(request),
        prev_trans_id=get_prev_transaction_id(request),
        ch_id=ch_id,
        caller_info=caller_info,
        content_type=request.headers.get("Content-Type", "application/json"),
        client_ip=get_client_ip(request),
        start_time=time.perf_counter(),
    )

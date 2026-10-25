"""FastAPI application for face comparison with BPJS Standard Log."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import settings
from app.request_context import build_compare_log_context, resolve_channel_id
from app.schemas import ErrorResponse, FaceCompareRequest, FaceCompareResponse
from app.services.compare_service import CompareFailure, run_face_compare
from app.services.face_engine import face_engine
from app.utils.standard_logger import StandardLogger

logger = StandardLogger(
    service_name=settings.SERVICE_NAME,
    log_dir=settings.LOG_DIR,
    log_file_pattern=settings.LOG_FILE_PATTERN,
)

UI_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Preload ML models at server startup so the first request is not cold."""
    _ = face_engine.device
    yield


app = FastAPI(
    title="BPJS Ketenagakerjaan Face Comparison API",
    description="High-performance Face Comparison Service using UniFace with Standard Log 6.1.1 compliance.",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def face_compare_page() -> FileResponse:
    """Serve the operator page for 1:1 face comparison."""
    return FileResponse(UI_INDEX, headers={"Cache-Control": "no-cache"})


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """Service health and device status check."""
    return {
        "status": "UP",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "device": face_engine.device,
        "environment": settings.ENVIRONMENT,
    }


@app.post(
    "/v1/compare",
    response_model=FaceCompareResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Face Comparison"],
)
async def compare_faces(request: Request, payload: FaceCompareRequest) -> Any:
    """Compare two facial images (1:1) with optional anti-spoofing and standard logging."""
    ch_id = resolve_channel_id(payload.ChId, request)
    log_context = build_compare_log_context(request, logger, ch_id)
    log_context.log_start()

    outcome = run_face_compare(payload)
    if isinstance(outcome, CompareFailure):
        return log_context.failure_response(outcome)

    elapsed_ms = log_context.elapsed_ms()
    response_payload = FaceCompareResponse(
        status="SUCCESS",
        ChId=ch_id,
        result=outcome.result,
        image1_metadata=outcome.image1_metadata,
        image2_metadata=outcome.image2_metadata,
        execution_time_ms=round(elapsed_ms, 2),
        message=outcome.message,
    )
    return log_context.success_response(response_payload)

"""FastAPI application for face comparison with BPJS Standard Log."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response

from app.config import settings
from app.gallery_routes import router as gallery_router
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
_SPACE_HIDE_START = "<!-- space-hide-start -->"
_SPACE_HIDE_END = "<!-- space-hide-end -->"
_PRIVATE_API_TITLE = "Face Comparison API"
_PRIVATE_API_DESCRIPTION = (
    "High-performance Face Comparison Service using UniFace with Standard Log 6.1.1 compliance."
)
_SPACE_API_TITLE = "Face comparison"
_SPACE_API_DESCRIPTION = "Compare two uploaded face photos."


def api_metadata(space: bool) -> tuple[str, str]:
    """OpenAPI title and description. The public Space does not name the private deployment."""
    if space:
        return _SPACE_API_TITLE, _SPACE_API_DESCRIPTION
    return _PRIVATE_API_TITLE, _PRIVATE_API_DESCRIPTION


def page_for_space(html: str) -> str:
    """Drop address fields and the channel box so the Space only accepts uploaded photos."""
    while _SPACE_HIDE_START in html and _SPACE_HIDE_END in html:
        before, _, rest = html.partition(_SPACE_HIDE_START)
        _, _, after = rest.partition(_SPACE_HIDE_END)
        html = before + after
    return html.replace(
        '<html lang="en" data-theme="light">',
        '<html lang="en" data-theme="light" data-upload-only="true">',
        1,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Preload ML models at server startup so the first request is not cold."""
    _ = face_engine.device
    yield


_api_title, _api_description = api_metadata(settings.HF_SPACE)
app = FastAPI(
    title=_api_title,
    description=_api_description,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

app.include_router(gallery_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def face_compare_page() -> Response:
    """Serve the operator page. On the Space, the page only offers file upload."""
    headers = {"Cache-Control": "no-cache"}
    if not settings.HF_SPACE:
        return FileResponse(UI_INDEX, headers=headers)
    html = page_for_space(UI_INDEX.read_text(encoding="utf-8"))
    return HTMLResponse(html, headers=headers)


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

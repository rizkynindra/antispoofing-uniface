"""HTTP routes for enrolling reference faces and searching a selfie."""

import hmac
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.gallery_schemas import (
    GalleryDeleteResponse,
    GalleryEnrollRequest,
    GalleryEnrollResponse,
    GalleryListResponse,
    GalleryPerson,
    GallerySearchRequest,
    GallerySearchResponse,
)
from app.schemas import ErrorResponse
from app.services.face_engine import face_engine
from app.services.face_gallery import FaceGallery
from app.services.gallery_service import GalleryFailure, enroll_person, search_gallery, to_person

router = APIRouter(prefix="/v1/gallery", tags=["Face Gallery"])
ADMIN_TOKEN_HEADER = "X-Admin-Token"

_gallery: FaceGallery | None = None


def get_gallery() -> FaceGallery:
    """Return the process gallery rooted at FACE_GALLERY_DIR."""
    global _gallery
    if _gallery is None:
        _gallery = FaceGallery(Path(settings.FACE_GALLERY_DIR))
    return _gallery


def _space_blocked() -> JSONResponse | None:
    if not settings.HF_SPACE:
        return None
    return _error(
        GalleryFailure(
            "NOT_AVAILABLE",
            "The face gallery is not part of this demo.",
            404,
        )
    )


def _admin_blocked(request: Request) -> JSONResponse | None:
    expected = settings.FACE_GALLERY_ADMIN_TOKEN
    if not expected:
        return None
    provided = request.headers.get(ADMIN_TOKEN_HEADER, "")
    if hmac.compare_digest(provided, expected):
        return None
    return _error(
        GalleryFailure(
            "UNAUTHORIZED",
            "Send the admin token in the X-Admin-Token header.",
            401,
        ),
        ch_id="UNKNOWN",
    )


def _error(failure: GalleryFailure, ch_id: str = "UNKNOWN") -> JSONResponse:
    payload = ErrorResponse(
        status="FAILED",
        ChId=ch_id,
        error_code=failure.error_code,
        message=failure.message,
    )
    return JSONResponse(status_code=failure.http_status, content=payload.model_dump())


@router.post(
    "",
    response_model=GalleryEnrollResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def enroll_face(request: Request, payload: GalleryEnrollRequest) -> GalleryEnrollResponse | JSONResponse:
    """Store a reference photo and its embedding. An admin calls this once per person."""
    blocked = _space_blocked() or _admin_blocked(request)
    if blocked is not None:
        return blocked
    outcome = enroll_person(payload, gallery=get_gallery(), engine=face_engine)
    if isinstance(outcome, GalleryFailure):
        return _error(outcome, payload.ChId)
    return outcome


@router.get("", response_model=GalleryListResponse, responses={401: {"model": ErrorResponse}})
def list_faces(request: Request) -> GalleryListResponse | JSONResponse:
    """List enrolled people. Embeddings and photo bytes stay on disk."""
    blocked = _space_blocked() or _admin_blocked(request)
    if blocked is not None:
        return blocked
    people = [to_person(entry) for entry in get_gallery().list_entries()]
    return GalleryListResponse(count=len(people), people=people)


@router.post(
    "/search",
    response_model=GallerySearchResponse,
    responses={400: {"model": ErrorResponse}},
)
def search_face(payload: GallerySearchRequest) -> GallerySearchResponse | JSONResponse:
    """Match one selfie against every enrolled face."""
    blocked = _space_blocked()
    if blocked is not None:
        return blocked
    outcome = search_gallery(payload, gallery=get_gallery(), engine=face_engine)
    if isinstance(outcome, GalleryFailure):
        return _error(outcome, payload.ChId)
    return outcome


@router.get(
    "/{person_id}/photo",
    response_model=None,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def read_photo(person_id: str, request: Request) -> FileResponse | JSONResponse:
    """Return the stored reference JPEG."""
    blocked = _space_blocked() or _admin_blocked(request)
    if blocked is not None:
        return blocked
    try:
        path = get_gallery().photo_path(person_id)
    except ValueError as exc:
        return _error(GalleryFailure("INVALID_PERSON_ID", str(exc), 400))
    if path is None:
        return _error(GalleryFailure("NOT_FOUND", f"{person_id} is not in the face gallery.", 404))
    return FileResponse(path, media_type="image/jpeg")


@router.get(
    "/{person_id}",
    response_model=GalleryPerson,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def read_face(person_id: str, request: Request) -> GalleryPerson | JSONResponse:
    """Return one enrolled person."""
    blocked = _space_blocked() or _admin_blocked(request)
    if blocked is not None:
        return blocked
    try:
        entry = get_gallery().get(person_id)
    except ValueError as exc:
        return _error(GalleryFailure("INVALID_PERSON_ID", str(exc), 400))
    if entry is None:
        return _error(GalleryFailure("NOT_FOUND", f"{person_id} is not in the face gallery.", 404))
    return to_person(entry)


@router.delete(
    "/{person_id}",
    response_model=GalleryDeleteResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def delete_face(person_id: str, request: Request) -> GalleryDeleteResponse | JSONResponse:
    """Remove one person from the gallery."""
    blocked = _space_blocked() or _admin_blocked(request)
    if blocked is not None:
        return blocked
    try:
        removed = get_gallery().delete(person_id)
    except ValueError as exc:
        return _error(GalleryFailure("INVALID_PERSON_ID", str(exc), 400))
    if not removed:
        return _error(GalleryFailure("NOT_FOUND", f"{person_id} is not in the face gallery.", 404))
    return GalleryDeleteResponse(person_id=person_id, message=f"Removed {person_id} from the face gallery.")

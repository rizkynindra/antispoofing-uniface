"""Enroll a reference face and search a selfie against the gallery."""

from dataclasses import dataclass

import numpy as np
from fastapi import status

from app.gallery_schemas import (
    GalleryCandidate,
    GalleryEnrollRequest,
    GalleryEnrollResponse,
    GalleryPerson,
    GallerySearchRequest,
    GallerySearchResponse,
)
from app.services.face_engine import FaceEngine
from app.services.face_gallery import FaceGallery, GalleryEntry
from app.services.image_loader import load_image

CANDIDATE_LIMIT = 5
SIMILARITY_SCORE_MIN = 0.0
SIMILARITY_SCORE_MAX = 1.0


@dataclass(frozen=True)
class GalleryFailure:
    """Gallery error that maps to the shared JSON error body."""

    error_code: str
    message: str
    http_status: int


def enroll_person(
    payload: GalleryEnrollRequest,
    *,
    gallery: FaceGallery,
    engine: FaceEngine,
    image_loader=load_image,
) -> GalleryEnrollResponse | GalleryFailure:
    """Detect the largest face, store its embedding, and keep the photo."""
    image, error = image_loader(payload.image)
    if error or image is None:
        return GalleryFailure("INVALID_IMAGE", f"Failed to load the reference photo: {error}", status.HTTP_400_BAD_REQUEST)

    try:
        faces = engine.detect_faces(image)
        face = _largest_face(faces)
        if face is None:
            return GalleryFailure(
                "NO_FACE",
                "No face detected in the reference photo.",
                status.HTTP_400_BAD_REQUEST,
            )
        if payload.check_liveness:
            is_live, _score = engine.check_liveness(image, face.bbox)
            if not is_live:
                return GalleryFailure(
                    "SPOOF_DETECTED",
                    "The reference photo did not pass the spoof check.",
                    status.HTTP_400_BAD_REQUEST,
                )
        embedding = engine.extract_embedding(image, face.landmarks)
        label = payload.label.strip() or payload.person_id.strip()
        entry, replaced = gallery.enroll(
            person_id=payload.person_id.strip(),
            label=label,
            image=image,
            embedding=embedding,
            faces_detected=len(faces),
            detection_confidence=round(float(face.confidence), 4),
            replace=payload.replace,
        )
    except ValueError as exc:
        return GalleryFailure("INVALID_PERSON_ID", str(exc), status.HTTP_400_BAD_REQUEST)
    except FileExistsError:
        return GalleryFailure(
            "ALREADY_ENROLLED",
            f"{payload.person_id} is already in the face gallery.",
            status.HTTP_409_CONFLICT,
        )
    except Exception as exc:
        return GalleryFailure(
            "INTERNAL_SERVER_ERROR",
            f"Could not store the reference photo: {exc}",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    verb = "Replaced" if replaced else "Stored"
    return GalleryEnrollResponse(
        ChId=payload.ChId,
        person_id=entry.person_id,
        label=entry.label,
        faces_detected=entry.faces_detected,
        detection_confidence=entry.detection_confidence,
        replaced=replaced,
        message=f"{verb} {entry.person_id} in the face gallery.",
    )


def search_gallery(
    payload: GallerySearchRequest,
    *,
    gallery: FaceGallery,
    engine: FaceEngine,
    image_loader=load_image,
) -> GallerySearchResponse | GalleryFailure:
    """Score a selfie against every enrolled embedding and return the closest people."""
    image, error = image_loader(payload.image)
    if error or image is None:
        return GalleryFailure("INVALID_IMAGE", f"Failed to load the selfie: {error}", status.HTTP_400_BAD_REQUEST)

    try:
        faces = engine.detect_faces(image)
        face = _largest_face(faces)
        if face is None:
            return GalleryFailure("NO_FACE", "No face detected in the selfie.", status.HTTP_400_BAD_REQUEST)
        is_live: bool | None = None
        liveness_score: float | None = None
        if payload.options.check_liveness:
            is_live, score = engine.check_liveness(image, face.bbox)
            liveness_score = round(float(score), 4)
            if not is_live:
                return GalleryFailure(
                    "SPOOF_DETECTED",
                    "The selfie did not pass the spoof check.",
                    status.HTTP_400_BAD_REQUEST,
                )
        query = np.asarray(engine.extract_embedding(image, face.landmarks), dtype=np.float32).reshape(-1)
        entries = gallery.list_entries()
    except Exception as exc:
        return GalleryFailure(
            "INTERNAL_SERVER_ERROR",
            f"Could not search the face gallery: {exc}",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    candidates = _rank(entries, query, payload)
    if not candidates:
        return GallerySearchResponse(
            ChId=payload.ChId,
            matched=False,
            is_match="Different",
            faces_detected=len(faces),
            is_live=is_live,
            liveness_score=liveness_score,
            candidates=[],
            message="The face gallery is empty.",
        )

    best = candidates[0]
    return GallerySearchResponse(
        ChId=payload.ChId,
        matched=best.is_match == "TRUE",
        person_id=best.person_id,
        label=best.label,
        is_match=best.is_match,
        similarity_score=best.similarity_score,
        similarity_percentage=best.similarity_percentage,
        confidence_level=best.confidence_level,
        faces_detected=len(faces),
        is_live=is_live,
        liveness_score=liveness_score,
        candidates=candidates,
        message=_search_message(best),
    )


def to_person(entry: GalleryEntry) -> GalleryPerson:
    """Drop the embedding before a record leaves the service."""
    return GalleryPerson(
        person_id=entry.person_id,
        label=entry.label,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        faces_detected=entry.faces_detected,
        detection_confidence=entry.detection_confidence,
    )


def _largest_face(faces: list):
    if not faces:
        return None

    def area(face) -> float:
        x1, y1, x2, y2 = [float(value) for value in face.bbox]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return max(faces, key=area)


def _rank(entries: list[GalleryEntry], query: np.ndarray, payload: GallerySearchRequest) -> list[GalleryCandidate]:
    scored: list[GalleryCandidate] = []
    for entry in entries:
        stored = np.asarray(entry.embedding, dtype=np.float32).reshape(-1)
        if stored.shape != query.shape:
            continue
        similarity = float(np.dot(stored, query))
        similarity = max(SIMILARITY_SCORE_MIN, min(SIMILARITY_SCORE_MAX, similarity))
        label, confidence = FaceEngine.classify_similarity(
            similarity,
            match_threshold=payload.options.similarity_threshold,
            need_confirm_threshold=payload.options.need_confirmation_threshold,
        )
        scored.append(
            GalleryCandidate(
                person_id=entry.person_id,
                label=entry.label,
                similarity_score=round(similarity, 4),
                similarity_percentage=f"{similarity * 100:.2f}%",
                is_match=label,
                confidence_level=confidence,
            )
        )
    scored.sort(key=lambda candidate: candidate.similarity_score, reverse=True)
    return scored[:CANDIDATE_LIMIT]


def _search_message(best: GalleryCandidate) -> str:
    if best.is_match == "TRUE":
        return f"Matched {best.person_id} ({best.similarity_percentage})."
    if best.is_match == "Need Confirmation":
        return f"Closest face is {best.person_id} and needs confirmation ({best.similarity_percentage})."
    return f"No enrolled face is close enough. Closest is {best.person_id} ({best.similarity_percentage})."

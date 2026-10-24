"""Face comparison pipeline: load, detect, optional liveness, then match."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.compare_errors import (
    CompareFailure,
    internal_failure,
    invalid_image_failure,
    no_face_failure,
)
from app.schemas import (
    CompareResultData,
    FaceCompareRequest,
    SingleImageMetadata,
)
from app.services.face_engine import FaceEngine, face_engine
from app.services.image_loader import load_image

ImageLoader = Callable[..., tuple[np.ndarray | None, str | None]]


@dataclass
class CompareSuccess:
    """Successful 1:1 comparison outcome."""

    result: CompareResultData
    image1_metadata: SingleImageMetadata
    image2_metadata: SingleImageMetadata
    message: str


CompareOutcome = CompareSuccess | CompareFailure


def _metadata_from_faces(faces: list) -> SingleImageMetadata:
    return SingleImageMetadata(
        faces_detected=len(faces),
        bbox=faces[0].bbox.tolist() if faces else None,
    )


def _apply_optional_liveness(
    engine: FaceEngine,
    image: np.ndarray,
    faces: list,
    metadata: SingleImageMetadata,
    enabled: bool,
) -> None:
    if not enabled or not faces:
        return
    is_live, score = engine.check_liveness(image, faces[0].bbox)
    metadata.is_live = is_live
    metadata.liveness_score = round(score, 4)


def run_face_compare(
    payload: FaceCompareRequest,
    *,
    engine: FaceEngine | None = None,
    image_loader: ImageLoader = load_image,
) -> CompareOutcome:
    """Run the 1:1 comparison pipeline without HTTP or BPJS logging."""
    engine = engine or face_engine

    image1, error1 = image_loader(payload.image1)
    if error1 or image1 is None:
        return invalid_image_failure(1, error1)

    image2, error2 = image_loader(payload.image2)
    if error2 or image2 is None:
        return invalid_image_failure(2, error2)

    try:
        faces1 = engine.detect_faces(image1)
        faces2 = engine.detect_faces(image2)
        meta1 = _metadata_from_faces(faces1)
        meta2 = _metadata_from_faces(faces2)

        if not faces1:
            return no_face_failure(1, meta1, meta2)
        if not faces2:
            return no_face_failure(2, meta1, meta2)

        _apply_optional_liveness(
            engine, image1, faces1, meta1, payload.options.check_liveness_image1
        )
        _apply_optional_liveness(
            engine, image2, faces2, meta2, payload.options.check_liveness_image2
        )

        embedding1 = engine.extract_embedding(image1, faces1[0].landmarks)
        embedding2 = engine.extract_embedding(image2, faces2[0].landmarks)
        similarity = engine.compute_cosine_similarity(embedding1, embedding2)
        match_label, confidence = engine.classify_similarity(
            similarity,
            match_threshold=payload.options.similarity_threshold,
            need_confirm_threshold=payload.options.need_confirmation_threshold,
        )

        result = CompareResultData(
            is_match=match_label,
            similarity_score=round(similarity, 4),
            similarity_percentage=f"{similarity * 100:.2f}%",
            confidence_level=confidence,
        )
        return CompareSuccess(
            result=result,
            image1_metadata=meta1,
            image2_metadata=meta2,
            message=f"Comparison finished with verdict '{match_label}' (Score: {similarity:.4f})",
        )
    except Exception as exc:
        return internal_failure(str(exc))

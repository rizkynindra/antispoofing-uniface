"""Tests for the comparison pipeline with injected fakes (no ONNX)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from app.schemas import FaceCompareRequest
from app.services.compare_service import CompareFailure, CompareSuccess, run_face_compare


def _image_loader_ok(_item):
    return np.zeros((8, 8, 3), dtype=np.uint8), None


def _image_loader_fail(_item):
    return None, "decode failed"


def _face(*, landmarks=True):
    bbox = np.array([0.0, 0.0, 4.0, 4.0])
    marks = np.zeros((5, 2), dtype=np.float32) if landmarks else None
    return SimpleNamespace(bbox=bbox, landmarks=marks)


def test_run_face_compare_invalid_first_image() -> None:
    payload = FaceCompareRequest(image1="a", image2="b")
    outcome = run_face_compare(payload, engine=MagicMock(), image_loader=_image_loader_fail)
    assert isinstance(outcome, CompareFailure)
    assert outcome.error_code == "INVALID_IMAGE_1"


def test_run_face_compare_no_face_in_second_image() -> None:
    engine = MagicMock()
    engine.detect_faces.side_effect = [[_face()], []]
    payload = FaceCompareRequest(image1="a", image2="b")
    outcome = run_face_compare(payload, engine=engine, image_loader=_image_loader_ok)
    assert isinstance(outcome, CompareFailure)
    assert outcome.error_code == "NO_FACE_IN_IMAGE_2"
    assert outcome.image1_metadata is not None
    assert outcome.image1_metadata.faces_detected == 1
    assert outcome.image2_metadata is not None
    assert outcome.image2_metadata.faces_detected == 0


def test_run_face_compare_success_path() -> None:
    engine = MagicMock()
    engine.detect_faces.side_effect = [[_face()], [_face()]]
    engine.extract_embedding.return_value = np.ones(4, dtype=np.float32)
    engine.compute_cosine_similarity.return_value = 0.88
    engine.classify_similarity.return_value = ("TRUE", "HIGH")

    payload = FaceCompareRequest(image1="a", image2="b")
    outcome = run_face_compare(payload, engine=engine, image_loader=_image_loader_ok)

    assert isinstance(outcome, CompareSuccess)
    assert outcome.result.is_match == "TRUE"
    assert outcome.result.confidence_level == "HIGH"
    assert outcome.result.similarity_score == 0.88
    engine.check_liveness.assert_not_called()


def test_run_face_compare_optional_liveness() -> None:
    engine = MagicMock()
    engine.detect_faces.side_effect = [[_face()], [_face()]]
    engine.extract_embedding.return_value = np.ones(4, dtype=np.float32)
    engine.compute_cosine_similarity.return_value = 0.5
    engine.classify_similarity.return_value = ("Need Confirmation", "MEDIUM")
    engine.check_liveness.return_value = (True, 0.91)

    payload = FaceCompareRequest(
        image1="a",
        image2="b",
        options={"check_liveness_image2": True},
    )
    outcome = run_face_compare(payload, engine=engine, image_loader=_image_loader_ok)
    assert isinstance(outcome, CompareSuccess)
    assert outcome.image2_metadata.is_live is True
    assert outcome.image2_metadata.liveness_score == 0.91
    engine.check_liveness.assert_called_once()

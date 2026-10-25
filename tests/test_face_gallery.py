"""Face gallery enroll and search, without loading ONNX models."""

import numpy as np

from app.gallery_routes import _admin_blocked
from app.gallery_schemas import GalleryEnrollRequest, GallerySearchOptions, GallerySearchRequest
from app.services.face_gallery import FaceGallery
from app.services.gallery_service import enroll_person, search_gallery


class _Face:
    def __init__(self, bbox, mark: float) -> None:
        self.bbox = np.array(bbox, dtype=np.float32)
        self.landmarks = np.array([[mark, mark]], dtype=np.float32)
        self.confidence = 0.99


class _Engine:
    def __init__(self, embedding: np.ndarray, live: bool = True) -> None:
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.live = live
        self.seen_mark: float | None = None
        self.faces = [_Face([0, 0, 10, 20], 1.0)]

    def detect_faces(self, image):
        return self.faces

    def extract_embedding(self, image, landmarks):
        self.seen_mark = float(landmarks[0, 0])
        return self.embedding

    def check_liveness(self, image, bbox):
        return self.live, 0.2 if not self.live else 0.99


def _image(_item):
    return np.zeros((8, 8, 3), dtype=np.uint8), None


def _unit(index: int, size: int = 4) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_enroll_persists_and_search_matches(tmp_path) -> None:
    gallery = FaceGallery(tmp_path)
    engine = _Engine(_unit(0))
    enrolled = enroll_person(
        GalleryEnrollRequest(person_id="EMP001", label="Ayu", image="ignored"),
        gallery=gallery,
        engine=engine,
        image_loader=_image,
    )
    assert enrolled.person_id == "EMP001"
    assert enrolled.label == "Ayu"
    assert enrolled.replaced is False
    assert (tmp_path / "people" / "EMP001" / "photo.jpg").is_file()

    reloaded = FaceGallery(tmp_path)
    found = search_gallery(
        GallerySearchRequest(image="ignored"),
        gallery=reloaded,
        engine=_Engine(_unit(0)),
        image_loader=_image,
    )
    assert found.matched is True
    assert found.person_id == "EMP001"
    assert found.is_match == "TRUE"
    assert found.similarity_score == 1.0


def test_search_reports_a_different_person(tmp_path) -> None:
    gallery = FaceGallery(tmp_path)
    enroll_person(
        GalleryEnrollRequest(person_id="EMP001", image="ignored"),
        gallery=gallery,
        engine=_Engine(_unit(0)),
        image_loader=_image,
    )
    found = search_gallery(
        GallerySearchRequest(image="ignored"),
        gallery=gallery,
        engine=_Engine(_unit(1)),
        image_loader=_image,
    )
    assert found.matched is False
    assert found.person_id == "EMP001"
    assert found.is_match == "Different"
    assert found.similarity_score == 0.0


def test_enroll_uses_the_largest_face(tmp_path) -> None:
    engine = _Engine(_unit(0))
    engine.faces = [
        _Face([0, 0, 4, 4], 1.0),
        _Face([0, 0, 30, 40], 7.0),
    ]
    enrolled = enroll_person(
        GalleryEnrollRequest(person_id="EMP002", image="ignored"),
        gallery=FaceGallery(tmp_path),
        engine=engine,
        image_loader=_image,
    )
    assert enrolled.faces_detected == 2
    assert engine.seen_mark == 7.0


def test_duplicate_without_replace_is_rejected(tmp_path) -> None:
    gallery = FaceGallery(tmp_path)
    request = GalleryEnrollRequest(person_id="EMP001", image="ignored", replace=False)
    first = enroll_person(request, gallery=gallery, engine=_Engine(_unit(0)), image_loader=_image)
    second = enroll_person(request, gallery=gallery, engine=_Engine(_unit(0)), image_loader=_image)
    assert first.person_id == "EMP001"
    assert second.error_code == "ALREADY_ENROLLED"
    assert second.http_status == 409


def test_spoofed_selfie_is_rejected(tmp_path) -> None:
    gallery = FaceGallery(tmp_path)
    enroll_person(
        GalleryEnrollRequest(person_id="EMP001", image="ignored"),
        gallery=gallery,
        engine=_Engine(_unit(0)),
        image_loader=_image,
    )
    found = search_gallery(
        GallerySearchRequest(image="ignored", options=GallerySearchOptions(check_liveness=True)),
        gallery=gallery,
        engine=_Engine(_unit(0), live=False),
        image_loader=_image,
    )
    assert found.error_code == "SPOOF_DETECTED"


def test_person_id_cannot_escape_the_gallery_directory(tmp_path) -> None:
    outcome = enroll_person(
        GalleryEnrollRequest(person_id="../secrets", image="ignored"),
        gallery=FaceGallery(tmp_path),
        engine=_Engine(_unit(0)),
        image_loader=_image,
    )
    assert outcome.error_code == "INVALID_PERSON_ID"
    assert not (tmp_path / "secrets").exists()


def test_delete_removes_the_stored_photo(tmp_path) -> None:
    gallery = FaceGallery(tmp_path)
    enroll_person(
        GalleryEnrollRequest(person_id="EMP001", image="ignored"),
        gallery=gallery,
        engine=_Engine(_unit(0)),
        image_loader=_image,
    )
    assert gallery.delete("EMP001") is True
    assert gallery.get("EMP001") is None
    assert gallery.delete("EMP001") is False


def test_admin_token_blocks_a_missing_header(monkeypatch) -> None:
    monkeypatch.setattr("app.gallery_routes.settings.FACE_GALLERY_ADMIN_TOKEN", "secret-token")

    class _Request:
        headers = {}

    blocked = _admin_blocked(_Request())
    assert blocked is not None
    assert blocked.status_code == 401

    class _Allowed:
        headers = {"X-Admin-Token": "secret-token"}

    assert _admin_blocked(_Allowed()) is None

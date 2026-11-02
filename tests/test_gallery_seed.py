"""Folder seeding for the face gallery, without loading ONNX models."""

from pathlib import Path

import numpy as np

from app.gallery_schemas import GallerySeedRequest
from app.services.face_gallery import FaceGallery
from app.services.gallery_service import seed_gallery

from test_face_gallery import _Engine, _image


def _seed_dir(tmp_path: Path, names: list[str]) -> Path:
    folder = tmp_path / "seed"
    folder.mkdir()
    for name in names:
        (folder / name).write_bytes(b"not-a-real-image")
    return folder


def _seed(folder: Path, gallery: FaceGallery, **overrides):
    request = GallerySeedRequest(**overrides)
    return seed_gallery(
        request,
        gallery=gallery,
        engine=_Engine(_unit()),
        seed_dir=folder,
        image_loader=_image,
    )


def _unit(index: int = 0, size: int = 4) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_seed_enrolls_one_person_per_file(tmp_path) -> None:
    gallery = FaceGallery(tmp_path / "db")
    outcome = _seed(_seed_dir(tmp_path, ["177641794.jpg", "158811473.jpg"]), gallery)
    assert outcome.enrolled == 2
    assert outcome.replaced == 0
    assert outcome.skipped == 0
    assert [result.state for result in outcome.results] == ["ENROLLED", "ENROLLED"]
    assert outcome.results[0].person_id == "158811473"
    assert outcome.results[1].person_id == "177641794"
    assert (tmp_path / "db" / "people" / "177641794" / "photo.jpg").is_file()
    assert len(gallery.list_entries()) == 2


def test_seed_reports_files_without_a_face_as_skipped(tmp_path) -> None:
    gallery = FaceGallery(tmp_path / "db")
    engine = _Engine(_unit())
    engine.faces = []
    outcome = seed_gallery(
        GallerySeedRequest(),
        gallery=gallery,
        engine=engine,
        seed_dir=_seed_dir(tmp_path, ["177641794.jpg"]),
        image_loader=_image,
    )
    assert outcome.enrolled == 0
    assert outcome.skipped == 1
    assert outcome.results[0].state == "SKIPPED"
    assert outcome.results[0].faces_detected is None


def test_seed_second_run_replaces_when_allowed(tmp_path) -> None:
    folder = _seed_dir(tmp_path, ["177641794.jpg"])
    gallery = FaceGallery(tmp_path / "db")
    assert _seed(folder, gallery).enrolled == 1
    outcome = _seed(folder, gallery)
    assert outcome.enrolled == 0
    assert outcome.replaced == 1


def test_seed_reports_a_missing_folder(tmp_path) -> None:
    outcome = _seed(tmp_path / "nowhere", FaceGallery(tmp_path / "db"))
    assert outcome.error_code == "SEED_DIR_MISSING"
    assert outcome.http_status == 400


def test_seed_reports_an_empty_folder(tmp_path) -> None:
    outcome = _seed(_seed_dir(tmp_path, []), FaceGallery(tmp_path / "db"))
    assert outcome.error_code == "SEED_DIR_EMPTY"
    assert outcome.http_status == 400


def test_seed_ignores_non_image_files(tmp_path) -> None:
    folder = _seed_dir(tmp_path, ["177641794.jpg"])
    (folder / "notes.txt").write_text("ignore me")
    outcome = _seed(folder, FaceGallery(tmp_path / "db"))
    assert outcome.files_found == 1


def test_seed_then_search_matches_the_enrolled_file(tmp_path) -> None:
    from app.gallery_schemas import GallerySearchRequest
    from app.services.gallery_service import search_gallery

    folder = _seed_dir(tmp_path, ["177641794.jpg"])
    gallery = FaceGallery(tmp_path / "db")
    _seed(folder, gallery)

    found = search_gallery(
        GallerySearchRequest(image="ignored"),
        gallery=gallery,
        engine=_Engine(_unit(0)),
        image_loader=_image,
    )
    assert found.matched is True
    assert found.person_id == "177641794"
"""On-disk face gallery: one reference photo and embedding per person."""

from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import fcntl
import numpy as np

PERSON_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
EMBEDDING_FILENAME = "embedding.npy"
RECORD_FILENAME = "record.json"
PHOTO_FILENAME = "photo.jpg"
JPEG_QUALITY = 90
PEOPLE_DIRNAME = "people"


@dataclass(frozen=True)
class GalleryEntry:
    """One enrolled person, without the embedding vector."""

    person_id: str
    label: str
    created_at: str
    updated_at: str
    faces_detected: int
    detection_confidence: float
    embedding: np.ndarray


class FaceGallery:
    """Persist enrolled faces so a later selfie can be searched after a restart.

    The layout follows the UniFace face-database example: a normalized embedding
    per person, scored later by cosine similarity. Photos stay on disk so an
    admin can open the reference image. Writes take an exclusive file lock so
    two API workers cannot tear a record.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.people_dir = root / PEOPLE_DIRNAME
        self._lock_path = root / ".lock"

    def enroll(
        self,
        *,
        person_id: str,
        label: str,
        image: np.ndarray,
        embedding: np.ndarray,
        faces_detected: int,
        detection_confidence: float,
        replace: bool,
    ) -> tuple[GalleryEntry, bool]:
        """Store or replace one person. Returns the entry and whether it replaced an older photo."""
        self._require_person_id(person_id)
        vector = _as_embedding(embedding)
        now = _timestamp()
        with self._locked(exclusive=True):
            directory = self._person_dir(person_id)
            existed = (directory / RECORD_FILENAME).is_file()
            if existed and not replace:
                raise FileExistsError(person_id)
            directory.mkdir(parents=True, exist_ok=True)
            previous = _read_record(directory / RECORD_FILENAME) if existed else None
            created_at = previous["created_at"] if previous else now
            record = {
                "person_id": person_id,
                "label": label,
                "created_at": created_at,
                "updated_at": now,
                "faces_detected": faces_detected,
                "detection_confidence": detection_confidence,
            }
            _write_photo(directory / PHOTO_FILENAME, image)
            _write_embedding(directory / EMBEDDING_FILENAME, vector)
            _write_json(directory / RECORD_FILENAME, record)
        return _entry_from_record(record, vector), existed

    def delete(self, person_id: str) -> bool:
        """Remove one person. Returns False when that person was not enrolled."""
        self._require_person_id(person_id)
        with self._locked(exclusive=True):
            directory = self._person_dir(person_id)
            if not directory.is_dir():
                return False
            shutil.rmtree(directory)
            return True

    def get(self, person_id: str) -> GalleryEntry | None:
        """Return one enrolled person, or None when the id is missing."""
        self._require_person_id(person_id)
        with self._locked(exclusive=False):
            return self._load_entry(self._person_dir(person_id))

    def photo_path(self, person_id: str) -> Path | None:
        """Return the stored JPEG when the person exists."""
        self._require_person_id(person_id)
        path = self._person_dir(person_id) / PHOTO_FILENAME
        with self._locked(exclusive=False):
            if path.is_file() and (path.parent / RECORD_FILENAME).is_file():
                return path
        return None

    def list_entries(self) -> list[GalleryEntry]:
        """Return every complete enrollment, ordered by person id."""
        with self._locked(exclusive=False):
            if not self.people_dir.is_dir():
                return []
            entries = [
                entry
                for directory in self.people_dir.iterdir()
                if directory.is_dir() and (entry := self._load_entry(directory)) is not None
            ]
        return sorted(entries, key=lambda entry: entry.person_id)

    def _load_entry(self, directory: Path) -> GalleryEntry | None:
        record_path = directory / RECORD_FILENAME
        embedding_path = directory / EMBEDDING_FILENAME
        if not record_path.is_file() or not embedding_path.is_file():
            return None
        record = _read_record(record_path)
        if record is None:
            return None
        try:
            embedding = np.load(embedding_path)
        except (OSError, ValueError):
            return None
        try:
            vector = _as_embedding(embedding)
        except ValueError:
            return None
        return _entry_from_record(record, vector)

    def _person_dir(self, person_id: str) -> Path:
        return self.people_dir / person_id

    @staticmethod
    def _require_person_id(person_id: str) -> None:
        if not PERSON_ID_PATTERN.fullmatch(person_id):
            raise ValueError(
                "person_id must start with a letter or digit and use only letters, digits, dots, underscores, or hyphens."
            )

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.people_dir.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+") as handle:
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle.fileno(), mode)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("embedding must be a finite vector")
    return vector


def _entry_from_record(record: dict, embedding: np.ndarray) -> GalleryEntry:
    return GalleryEntry(
        person_id=str(record["person_id"]),
        label=str(record.get("label") or record["person_id"]),
        created_at=str(record.get("created_at") or ""),
        updated_at=str(record.get("updated_at") or ""),
        faces_detected=int(record.get("faces_detected") or 0),
        detection_confidence=float(record.get("detection_confidence") or 0.0),
        embedding=embedding,
    )


def _read_record(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "person_id" not in payload:
        return None
    return payload


def _write_json(path: Path, payload: dict) -> None:
    _atomic_bytes(path, json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _write_embedding(path: Path, embedding: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, embedding)
    os.replace(temporary, path)


def _write_photo(path: Path, image: np.ndarray) -> None:
    saved, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not saved:
        raise OSError("Could not encode the reference photo.")
    _atomic_bytes(path, encoded.tobytes())


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)

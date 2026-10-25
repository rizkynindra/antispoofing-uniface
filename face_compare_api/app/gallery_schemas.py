"""Request and response models for the face gallery."""

from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.schemas import ImageInputType


class GalleryEnrollRequest(BaseModel):
    """Store one reference face for later search."""

    person_id: str = Field(..., description="Stable id for this person, such as an employee number.")
    label: str = Field(default="", description="Name shown when this person is matched. Defaults to person_id.")
    image: ImageInputType = Field(..., description="Reference photo. URL string, base64 string, or image object.")
    replace: bool = Field(default=True, description="Replace the stored photo when this person_id already exists.")
    check_liveness: bool = Field(
        default=False,
        description="Reject the photo when MiniFASNet marks it as a print, screen, or mask.",
    )
    ChId: str = Field(default="UNKNOWN", description="Caller channel id. Echoed on the response.")


class GallerySearchOptions(BaseModel):
    """How a selfie is judged against the gallery."""

    check_liveness: bool = Field(
        default=False,
        description="Reject the selfie when MiniFASNet marks it as a print, screen, or mask.",
    )
    similarity_threshold: float = Field(
        default_factory=lambda: settings.SIMILARITY_MATCH_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Score at or above this value is the same person.",
    )
    need_confirmation_threshold: float = Field(
        default_factory=lambda: settings.NEED_CONFIRMATION_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Score at or above this value, and below the match line, needs a person to confirm.",
    )


class GallerySearchRequest(BaseModel):
    """Search the gallery with one selfie."""

    image: ImageInputType = Field(..., description="Selfie. URL string, base64 string, or image object.")
    options: GallerySearchOptions = Field(default_factory=GallerySearchOptions)
    ChId: str = Field(default="UNKNOWN", description="Caller channel id. Echoed on the response.")


class GalleryPerson(BaseModel):
    """Enrolled person, without the embedding."""

    person_id: str
    label: str
    created_at: str
    updated_at: str
    faces_detected: int
    detection_confidence: float


class GalleryEnrollResponse(BaseModel):
    """Result of storing a reference photo."""

    status: Literal["SUCCESS"] = "SUCCESS"
    ChId: str
    person_id: str
    label: str
    faces_detected: int
    detection_confidence: float
    replaced: bool
    message: str


class GalleryListResponse(BaseModel):
    """Every person currently stored."""

    status: Literal["SUCCESS"] = "SUCCESS"
    count: int
    people: list[GalleryPerson]


class GalleryCandidate(BaseModel):
    """One enrolled person scored against the selfie."""

    person_id: str
    label: str
    similarity_score: float
    similarity_percentage: str
    is_match: Literal["TRUE", "Need Confirmation", "Different"]
    confidence_level: Literal["HIGH", "MEDIUM", "LOW"]


class GallerySearchResponse(BaseModel):
    """Best gallery hit for a selfie."""

    status: Literal["SUCCESS"] = "SUCCESS"
    ChId: str
    matched: bool
    person_id: str | None = None
    label: str | None = None
    is_match: Literal["TRUE", "Need Confirmation", "Different"]
    similarity_score: float | None = None
    similarity_percentage: str | None = None
    confidence_level: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    faces_detected: int
    is_live: bool | None = None
    liveness_score: float | None = None
    candidates: list[GalleryCandidate]
    message: str


class GalleryDeleteResponse(BaseModel):
    """Result of removing one person."""

    status: Literal["SUCCESS"] = "SUCCESS"
    person_id: str
    message: str

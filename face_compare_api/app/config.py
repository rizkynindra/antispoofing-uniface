"""Configuration settings for Face Comparison API Service."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "logs"))
MILLISECONDS_PER_SECOND = 1000.0
DEFAULT_S3_BASE_URL = "http://s3.bpjsketenagakerjaan.go.id"


def env_flag(name: str) -> bool:
    """True when an environment variable is 1, true, or yes."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def resolve_s3_base_url(space: bool) -> str:
    """Return no S3 host on the public Space. The private service keeps its default."""
    if space:
        return ""
    return os.getenv("S3_BASE_URL", DEFAULT_S3_BASE_URL)


class Settings(BaseModel):
    """Runtime settings loaded from environment variables."""

    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "svc-face-compare")
    SERVICE_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")

    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "2112"))
    HF_SPACE: bool = env_flag("HF_SPACE")

    S3_BASE_URL: str = resolve_s3_base_url(env_flag("HF_SPACE"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", os.getenv("HTTP_TIMEOUT", "15")))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BACKOFF_BASE_SECONDS: float = float(os.getenv("RETRY_BACKOFF_BASE_SECONDS", "1.0"))
    REQUEST_DELAY_SECONDS: float = float(os.getenv("REQUEST_DELAY", "0.05"))

    SIMILARITY_MATCH_THRESHOLD: float = float(os.getenv("SIMILARITY_MATCH_THRESHOLD", "0.75"))
    NEED_CONFIRMATION_THRESHOLD: float = float(os.getenv("NEED_CONFIRMATION_THRESHOLD", "0.40"))
    LIVENESS_THRESHOLD: float = float(
        os.getenv("LIVENESS_THRESHOLD", os.getenv("SPOOFING_THRESHOLD", "0.50"))
    )

    FACE_GALLERY_DIR: str = os.getenv(
        "FACE_GALLERY_DIR",
        os.path.join(BASE_DIR, "data", "face_db"),
    )
    FACE_GALLERY_ADMIN_TOKEN: str = os.getenv("FACE_GALLERY_ADMIN_TOKEN", "")

    LOG_DIR: str = DEFAULT_LOG_DIR
    LOG_FILE_PATTERN: str = os.getenv("LOG_FILE_PATTERN", "system_log_{date}.log")

    HTTP_URL_SCHEMES: tuple[str, ...] = ("http://", "https://")
    URL_OR_PATH_SCHEMES: tuple[str, ...] = ("http://", "https://", "s3://")
    URL_HEURISTIC_MAX_LENGTH: int = 512
    IMAGE_FILE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    IMAGE_DOWNLOAD_HEADERS: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }


settings = Settings()

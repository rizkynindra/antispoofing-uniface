"""UniFace engine wrapping RetinaFace, AdaFace IR-101, and MiniFASNet."""

import logging

import numpy as np

from uniface.constants import AdaFaceWeights
from uniface.detection import RetinaFace
from uniface.face_utils import compute_similarity
from uniface.recognition import AdaFace
from uniface.spoofing import MiniFASNet

from app.config import settings
from app.core.similarity import ConfidenceLevel, MatchLabel, classify_similarity

logger = logging.getLogger(__name__)

CUDA_PROVIDER_TOKENS = ("CUDA", "Tensorrt")
SIMILARITY_SCORE_MIN = 0.0
SIMILARITY_SCORE_MAX = 1.0
LIVENESS_FAIL_CLOSED_SCORE = 0.0


class FaceEngine:
    """Process-wide Face Detection, Recognition, and Anti-Spoofing engine.

    Models are loaded on first use so importing this module in tests does not
    download or initialize ONNX sessions.
    """

    _instance: "FaceEngine | None" = None

    def __new__(cls) -> "FaceEngine":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._models_ready = False
            instance._device = "CPU"
            cls._instance = instance
        return cls._instance

    def _ensure_models(self) -> None:
        if not self._models_ready:
            self._init_models()
            self._models_ready = True

    def _init_models(self) -> None:
        logger.info("Initializing RetinaFace detector")
        self.detector = RetinaFace()

        logger.info("Initializing AdaFace IR-101 recognizer")
        self.recognizer = AdaFace(model_name=AdaFaceWeights.IR_101)

        logger.info(
            "Initializing MiniFASNet anti-spoofing (threshold=%s)",
            settings.LIVENESS_THRESHOLD,
        )
        self.spoof_detector = MiniFASNet(threshold=settings.LIVENESS_THRESHOLD)

        detector_provider = self.detector.session.get_providers()[0]
        recognizer_provider = self.recognizer.session.get_providers()[0]
        spoof_provider = self.spoof_detector.session.get_providers()[0]
        providers = (detector_provider, recognizer_provider, spoof_provider)
        uses_gpu = any(
            any(token in provider for token in CUDA_PROVIDER_TOKENS) for provider in providers
        )
        self._device = "GPU (CUDA)" if uses_gpu else "CPU"
        logger.info("Models initialized. Active device: %s providers=%s", self._device, providers)

    @property
    def device(self) -> str:
        self._ensure_models()
        return self._device

    def detect_faces(self, image: np.ndarray) -> list:
        """Detect all faces in the given BGR image."""
        self._ensure_models()
        if image is None or image.size == 0:
            return []
        return self.detector.detect(image)

    def extract_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Extract a 512-D normalized facial embedding."""
        self._ensure_models()
        return self.recognizer.get_normalized_embedding(image, landmarks)

    def check_liveness(self, image: np.ndarray, bbox: list | np.ndarray) -> tuple[bool, float]:
        """Run MiniFASNet on a face box.

        Fail closed: crop or inference errors are treated as not live so a
        spoof cannot pass by triggering an exception.
        """
        self._ensure_models()
        try:
            result = self.spoof_detector.predict(image, bbox)
            return bool(result.is_real), float(result.confidence)
        except Exception:
            logger.exception("Liveness check failed; treating face as not live")
            return False, LIVENESS_FAIL_CLOSED_SCORE

    @staticmethod
    def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two normalized embeddings, clamped to [0, 1]."""
        similarity = float(compute_similarity(embedding1, embedding2, normalized=True))
        return max(SIMILARITY_SCORE_MIN, min(SIMILARITY_SCORE_MAX, similarity))

    @staticmethod
    def classify_similarity(
        similarity: float,
        match_threshold: float = settings.SIMILARITY_MATCH_THRESHOLD,
        need_confirm_threshold: float = settings.NEED_CONFIRMATION_THRESHOLD,
    ) -> tuple[MatchLabel, ConfidenceLevel]:
        """Classify similarity score into decision tiers."""
        classified = classify_similarity(
            similarity,
            match_threshold=match_threshold,
            need_confirm_threshold=need_confirm_threshold,
        )
        if classified is None:
            raise ValueError("similarity must be a float")
        return classified


face_engine = FaceEngine()

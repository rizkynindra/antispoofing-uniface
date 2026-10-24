"""Tests for FaceEngine liveness fail-closed behavior."""

import numpy as np

from app.services.face_engine import FaceEngine, LIVENESS_FAIL_CLOSED_SCORE


def test_check_liveness_fails_closed_on_exception() -> None:
    engine = FaceEngine()
    previous_ready = engine._models_ready
    previous_spoofer = getattr(engine, "spoof_detector", None)

    class _FailingSpoofer:
        def predict(self, image, bbox):
            raise RuntimeError("crop out of bounds")

    engine._models_ready = True
    engine.spoof_detector = _FailingSpoofer()
    try:
        is_live, score = engine.check_liveness(
            np.zeros((8, 8, 3), dtype=np.uint8),
            [0, 0, 4, 4],
        )
        assert is_live is False
        assert score == LIVENESS_FAIL_CLOSED_SCORE
    finally:
        engine._models_ready = previous_ready
        if previous_spoofer is not None:
            engine.spoof_detector = previous_spoofer
        elif hasattr(engine, "spoof_detector"):
            delattr(engine, "spoof_detector")

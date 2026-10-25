"""Tests for auto-detecting URL vs base64 image payloads."""

from app.services.image_loader import decode_base64_image, load_image


def test_decode_base64_invalid_payload() -> None:
    image, error = decode_base64_image("%%%not-base64%%%")
    assert image is None
    assert error is not None
    assert "Invalid base64" in error or "Failed" in error


def test_load_image_empty_string() -> None:
    image, error = load_image("   ")
    assert image is None
    assert error == "Image data is empty."


def test_space_refuses_a_web_address(monkeypatch) -> None:
    monkeypatch.setattr("app.services.image_loader.settings.HF_SPACE", True)
    image, error = load_image("https://example.test/face.jpg")
    assert image is None
    assert error == "Upload a photo file. This demo does not fetch a web address."

"""The attendance page opens the camera and checks the capture against the gallery."""

import asyncio

from fastapi.responses import FileResponse

from app.main import PRESENSI_PAGE, presensi_camera_page


def test_presensi_page_is_served() -> None:
    response = asyncio.run(presensi_camera_page())

    assert isinstance(response, FileResponse)
    assert response.path == PRESENSI_PAGE
    html = PRESENSI_PAGE.read_text(encoding="utf-8")
    assert "getUserMedia" in html
    assert 'facingMode: "user"' in html
    assert "/v1/gallery/search" in html
    assert "/v1/compare" in html
    assert "startCamera();" in html
    assert "gallery/seed" not in html
    assert "admin-token" not in html
    assert "<!DOCTYPE html>" in html
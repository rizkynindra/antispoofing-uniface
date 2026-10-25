"""The operator page is served with the comparison form."""

import asyncio

from fastapi.responses import FileResponse

from app.main import UI_INDEX, face_compare_page


def test_face_compare_page_serves_the_form() -> None:
    response = asyncio.run(face_compare_page())

    assert isinstance(response, FileResponse)
    assert response.path == UI_INDEX
    html = UI_INDEX.read_text(encoding="utf-8")
    assert "Compare these photos" in html
    assert "/v1/compare" in html
    assert "No comparison yet" in html
    assert "Comparing the two photos" in html
    assert "Draft without direction." in html
    assert "\u2014" not in html

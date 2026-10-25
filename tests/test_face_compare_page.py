"""The operator page is served with the comparison form."""

import asyncio

from fastapi.responses import FileResponse, HTMLResponse

from app.config import resolve_s3_base_url, settings
from app.main import UI_INDEX, api_metadata, face_compare_page, page_for_space


def test_face_compare_page_serves_the_form(monkeypatch) -> None:
    monkeypatch.setattr(settings, "HF_SPACE", False)
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


def test_space_page_only_offers_file_upload(monkeypatch) -> None:
    monkeypatch.setattr(settings, "HF_SPACE", True)

    response = asyncio.run(face_compare_page())

    assert isinstance(response, HTMLResponse)
    html = response.body.decode("utf-8")
    assert 'data-upload-only="true"' in html
    assert "image1-file" in html
    assert "image2-file" in html
    assert "Use an address" not in html
    assert "S3" not in html
    assert 'id="channel-id"' not in html
    assert "bpjs" not in html.lower()
    assert "ketenagakerjaan" not in html.lower()


def test_space_metadata_has_no_private_name() -> None:
    title, description = api_metadata(True)
    assert title == "Face comparison"
    assert "bpjs" not in title.lower()
    assert "bpjs" not in description.lower()
    assert resolve_s3_base_url(True) == ""
    private_html = UI_INDEX.read_text(encoding="utf-8")
    assert "Use an address" not in page_for_space(private_html)

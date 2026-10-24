"""Batch anti-spoofing detection from Excel image URLs using UniFace."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

_API_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_compare_api")
if _API_ROOT not in sys.path:
    sys.path.insert(0, _API_ROOT)

from app.config import settings
from app.core.image_bytes import create_image_session, download_and_decode_image
from app.core.url_utils import build_full_url

from uniface.detection import RetinaFace
from uniface.spoofing import MiniFASNet


def process_excel_urls(
    input_file: str,
    output_file: str,
    url_column: str = "link_download_foto_diri",
    threshold: float = settings.LIVENESS_THRESHOLD,
    timeout: int = settings.REQUEST_TIMEOUT,
    retries: int = settings.MAX_RETRIES,
) -> None:
    """Read an Excel file with image URLs, run anti-spoofing, and save results."""
    print(f"Loading input file: {input_file}")
    df = pd.read_excel(input_file)

    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in {input_file}. Available columns: {list(df.columns)}")

    print(f"Initializing UniFace models (RetinaFace & MiniFASNet with threshold={threshold})...")
    detector = RetinaFace()
    spoofer = MiniFASNet(threshold=threshold)

    detector_provider = detector.session.get_providers()[0]
    spoofer_provider = spoofer.session.get_providers()[0]
    print(f"RetinaFace Provider: {detector_provider}")
    print(f"MiniFASNet Provider: {spoofer_provider}")
    if "CUDA" in detector_provider or "Tensorrt" in detector_provider:
        print("Execution Device: GPU (CUDA)")
    else:
        print("Execution Device: CPU (Fallback)")

    is_real_list: list[bool | None] = []
    confidence_list: list[float | None] = []
    faces_count_list: list[int] = []
    status_list: list[str] = []

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    session = create_image_session()
    print(f"Processing {len(df)} URLs...")
    for url in tqdm(df[url_column], desc="Anti-spoofing"):
        url_str = build_full_url(url, settings.S3_BASE_URL)
        if not url_str:
            is_real_list.append(None)
            confidence_list.append(None)
            faces_count_list.append(0)
            status_list.append("Empty URL")
            continue

        image, download_error = download_and_decode_image(
            url_str, timeout=timeout, max_retries=retries, session=session
        )
        if download_error or image is None:
            is_real_list.append(None)
            confidence_list.append(None)
            faces_count_list.append(0)
            status_list.append(download_error or "Failed to decode image")
            continue

        try:
            faces = detector.detect(image)
            face_count = len(faces)
            if not faces:
                is_real_list.append(None)
                confidence_list.append(None)
                status_list.append("No face detected")
            else:
                result = spoofer.predict(image, faces[0].bbox)
                is_real_list.append(result.is_real)
                confidence_list.append(round(result.confidence, 4))
                status_list.append("Success")
        except Exception as exc:
            face_count = 0
            is_real_list.append(None)
            confidence_list.append(None)
            status_list.append(f"Error: {type(exc).__name__}")
        faces_count_list.append(face_count)

    df["faces_count"] = faces_count_list
    df["is_real"] = is_real_list
    df["confidence"] = confidence_list
    df["status"] = status_list

    df.to_excel(output_file, index=False)
    print(f"Successfully saved results to {output_file}")


def main() -> None:
    env_input = os.getenv("INPUT_FILE", "input_urls.xlsx")
    env_output = os.getenv("OUTPUT_FILE", "anti_spoofing_results.xlsx")
    env_column = os.getenv("URL_COLUMN", "link_download_foto_diri")
    env_threshold = float(os.getenv("SPOOFING_THRESHOLD", str(settings.LIVENESS_THRESHOLD)))
    env_timeout = int(os.getenv("HTTP_TIMEOUT", str(settings.REQUEST_TIMEOUT)))
    env_retries = int(os.getenv("MAX_RETRIES", str(settings.MAX_RETRIES)))

    parser = argparse.ArgumentParser(description="Run anti-spoofing on image URLs in an Excel file.")
    parser.add_argument("--input", "-i", default=env_input, help=f"Path to input Excel file (default from .env: {env_input})")
    parser.add_argument("--output", "-o", default=env_output, help=f"Path to output Excel file (default from .env: {env_output})")
    parser.add_argument("--column", "-c", default=env_column, help=f"Column name containing image URLs (default from .env: {env_column})")
    parser.add_argument(
        "--threshold",
        "-th",
        type=float,
        default=env_threshold,
        help=f"Anti-spoofing confidence threshold for real face classification (default from .env: {env_threshold})",
    )
    parser.add_argument("--timeout", "-t", type=int, default=env_timeout, help=f"HTTP request timeout in seconds (default from .env: {env_timeout})")
    parser.add_argument("--retries", "-r", type=int, default=env_retries, help=f"Max retry attempts on network error (default: {env_retries})")

    args = parser.parse_args()

    try:
        process_excel_urls(
            input_file=args.input,
            output_file=args.output,
            url_column=args.column,
            threshold=args.threshold,
            timeout=args.timeout,
            retries=args.retries,
        )
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

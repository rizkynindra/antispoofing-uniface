"""Batch face comparison between two image columns in an Excel file using UniFace."""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

_API_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_compare_api")
if _API_ROOT not in sys.path:
    sys.path.insert(0, _API_ROOT)

from app.config import settings
from app.core.image_bytes import create_image_session, download_and_decode_image
from app.core.similarity import classify_similarity_label
from app.core.url_utils import build_full_url

from uniface.detection import RetinaFace
from uniface.face_utils import compute_similarity
from uniface.recognition import ArcFace


def compare_excel_faces(
    input_file: str,
    output_file: str,
    column1: str = "FOTO_PEMULIHAN",
    column2: str = "FOTO_PENGKINIAN",
    base_url: str = settings.S3_BASE_URL,
    limit: int | None = None,
    delay: float = settings.REQUEST_DELAY_SECONDS,
    retries: int = settings.MAX_RETRIES,
    timeout: int = settings.REQUEST_TIMEOUT,
) -> None:
    """Read an Excel file with two image URL columns, compare faces, and save results."""
    print(f"Loading input file: {input_file}")
    df = pd.read_excel(input_file, nrows=limit if limit and limit > 0 else None)
    if limit and limit > 0:
        print(f"Limiting execution to first {len(df)} rows")

    for col in (column1, column2):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {input_file}. Available columns: {list(df.columns)}")

    print("Initializing UniFace models (RetinaFace & ArcFace)...")
    detector = RetinaFace()
    recognizer = ArcFace()

    detector_provider = detector.session.get_providers()[0]
    recognizer_provider = recognizer.session.get_providers()[0]
    print(f"RetinaFace Provider: {detector_provider}")
    print(f"ArcFace Provider: {recognizer_provider}")
    if "CUDA" in detector_provider or "Tensorrt" in detector_provider:
        print("Execution Device: GPU (CUDA)")
    else:
        print("Execution Device: CPU (Fallback)")

    print(f"S3 Base URL: {base_url}")

    faces_count_col1: list[int] = []
    faces_count_col2: list[int] = []
    similarity_list: list[float | None] = []
    is_match_list: list[str | None] = []
    status_list: list[str] = []

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    session = create_image_session()

    print(f"Comparing faces across {len(df)} rows...")
    for idx in tqdm(range(len(df)), desc="Face Comparison"):
        url1 = build_full_url(df.at[idx, column1], base_url)
        url2 = build_full_url(df.at[idx, column2], base_url)

        if not url1 and not url2:
            faces_count_col1.append(0)
            faces_count_col2.append(0)
            similarity_list.append(None)
            is_match_list.append(None)
            status_list.append("Both URLs empty")
            continue
        if not url1:
            faces_count_col1.append(0)
            faces_count_col2.append(0)
            similarity_list.append(None)
            is_match_list.append(None)
            status_list.append(f"Empty URL in {column1}")
            continue
        if not url2:
            faces_count_col1.append(0)
            faces_count_col2.append(0)
            similarity_list.append(None)
            is_match_list.append(None)
            status_list.append(f"Empty URL in {column2}")
            continue

        img1, err1 = download_and_decode_image(
            url1, timeout=timeout, max_retries=retries, session=session
        )
        img2, err2 = download_and_decode_image(
            url2, timeout=timeout, max_retries=retries, session=session
        )

        if delay > 0:
            time.sleep(delay)

        if err1 or img1 is None:
            faces_count_col1.append(0)
            faces_count_col2.append(0 if img2 is None else len(detector.detect(img2)))
            similarity_list.append(None)
            is_match_list.append(None)
            status_list.append(f"{column1}: {err1}")
            continue

        if err2 or img2 is None:
            faces_count_col1.append(len(detector.detect(img1)))
            faces_count_col2.append(0)
            similarity_list.append(None)
            is_match_list.append(None)
            status_list.append(f"{column2}: {err2}")
            continue

        try:
            faces1 = detector.detect(img1)
            faces2 = detector.detect(img2)

            cnt1 = len(faces1)
            cnt2 = len(faces2)
            faces_count_col1.append(cnt1)
            faces_count_col2.append(cnt2)

            if cnt1 == 0 and cnt2 == 0:
                similarity_list.append(None)
                is_match_list.append(None)
                status_list.append("No face detected in both images")
            elif cnt1 == 0:
                similarity_list.append(None)
                is_match_list.append(None)
                status_list.append(f"No face detected in {column1}")
            elif cnt2 == 0:
                similarity_list.append(None)
                is_match_list.append(None)
                status_list.append(f"No face detected in {column2}")
            else:
                embedding1 = recognizer.get_normalized_embedding(img1, faces1[0].landmarks)
                embedding2 = recognizer.get_normalized_embedding(img2, faces2[0].landmarks)
                similarity = float(compute_similarity(embedding1, embedding2, normalized=True))
                similarity_list.append(round(similarity, 4))
                is_match_list.append(classify_similarity_label(similarity))
                status_list.append("Success")

        except Exception as exc:
            if len(faces_count_col1) <= idx:
                faces_count_col1.append(0)
            if len(faces_count_col2) <= idx:
                faces_count_col2.append(0)
            similarity_list.append(None)
            is_match_list.append(None)
            status_list.append(f"Inference Error: {type(exc).__name__}")

    df[f"faces_count_{column1.lower()}"] = faces_count_col1
    df[f"faces_count_{column2.lower()}"] = faces_count_col2
    df["similarity_score"] = similarity_list
    df["is_match"] = is_match_list
    df["comparison_status"] = status_list

    df.to_excel(output_file, index=False)
    print(f"Successfully saved results to {output_file}")


def main() -> None:
    env_input = os.getenv("COMPARE_INPUT_FILE", "excel/Pemulihan_Akun_Pengkinian_Data.xlsx")
    env_output = os.getenv("COMPARE_OUTPUT_FILE", "result/result_face_comparison.xlsx")
    env_col1 = os.getenv("COLUMN_PEMULIHAN", "FOTO_PEMULIHAN")
    env_col2 = os.getenv("COLUMN_PENGKINIAN", "FOTO_PENGKINIAN")
    env_base_url = os.getenv("S3_BASE_URL", settings.S3_BASE_URL)
    env_delay = float(os.getenv("REQUEST_DELAY", str(settings.REQUEST_DELAY_SECONDS)))
    env_retries = int(os.getenv("MAX_RETRIES", str(settings.MAX_RETRIES)))
    env_timeout = int(os.getenv("HTTP_TIMEOUT", str(settings.REQUEST_TIMEOUT)))

    parser = argparse.ArgumentParser(description="Compare faces between two image columns in an Excel file.")
    parser.add_argument("--input", "-i", default=env_input, help=f"Path to input Excel file (default: {env_input})")
    parser.add_argument("--output", "-o", default=env_output, help=f"Path to output Excel file (default: {env_output})")
    parser.add_argument("--col1", "-c1", default=env_col1, help=f"First image column name (default: {env_col1})")
    parser.add_argument("--col2", "-c2", default=env_col2, help=f"Second image column name (default: {env_col2})")
    parser.add_argument("--base-url", "-u", default=env_base_url, help=f"S3 Base URL (default: {env_base_url})")
    parser.add_argument("--limit", "-n", "-l", type=int, default=None, help="Limit processing to the first N rows (optional)")
    parser.add_argument("--delay", "-d", type=float, default=env_delay, help=f"Rest delay in seconds between rows (default: {env_delay})")
    parser.add_argument("--retries", "-r", type=int, default=env_retries, help=f"Max retry attempts on network/timeout error (default: {env_retries})")
    parser.add_argument("--timeout", "-t", type=int, default=env_timeout, help=f"HTTP request timeout in seconds (default: {env_timeout})")

    args = parser.parse_args()

    try:
        compare_excel_faces(
            input_file=args.input,
            output_file=args.output,
            column1=args.col1,
            column2=args.col2,
            base_url=args.base_url,
            limit=args.limit,
            delay=args.delay,
            retries=args.retries,
            timeout=args.timeout,
        )
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

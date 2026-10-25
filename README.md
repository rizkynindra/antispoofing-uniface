# UniFace: Face Comparison, Anti-Spoofing & eKYC Service

Unified Face Verification, Anti-Spoofing (Liveness Detection), and Batch Face Comparison system built with **UniFace** (RetinaFace, AdaFace IR-101, MiniFASNet) and **FastAPI**

---

## 📋 Features

* **1:1 Face Comparison / Verification**: Cosine similarity matching using normalized 512-D embeddings (AdaFace IR-101).
* **Face gallery search**: An admin stores one reference photo per person. A later selfie is scored against that gallery.
* **Anti-Spoofing / Presentation Attack Detection (PAD)**: Passive RGB liveness detection using MiniFASNet.
* **Face Quality & Pre-checks**: 5-point facial landmark alignment & bounding box detection (RetinaFace).
* **Flexible Input Formats**: Accepts direct HTTP/S3 URLs, relative S3 paths, or Base64 encoded strings.
* **REST API Service**: High-performance FastAPI server running on default port `2112` with interactive Swagger UI (`/docs`).
* **Multi-GPU Ready**: Optimized for multi-GPU setups (e.g., $2 \times 24\text{ GB}$ NVIDIA L4).
* **Batch Processing**: CLI tool to compare image URL pairs across Excel datasets.

---

## 🛠️ Installation

### 1. Prerequisites
* **Python**: `3.10` recommended (`>=3.10,<3.15`)
* **NVIDIA driver**: a driver that supports CUDA 12. Check with `nvidia-smi`. The CUDA toolkit does not need a separate install. `pip install -e ".[gpu]"` installs ONNX Runtime GPU plus the CUDA 12 and cuDNN 9 libraries, and the app loads those libraries before the face models start. Each process uses the first GPU the driver exposes. Set `CUDA_VISIBLE_DEVICES` only when you want to hide some of them.
* **CPU-only machine**: skip the driver and use the CPU install in step 3.

### 2. Setup Virtual Environment

#### Option A: Using `uv` (Recommended / Fastest)
```bash
# Create virtual environment with Python 3.10
uv venv --python 3.10

# Activate virtual environment
source .venv/bin/activate
```

#### Option B: Using standard Python `venv`
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

On a GPU machine, install the local project with the GPU extra first. That extra is what makes the next machine use the GPU. `requirements.txt` only adds the API and the batch tools. It does not install UniFace from PyPI.

```bash
# GPU machine: ONNX Runtime CUDA, cuDNN, and the CUDA runtime libraries
pip install -e ".[gpu]"

# API, Gunicorn, and batch dependencies
pip install -r requirements.txt
```

CPU-only machine:

```bash
pip install -e ".[cpu]"
pip install -r requirements.txt
```

Check that the GPU install can see CUDA. `CUDAExecutionProvider` should be in the list:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Leave `CUDA_VISIBLE_DEVICES` unset in `.env`. Each worker then uses the first GPU the driver exposes on that machine.

---

## 🚀 How to Use

### Method 1: Running the FastAPI Service (Development)

Start the API service locally using the CLI runner:

```bash
# Run server on port 2112 with auto-reload
python face_compare_api/run.py --port 2112 --reload
```

* **Face compare page**: [http://localhost:2112/](http://localhost:2112/)
* **Interactive Swagger UI**: [http://localhost:2112/docs](http://localhost:2112/docs)
* **ReDoc Documentation**: [http://localhost:2112/redoc](http://localhost:2112/redoc)
* **Health Check**: `GET http://localhost:2112/health`

On a GPU machine the health JSON should report `"device": "GPU (CUDA)"`. `CPU` means the driver did not expose a GPU, or the GPU extra was not installed.

---

### Method 2: Running as a Production Systemd Service (2x 24GB NVIDIA L4)

The unit file is [face-compare.service](face-compare.service). Gunicorn settings are in [face_compare_api/gunicorn_conf.py](face_compare_api/gunicorn_conf.py).

#### 1. Workers and VRAM
* **Workers**: `4 - 6` on a 24 GB GPU. This laptop uses `2` because the card has 6 GB.
* **VRAM**: each worker takes about 1.5 GB. Workers share the first GPU. The unit does not place workers on a second card.
* **Worker lifecycle**: `max_requests = 5000` with `max_requests_jitter = 500`, so long-running workers restart before memory fragments.

Before enabling the unit, replace `/home/sysadmin/antispoofing-uniface` in [face-compare.service](face-compare.service) with the checkout path on the new machine, and install with `pip install -e ".[gpu]"`. Leave `CUDA_VISIBLE_DEVICES` unset. Pinning `0,1` hides nothing on a two-GPU host and can fail CUDA on a machine that only has GPU 0.

#### 2. Install and Enable the Service

```bash
# 1. Copy service file to systemd directory
sudo cp face-compare.service /etc/systemd/system/

# 2. Reload systemd daemon
sudo systemctl daemon-reload

# 3. Enable service on system boot
sudo systemctl enable face-compare

# 4. Start the service
sudo systemctl start face-compare

# 5. Check status
sudo systemctl status face-compare
```

#### 3. Monitoring & Logs

```bash
# View real-time service logs
journalctl -u face-compare -f

# View application standard log file
tail -f face_compare_api/logs/system_log_$(date +%d%m%Y).log
```

---

### Method 3: Batch Face Comparison (`face-compare.py`)

To run bulk face comparisons across rows of an Excel file containing image URLs:

```bash
python face-compare.py \
  --input excel/Pemulihan_Akun_Pengkinian_Data.xlsx \
  --output result/result_face_comparison_20082026.xlsx \
  --col1 FOTO_PEMULIHAN \
  --col2 FOTO_PENGKINIAN \
  --base-url 
```

---

## SDK tutorial

The SDK calls a face comparison API that is already running. It does not install the models or the GPU libraries. Use the Python package from a backend or a script. Use the Flutter package from an Android or iOS app. Both packages send `image1` as the reference photo and `image2` as the photo being checked.

Matching always runs. The result is `TRUE`, `Need Confirmation`, or `Different`. The decision threshold matrix later in this file lists the default cutoffs, 0.75 and 0.40.

The spoof check is optional. It is the passive MiniFASNet test: a real face versus a printed photo, a screen replay, or a mask. It does not ask the person to blink or turn their head. When the check is on, the image metadata includes `is_live` and `liveness_score`. When it is off, those fields stay empty.

### 1. Start the API

From this repository, with the virtual environment active:

```bash
python face_compare_api/run.py --port 2112
```

Check it before calling from another program:

```bash
curl http://127.0.0.1:2112/health
```

`device` is `GPU (CUDA)` when that process is using the GPU, and `CPU` when it is not. The page at [http://127.0.0.1:2112/](http://127.0.0.1:2112/) shows the same line.

### 2. Python

Install the client into the environment that will call the API. This can be a different machine from the one running the models.

```bash
pip install -e ./sdk
```

Confirm the service, then compare two photos:

```python
from face_compare_sdk import FaceCompareClient, FaceCompareError

with FaceCompareClient("http://127.0.0.1:2112", channel_id="PORTAL_KLAIM") as client:
    print(client.health().device)
    try:
        verdict = client.compare(
            "http://your_url/kyc/ref.jpg",
            "live.jpg",
            check_liveness_image2=True,
        )
    except FaceCompareError as exc:
        print(exc.error_code, exc)
    else:
        print(verdict.is_match, verdict.similarity_score, verdict.similarity_percentage)
        print(verdict.image2.is_live, verdict.image2.liveness_score)
```

`check_liveness_image2=True` runs the spoof check on the photo being checked. `check_liveness_image1=True` runs it on the reference photo. Leave both off for a match-only call.

Each argument accepts:

| You pass | What the SDK sends |
|---|---|
| `http://`, `https://`, `s3://`, or a `data:` URL | the string, and the API downloads or decodes it |
| an S3 key such as `kyc/ref.jpg` | the string, and the API joins it with `S3_BASE_URL` |
| a path to a local file, or raw image `bytes` | base64 |

Optional arguments on `compare`:

| Argument | Meaning |
|---|---|
| `channel_id` | Overrides the client `channel_id` for this call. Sent as `ChId`. |
| `similarity_threshold` | Score at or above this value is `TRUE`. Default on the server is `0.75`. |
| `need_confirmation_threshold` | Score at or above this value, and below the match threshold, is `Need Confirmation`. Default on the server is `0.40`. |
| `trace_id` | Sent as `X-TRACE-ID` so this call can be found in the API log. |
| `prev_transaction_id` | Sent as `X-PREV-TRANS-ID`. |

On success, `verdict.trace_id` and `verdict.transaction_id` come back from the response headers. A failed comparison raises `FaceCompareError`. `error_code` is the API code, or `TRANSPORT` when the client cannot reach the host.

| `error_code` | When |
|---|---|
| `INVALID_IMAGE_1`, `INVALID_IMAGE_2` | That image could not be loaded or decoded. |
| `NO_FACE_IN_IMAGE_1`, `NO_FACE_IN_IMAGE_2` | No face was found. `image1` and `image2` on the exception still hold the metadata. |
| `INTERNAL_SERVER_ERROR` | The API failed while comparing. |
| `EMPTY_IMAGE`, `IMAGE_NOT_FOUND` | The SDK rejected the input before the request. |
| `TRANSPORT` | Connection failed or timed out. The default timeout is 60 seconds. |

### 3. Flutter (Android and iOS)

One Dart package covers both platforms. It has no Android or iOS plugin code, and it does not open the camera. Take the JPEG bytes or the file path from your camera or gallery plugin and pass them in.

In the Flutter app `pubspec.yaml`, point `path` at this checkout:

```yaml
dependencies:
  face_compare_sdk:
    path: ../antispoofing-uniface/sdk/flutter/face_compare_sdk
```

Then run `flutter pub get`.

`baseUrl` depends on where the app is running. The API must listen on `0.0.0.0` (that is the default `HOST`).

| App is running on | `baseUrl` |
|---|---|
| Android emulator | `http://10.0.2.2:2112` |
| iOS simulator | `http://127.0.0.1:2112` |
| A physical phone | `http://<the API machine's LAN address>:2112` |

Android blocks plain `http://` unless the app allows cleartext traffic. iOS blocks it unless ATS allows that host. A production app should use `https://`.

```dart
import 'dart:typed_data';

import 'package:face_compare_sdk/face_compare_sdk.dart';

Future<void> matchPhoto(Uint8List capturedJpeg) async {
  final client = FaceCompareClient(
    baseUrl: 'http://10.0.2.2:2112',
    channelId: 'PORTAL_KLAIM',
  );
  try {
    print((await client.health()).device);
    final verdict = await client.compare(
      image1: ImageRef.url('http://your_url/kyc/ref.jpg'),
      image2: ImageRef.bytes(capturedJpeg),
      checkSpoof: true,
    );
    print('${verdict.isMatch} ${verdict.similarityScore} ${verdict.similarityPercentage}');
    print('${verdict.image2.isLive} ${verdict.image2.livenessScore}');
  } on FaceCompareException catch (error) {
    print('${error.errorCode} ${error.message}');
  } finally {
    client.close();
  }
}
```

`checkSpoof: true` is the spoof check on `image2`. `checkSpoofOnReference: true` is the same check on `image1`. Both flags map to the API fields `check_liveness_image2` and `check_liveness_image1`.

| You pass | What the SDK sends |
|---|---|
| `ImageRef.url(...)` or `ImageRef.raw(...)` | the string, for the API to download or decode |
| `ImageRef.bytes(...)` or `ImageRef.file(...)` | base64 |

`similarityThreshold`, `needConfirmationThreshold`, `traceId`, and `prevTransactionId` match the Python arguments. `FaceCompareException.errorCode` uses the same codes as the table in the Python section, including `TRANSPORT`.

### 4. Hugging Face Space

The public demo is the Gradio app in [huggingface/space](huggingface/space). It is published at https://huggingface.co/spaces/rizkynindra/face-compare. Visitors upload two photos, or take them with the phone camera, then press **Compare the photos**. The photos are stacked, one under the other, so each preview uses the full width. Matching uses AdaFace IR-101. The first start downloads that model (about 249 MB). The spoof check is optional and looks at the second photo only. Do not copy `.env` into the Space.

[Dockerfile](Dockerfile) is a separate CPU image of this FastAPI app for a Docker Space (`HF_SPACE=1`, port `7860`). It is not the Space that is running. Its card text is [huggingface/README.md](huggingface/README.md). The running Space card is [huggingface/space/README.md](huggingface/space/README.md).

---

## 📡 API Specification (`POST /v1/compare`)

#### Request Body
```json
{
  "ChId": "PORTAL_KLAIM",
  "image1": "http://your_url/kyc/ref_photo_123.jpg",
  "image2": {
    "type": "base64",
    "data": "/9j/4AAQSkZJRgABAQAAAQ..."
  },
  "options": {
    "check_liveness_image2": true,
    "similarity_threshold": 0.75,
    "need_confirmation_threshold": 0.40
  }
}
```

#### Response (`200 OK`)
```json
{
  "status": "SUCCESS",
  "ChId": "PORTAL_KLAIM",
  "result": {
    "is_match": "TRUE",
    "similarity_score": 0.8642,
    "similarity_percentage": "86.42%",
    "confidence_level": "HIGH"
  },
  "image1_metadata": {
    "faces_detected": 1,
    "bbox": [120.5, 80.2, 310.4, 320.1],
    "is_live": null,
    "liveness_score": null
  },
  "image2_metadata": {
    "faces_detected": 1,
    "bbox": [95.0, 75.3, 290.1, 315.8],
    "is_live": true,
    "liveness_score": 0.9955
  },
  "execution_time_ms": 68.4,
  "message": "Comparison finished with verdict 'TRUE' (Score: 0.8642)"
}
```

---

## Face gallery

This follows the [UniFace enroll-then-search flow](https://yakhyo.github.io/uniface/modules/recognition/#building-a-face-database). `POST /v1/compare` still compares two photos you send together. The gallery keeps a reference photo on disk and searches a new selfie against those stored faces.

The same lines as one-to-one compare apply: **0.75** is a match, **0.40** needs confirmation, and anything lower is a different person. Set `check_liveness` to `true` on the selfie when you want MiniFASNet to reject a printed photo, a screen, or a mask.

Stored photos live in `face_compare_api/data/face_db` unless `FACE_GALLERY_DIR` is set. When `FACE_GALLERY_ADMIN_TOKEN` is set, enroll, list, read, and delete require the header `X-Admin-Token`. Search stays open so another user can send a selfie.

Enroll a person:

```bash
curl -X POST http://127.0.0.1:2112/v1/gallery \
  -H 'Content-Type: application/json' \
  -d '{"person_id":"EMP001","label":"Ayu","image":"<base64 or url>"}'
```

Search with a selfie:

```bash
curl -X POST http://127.0.0.1:2112/v1/gallery/search \
  -H 'Content-Type: application/json' \
  -d '{"image":"<base64 or url>","options":{"check_liveness":true}}'
```

`GET /v1/gallery` lists people. `GET /v1/gallery/EMP001/photo` returns the stored JPEG. `DELETE /v1/gallery/EMP001` removes that person. OpenAPI for these routes is on `/docs`.

---

## 📊 Decision Threshold Matrix

| Cosine Similarity Score | Verdict (`is_match`) | Confidence | Description |
| :--- | :--- | :--- | :--- |
| **Score $\ge$ 0.75** | `TRUE` | `HIGH` | Same person confirmed. |
| **0.40 $\le$ Score < 0.75** | `Need Confirmation` | `MEDIUM` | Uncertain (aging/angle); routed for manual review. |
| **Score < 0.40** | `Different` | `LOW` | Different person detected. |

---

## 📝 Logging Specifications (Standard Log 6.1.1)

All service logs are written to `face_compare_api/logs/system_log_ddmmyyyy.log` with automatic daily rotation in **WIB (UTC+7)**:

```text
2026-08-20 15:28:11.511 [INFO] TR-648e1d26-4ead-4127-a788-f31bbba06a94 svc-face-compare /v1/compare REST POST SYNC application/json compare PORTAL_KLAIM 0.00 ms 172.26.7.5 10.128.4.2 Comparison Request TC-2c69db96-71b8-496e-a41e-92736aeae488 | START | 'Start Prediction' on 'Image' with result 'PENDING' with error ''. Processing request for ChId: PORTAL_KLAIM
2026-08-20 15:28:12.412 [INFO] TR-648e1d26-4ead-4127-a788-f31bbba06a94 svc-face-compare /v1/compare REST POST SYNC application/json compare PORTAL_KLAIM 900.96 ms 172.26.7.5 10.128.4.2 Comparison Result TC-2c69db96-71b8-496e-a41e-92736aeae488 Max Confidence: 1.0000 | STOP | 'Finish Prediction' on 'Image' with result 'TRUE' with error ''. Comparison result: TRUE. Similarity: 1.0000. Match: True
```

---

## ⚠️ Model Notes & Behavior

* **RetinaFace**: Detects faces and extracts 5 facial landmarks.
* **AdaFace IR-101**: Generates normalized 512-D embeddings. This is the matcher. It handles uneven photo quality better than the smaller ArcFace MobileNet model.
* **MiniFASNet**: Evaluates texture and moiré patterns to detect a printed photo, a screen replay, or a mask. It does not detect a deepfake of the same person. A convincing fake of that person still matches, because the face looks the same.
* **Lighting Consideration**: Detection confidence may drop under extremely bright, washed-out white backgrounds. Ensure adequate contrast when capturing live photos.
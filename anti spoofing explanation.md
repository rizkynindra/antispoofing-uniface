# Anti-Spoofing System Explanation

## Overview

This anti-spoofing system detects whether a face in a photo comes from a **live, physical human** standing in front of the camera, or from a **spoof attempt** such as:

- A photo taken off a smartphone screen
- A photo of a printed paper/document
- A photo from a monitor/display replay
- A mask or other physical spoof

The system uses two AI models working in sequence:

1. **RetinaFace** — Face detection model that locates faces in the image and returns bounding boxes + facial landmarks.
2. **MiniFASNet** — Face anti-spoofing model that analyzes the detected face region and predicts whether it is real or fake.

---

## How It Works (Step by Step)

```
Image URL → Download Image → Detect Faces (RetinaFace) → Anti-Spoofing Check (MiniFASNet) → Result
```

### Step 1: Image Download
The system downloads each image from the URL provided in the Excel input file directly into memory (no disk writing).

### Step 2: Face Detection (RetinaFace)
RetinaFace scans the image to detect all faces present. For each face, it returns:
- **Bounding box** coordinates (x1, y1, x2, y2)
- **Facial landmarks** (5 key points: left eye, right eye, nose, left mouth, right mouth)
- **Detection confidence** score

### Step 3: Anti-Spoofing Prediction (MiniFASNet)
The primary (largest) detected face is cropped from the image and fed into the MiniFASNet model. The model outputs a **softmax probability distribution** over two classes:

| Class Index | Label | Description |
|:-----------:|:-----:|:------------|
| 0 | **Fake** | Spoof / attack (photo from screen, print, etc.) |
| 1 | **Real** | Live person captured directly by camera |

The probability for class 1 (Real) becomes the **confidence score**.

### Step 4: Threshold Decision
The final `is_real` classification is determined by comparing the confidence score against the configured threshold:

```
is_real = confidence >= SPOOFING_THRESHOLD
```

---

## Output Columns in Result Excel

| Column | Type | Description |
|:-------|:-----|:------------|
| `faces_count` | Integer | Number of faces detected in the image. `0` = no face found. |
| `is_real` | Boolean | `TRUE` if real person, `FALSE` if spoof/fake. Blank if no face detected. |
| `confidence` | Float (0.0 – 1.0) | The model's probability score that the face is a real, live person. |
| `status` | String | Process execution status (see table below). |

### Status Values

| Status | Meaning |
|:-------|:--------|
| `Success` | Face detected and anti-spoofing analysis completed. |
| `No face detected` | RetinaFace found zero faces in the image. |
| `Empty URL` | The URL cell was empty or blank. |
| `Failed to decode image` | Image data could not be decoded (corrupted or unsupported format). |
| `HTTP Error: ...` | Failed to download image (network error, 404, timeout, etc.). |
| `Error: ...` | Unexpected processing error. |

---

## Understanding the Confidence Score

The `confidence` column is the **Real Person Probability Score** P(Real), ranging from `0.0000` (0%) to `1.0000` (100%).

### Interpretation Examples (with SPOOFING_THRESHOLD = 0.85)

| Photo Scenario | Confidence | is_real | Interpretation |
|:---------------|:----------:|:-------:|:---------------|
| Direct selfie of a live user | `0.9842` (98.4%) | **TRUE** | Model is 98.4% confident it's a live capture. Exceeds threshold. |
| Photo taken from a phone screen | `0.7230` (72.3%) | **FALSE** | Model detected screen artifacts (moiré, glare). Below threshold. |
| Printed paper photo held to camera | `0.1250` (12.5%) | **FALSE** | Model is 87.5% sure it is a fake/spoof photo. |
| Blurry or low-quality selfie | `0.8600` (86.0%) | **TRUE** | Barely passes threshold. Could be borderline. |

### Key Points
- **Higher confidence** = model is more certain the face is from a real, live person.
- **Lower confidence** = model suspects a spoof attack.
- The confidence value is **always the probability of being Real**, regardless of the final `is_real` decision.

---

## How Face Count Affects the Result

### faces_count == 0 (No Face Detected)
- `is_real` → **Blank** (None)
- `confidence` → **Blank** (None)
- `status` → `"No face detected"`
- Anti-spoofing model is **NOT executed** because there is no face bounding box to analyze.

### faces_count == 1 (Single Face)
- Standard case for selfie/onboarding photos.
- Anti-spoofing runs on the single detected face.

### faces_count > 1 (Multiple Faces)
- Anti-spoofing runs **only on the primary face** (`faces[0]`), which is the largest/most prominent face in the image.
- The face count number itself does **NOT alter** the confidence score calculation.
- However, multiple faces may indicate the photo was taken in a crowd or has background people.

> **Note**: If your onboarding policy requires strictly one person in the photo, you can use `faces_count > 1` or `faces_count == 0` as additional rejection criteria in post-processing.

---

## Threshold Configuration

The threshold is configured in the `.env` file:

```env
# Decision threshold for real face classification (0.0 to 1.0)
SPOOFING_THRESHOLD=0.85
```

### Threshold Guidelines

| Threshold | Strictness | Use Case |
|:---------:|:----------:|:---------|
| `0.50` | Low (default) | Standard sensitivity. May allow some borderline spoof photos through. |
| `0.70` | Medium | Moderate filtering. Catches obvious spoofs. |
| `0.85` | High (recommended) | Strict filtering. Effective against phone screen re-photography. |
| `0.90` | Very High | Very strict. May reject some legitimate low-quality selfies. |

### Trade-off
- **Higher threshold** → More spoofs rejected, but risk of rejecting legitimate low-quality selfies (false negatives).
- **Lower threshold** → Fewer legitimate photos rejected, but some spoofs may pass through (false positives).

---

## Known Limitations

- **Bright white backgrounds**: The model may struggle with faces against very bright white backgrounds, potentially affecting confidence scores. (Catatan: lemah terhadap background putih cerah)
- **Low-quality images**: Very blurry, dark, or heavily compressed images may produce unreliable confidence scores.
- **Unusual angles**: Extreme head rotations or partial face occlusion may reduce detection accuracy.

---

## Running the Script

### Using .env defaults (recommended)
```bash
uv run python process_excel_spoofing.py
```

### With CLI overrides
```bash
uv run python process_excel_spoofing.py \
    -i excel/Foto_Onboarding_.xlsx \
    -o result/result_anti_spoofing.xlsx \
    --threshold 0.85
```

### All Available Options
```
--input, -i       Input Excel file path
--output, -o      Output Excel file path
--column, -c      Column name containing image URLs
--threshold, -th  Anti-spoofing confidence threshold (0.0 to 1.0)
--timeout, -t     HTTP request timeout in seconds
```

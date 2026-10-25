"""Upload two face photos and compare them."""

import spaces
import cv2
import gradio as gr
import numpy as np

from uniface.constants import AdaFaceWeights
from uniface.detection import RetinaFace
from uniface.face_utils import compute_similarity
from uniface.recognition import AdaFace
from uniface.spoofing import MiniFASNet

MATCH_THRESHOLD = 0.70
CONFIRM_THRESHOLD = 0.40

detector = RetinaFace()
recognizer = AdaFace(model_name=AdaFaceWeights.IR_101)
spoofer = MiniFASNet()


@spaces.GPU(duration=1)
def _zerogpu_slot() -> None:
    """Keep this Space eligible for ZeroGPU. Comparison runs on CPU and does not call this."""
    return None


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _largest_face(image: np.ndarray):
    faces = detector.detect(image)
    if not faces:
        return None
    return max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


def _verdict(score: float) -> str:
    if score >= MATCH_THRESHOLD:
        return "TRUE"
    if score >= CONFIRM_THRESHOLD:
        return "Need Confirmation"
    return "Different"


def compare_faces(
    reference: np.ndarray | None,
    photo: np.ndarray | None,
    check_spoof: bool,
) -> str:
    """Compare a reference photo with a second photo.

    Matching always runs on the uploaded files. When check_spoof is set, the
    second photo is also checked for a printed photo, a screen replay, or a mask.
    """
    if reference is None or photo is None:
        return "Upload both photos."

    reference_bgr = _as_bgr(reference)
    photo_bgr = _as_bgr(photo)
    reference_face = _largest_face(reference_bgr)
    photo_face = _largest_face(photo_bgr)
    if reference_face is None:
        return "No face found in the reference photo."
    if photo_face is None:
        return "No face found in the second photo."

    reference_embedding = recognizer.get_normalized_embedding(reference_bgr, reference_face.landmarks)
    photo_embedding = recognizer.get_normalized_embedding(photo_bgr, photo_face.landmarks)
    score = float(compute_similarity(reference_embedding, photo_embedding, normalized=True))
    lines = [
        f"Verdict: {_verdict(score)}",
        f"Similarity: {score * 100:.2f}% (score {score:.4f})",
    ]
    if check_spoof:
        spoof = spoofer.predict(photo_bgr, photo_face.bbox)
        label = "real face" if spoof.is_real else "not a real face"
        lines.append(f"Second photo spoof check: {label} ({spoof.confidence:.2%})")
    return "\n".join(lines)


# Full-width previews, one under the other. A phone column is too narrow for two
# usable face previews. The compare control stays under both photos.
_LAYOUT_CSS = """
.compare-action button {
  min-height: 44px;
  width: 100%;
}
"""

_WEBCAM = gr.WebcamOptions(mirror=True, constraints={"video": {"facingMode": "user"}})

with gr.Blocks(title="Face compare", css=_LAYOUT_CSS) as demo:
    gr.Markdown("## Face compare")
    reference = gr.Image(
        label="Reference photo",
        type="numpy",
        height=320,
        sources=["upload", "webcam"],
        webcam_options=_WEBCAM,
    )
    photo = gr.Image(
        label="Second photo",
        type="numpy",
        height=320,
        sources=["upload", "webcam"],
        webcam_options=_WEBCAM,
    )
    compare_button = gr.Button("Compare the photos", variant="primary", elem_classes=["compare-action"])
    check_spoof = gr.Checkbox(
        label="Also check the second photo for a printed photo, screen, or mask",
        value=False,
    )
    gr.Markdown(
        "A score of 0.70 or more is the same person. "
        "A score from 0.40 up to 0.70 needs confirmation. Below 0.40 is a different person."
    )
    result = gr.Textbox(label="Result", lines=4)
    compare_button.click(
        fn=compare_faces,
        inputs=[reference, photo, check_spoof],
        outputs=result,
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True)

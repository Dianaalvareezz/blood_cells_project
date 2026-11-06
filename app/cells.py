"""
Shared pieces used by both tabs of the app: where the models are, which device to use
(Mac GPU or CPU), how to open an image, and how to run the cell detector (models/best.pt).
"""

from pathlib import Path

import gradio as gr  # used here only to show error messages in the page (gr.Error)
import torch  # PyTorch: used by YOLO; here only to check whether the Mac GPU is available
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

# Project root folder (the one that contains app/, models/, data/...).
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best.pt"  # the detector: finds RBC, WBC and platelets

# Default confidence threshold of the detector. 0.40 gave the best balance between hits and misses on the
# validation split (see reports/model/val/BoxF1_curve.png).
DEFAULT_CONF = 0.40

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Grey border added around the image (see detect_cells).
MODEL_SIZE = 640  # image size the detector was trained at
BORDER = 16  # border in pixels, measured at that 640 size
GRAY = (114, 114, 114)  # the same grey YOLO uses for padding


def choose_device():
    """Where the maths runs: the Mac GPU (MPS) if there is one, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = choose_device()
DEVICE_NAMES = {"mps": "your Mac's GPU (MPS)", "cpu": "the CPU"}

_detector = None  # the detector is loaded once and kept here


def get_model():
    """Loads the detector (only the first time) and checks it has the expected classes."""
    global _detector
    if _detector is None:
        if not MODEL_PATH.exists():
            raise gr.Error(
                f"The detector was not found at {MODEL_PATH}. "
                "Check that models/best.pt exists and restart the app."
            )
        model = YOLO(str(MODEL_PATH))
        # model.names is a dictionary {0: 'WBC', 1: 'RBC', 2: 'Platelets'} stored inside the model.
        unknown = set(model.names.values()) - {"RBC", "WBC", "Platelets"}
        if unknown:
            raise gr.Error(f"The detector has classes the app does not know: {sorted(unknown)}.")
        _detector = model
    return _detector


def predict(model, source, **settings):
    """Runs a YOLO model on the current device. If the Mac GPU fails, switches to the CPU and retries."""
    global DEVICE
    try:
        return model.predict(source, device=DEVICE, verbose=False, **settings)
    except RuntimeError:
        if DEVICE == "cpu":
            raise
        DEVICE = "cpu"
        return model.predict(source, device=DEVICE, verbose=False, **settings)


def open_image(path, check_extension=True):
    """
    Opens an image in colour (RGB). Returns None if it is not a valid image.
    With check_extension=True (tab 1) only .jpg/.jpeg/.png files are accepted; with False (tab 2, where
    pasted images can arrive as other formats) any image Pillow can read is accepted.
    """
    if check_extension and Path(path).suffix.lower() not in ALLOWED_EXTENSIONS:
        return None
    try:
        with Image.open(path) as img:
            # exif_transpose rotates the photo if the phone or camera saved it "sideways".
            return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return None


def detect_cells(image, conf):
    """
    Runs the detector on the image and returns a list of detections.
    Each detection is: (class, confidence, (x1, y1, x2, y2)), where (x1, y1) is the top-left
    corner of the box and (x2, y2) the bottom-right corner.
    """
    model = get_model()
    width, height = image.size

    # Important trick: when YOLO evaluates the model, it adds a grey border around each image. The model
    # learned to count that way: without the border, it also counts the bits of cells cut off by the edge
    # of the photo (on the test images that was ~3 extra red cells per image). So we add the same border.
    border = round(max(width, height) * BORDER / MODEL_SIZE)
    padded = ImageOps.expand(image, border=border, fill=GRAY)
    result = predict(model, padded, conf=conf, imgsz=MODEL_SIZE + 2 * BORDER, max_det=1000)[0]

    detections = []
    boxes = result.boxes
    for class_id, score, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
        # Subtract the border so the box fits the original photo (without the border).
        x1, y1, x2, y2 = (value - border for value in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        detections.append((model.names[int(class_id)], score, (x1, y1, x2, y2)))
    return detections

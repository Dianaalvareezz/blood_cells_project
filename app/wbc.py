"""
White-blood-cell typing: each white blood cell found by the detector is cropped, and the classifier
(models/wbc_classifier.pt) says which type it is.

How a crop is made so that it looks like the images the classifier learned from (BloodMNIST):
  1. Take a square centred on the detected box, CROP_SCALE times bigger than the box, so some of the
     surrounding cells are included. In BloodMNIST the white blood cell fills about 70% of the image.
  2. Shrink it to 64 x 64 pixels, the resolution of BloodMNIST, so the classifier sees the same level of
     detail as during training.
"""

import gradio as gr
from PIL import ImageOps
from ultralytics import YOLO

import cells  # shared code: device, predict()

CLASSIFIER_PATH = cells.ROOT / "models" / "wbc_classifier.pt"

CROP_SCALE = 1.4  # crop side = 1.4 x the longest side of the detected box
CLASSIFIER_INPUT = 64  # BloodMNIST image size
CLASSIFIER_IMGSZ = 224  # size the classifier was trained at (it upscales the 64 px crop itself)
GALLERY_SIZE = 160  # size of the crops shown in the gallery

# Below this confidence the type is shown as "uncertain" instead of being trusted.
UNCERTAIN_BELOW = 0.50

# The 6 white-blood-cell types (classifier class name -> name shown, box colour), in display order.
# The colours avoid the orange of red cells and the green of platelets.
WBC_TYPES = {
    "basophil": {"label": "Basophils", "color": "#d63384"},  # pink
    "eosinophil": {"label": "Eosinophils", "color": "#e34948"},  # red
    "immature_granulocytes": {"label": "Immature granulocytes", "color": "#8b5a2b"},  # brown
    "lymphocyte": {"label": "Lymphocytes", "color": "#00b8d9"},  # cyan
    "monocyte": {"label": "Monocytes", "color": "#6250d6"},  # violet
    "neutrophil": {"label": "Neutrophils", "color": "#2a78d6"},  # blue
}
UNCERTAIN = {"label": "Uncertain", "color": "#6b6b6b"}  # grey

# Classes the classifier knows that are NOT white blood cells (kept so that, for example, an erythroblast
# the detector mistook for a WBC is not forced into a WBC type). Such cells are shown as "uncertain".
NON_WBC_CLASSES = {"erythroblast": "Erythroblast", "platelet": "Platelet"}

_classifier = None  # loaded once and kept here


def get_classifier():
    """Loads the white-blood-cell classifier (only the first time) and checks its classes."""
    global _classifier
    if _classifier is None:
        if not CLASSIFIER_PATH.exists():
            raise gr.Error(
                f"The white-blood-cell classifier was not found at {CLASSIFIER_PATH}. "
                "Check that models/wbc_classifier.pt exists and restart the app."
            )
        model = YOLO(str(CLASSIFIER_PATH))
        unknown = set(model.names.values()) - set(WBC_TYPES) - set(NON_WBC_CLASSES)
        if unknown:
            raise gr.Error(f"The classifier has classes the app does not know: {sorted(unknown)}.")
        _classifier = model
    return _classifier


def median_color(image):
    """Median colour of a small version of the photo: in a blood smear that is the background."""
    small = image.resize((64, 64))
    channels = [sorted(band.getdata()) for band in small.split()]
    return tuple(c[len(c) // 2] for c in channels)


def crop_cell(image, box, scale=CROP_SCALE):
    """
    Cuts a square around a detected cell. If the square goes past the edge of the photo, the missing part
    is filled with the photo's background colour, so the cell stays centred.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * scale / 2
    left, top, right, bottom = (round(v) for v in (cx - half, cy - half, cx + half, cy + half))
    pad = max(0, -left, -top, right - image.width, bottom - image.height)
    if pad:
        image = ImageOps.expand(image, border=pad, fill=median_color(image))
        left, top, right, bottom = left + pad, top + pad, right + pad, bottom + pad
    return image.crop((left, top, right, bottom))


def classify(crops):
    """Classifies a list of crops. Returns a list of (class name, confidence between 0 and 1)."""
    if not crops:
        return []
    model = get_classifier()
    small = [c.resize((CLASSIFIER_INPUT, CLASSIFIER_INPUT), resample=3) for c in crops]  # 3 = bicubic
    results = cells.predict(model, small, imgsz=CLASSIFIER_IMGSZ)
    return [(model.names[int(r.probs.top1)], float(r.probs.top1conf)) for r in results]


def classify_white_cells(image, wbc_detections):
    """
    Takes the detections the detector labelled "WBC" and returns one dictionary per white blood cell:
    its number, box, crop, predicted type and confidence, and whether it is "uncertain".
    """
    ordered = sorted(wbc_detections, key=lambda d: (d[2][1], d[2][0]))  # number them top to bottom
    crops = [crop_cell(image, box) for _name, _score, box in ordered]
    found = []
    for i, ((_name, det_conf, box), crop, (cls, conf)) in enumerate(zip(ordered, crops, classify(crops))):
        uncertain = cls not in WBC_TYPES or conf < UNCERTAIN_BELOW
        if cls in WBC_TYPES:
            predicted = WBC_TYPES[cls]["label"].rstrip("s")  # "Neutrophils" -> "Neutrophil"
        else:
            predicted = NON_WBC_CLASSES.get(cls, cls) + " (not a WBC)"
        found.append({
            "n": i + 1, "box": box, "det_conf": det_conf, "crop": crop,
            "class": cls, "conf": conf, "predicted": predicted, "uncertain": uncertain,
            # what is written on the image and in the gallery
            "text": (f"{i + 1} · Uncertain ({predicted} {100 * conf:.0f}%)" if uncertain
                     else f"{i + 1} · {predicted} {100 * conf:.0f}%"),
            "color": UNCERTAIN["color"] if uncertain else WBC_TYPES[cls]["color"],
        })
    return found


def gallery_items(found):
    """Crops for the gallery: (image, caption) pairs."""
    return [(c["crop"].resize((GALLERY_SIZE, GALLERY_SIZE), resample=3), c["text"]) for c in found]

"""
Web app that analyses a blood smear photo with our two trained YOLO26 models, in one view:

  1. The detector (models/best.pt) finds every cell and labels it RBC, WBC or Platelets.
  2. Each white blood cell (WBC) is cropped and the classifier (models/wbc_classifier.pt) says its type:
     basophil, eosinophil, immature granulocyte, lymphocyte, monocyte or neutrophil.
  3. The app draws the boxes, counts everything, and lets you download the image and a CSV.

The code is split into three files:
  app/cells.py  shared pieces: device (Mac GPU or CPU), the detector, opening images
  app/wbc.py    the white-blood-cell classifier: crop, classify, "uncertain" rule
  app/app.py    this file: drawing, counting and the web page

How to start it (in the Terminal, from the project folder):
    .venv/bin/python app/app.py
Then open the address it prints in the browser (usually http://127.0.0.1:7860).
To stop it: go back to the Terminal and press Ctrl + C.
"""

import html
import tempfile
from pathlib import Path

import gradio as gr  # Gradio: builds the web page from Python functions
import pandas as pd  # pandas: writes the CSV file
from PIL import Image, ImageDraw, ImageFont

import cells  # shared code (app/cells.py)
import wbc  # white-blood-cell classifier (app/wbc.py)

# Box colours for the cells that are not white blood cells (same colours as before).
RBC_COLOR = "#FF8C00"  # orange
PLATELET_COLOR = "#1FAA4B"  # green

# ---------------------------------------------------------------------------
# 1. DRAWING THE BOXES
# ---------------------------------------------------------------------------


def text_color(hex_color):
    """Black or white text, whichever reads better on a label of this colour."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#ffffff"


def draw_results(image, detections, white_cells):
    """
    Returns a copy of the photo with:
      - a thin orange box on each red blood cell and a green box on each platelet,
      - a thicker box on each white blood cell, in the colour of its type, with a label "number · type %".
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    thin = max(2, round(max(image.size) / 250))
    thick = thin + 1
    for name, _score, box in detections:
        if name == "RBC":
            draw.rectangle(box, outline=RBC_COLOR, width=thin)
        elif name == "Platelets":
            draw.rectangle(box, outline=PLATELET_COLOR, width=thin)

    font = ImageFont.load_default(size=max(12, round(max(image.size) / 40)))
    for cell in white_cells:
        x1, y1, x2, y2 = cell["box"]
        draw.rectangle((x1, y1, x2, y2), outline=cell["color"], width=thick)
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), cell["text"], font=font)
        w, h = tx2 - tx1 + 8, ty2 - ty1 + 6
        lx = min(max(0, x1), max(0, image.width - w))  # keep the label inside the photo
        ly = y1 - h if y1 - h >= 0 else y1  # above the box, or inside it if the box touches the top
        draw.rectangle((lx, ly, lx + w, ly + h), fill=cell["color"])
        draw.text((lx + 4 - tx1, ly + 3 - ty1), cell["text"], fill=text_color(cell["color"]), font=font)
    return annotated


# ---------------------------------------------------------------------------
# 2. COUNTING
# ---------------------------------------------------------------------------


def count_everything(detections, white_cells):
    """
    Returns the rows of the results, in display order: (category, count, % of WBC, is it a sub-row?).
    Uncertain white blood cells are still white blood cells: they count in the WBC total and get their own row.
    """
    n_rbc = sum(d[0] == "RBC" for d in detections)
    n_plt = sum(d[0] == "Platelets" for d in detections)
    n_wbc = len(white_cells)

    def pct(n):
        return f"{100 * n / n_wbc:.0f}%" if n_wbc else "–"

    rows = [
        ("Red blood cells (RBC)", n_rbc, "", False),
        ("Platelets", n_plt, "", False),
        ("White blood cells (WBC)", n_wbc, "", False),
    ]
    for class_name, info in wbc.WBC_TYPES.items():
        n = sum(c["class"] == class_name and not c["uncertain"] for c in white_cells)
        rows.append((info["label"], n, pct(n), True))
    n_unc = sum(c["uncertain"] for c in white_cells)
    if n_unc:
        rows.append(("Uncertain (low confidence or not a WBC type)", n_unc, pct(n_unc), True))
    rows.append(("Total cells", n_rbc + n_plt + n_wbc, "", False))
    return rows


def results_html(rows):
    """The results as an indented table: WBC types appear under the WBC total."""
    body = []
    for label, count, pct, sub in rows:
        cls = "sub" if sub else ("total" if label == "Total cells" else "main")
        pct_text = f'<span class="pct">({pct} of WBC)</span>' if sub else ""
        body.append(f'<tr class="{cls}"><td>{html.escape(label)}</td><td>{count} {pct_text}</td></tr>')
    return f'<table class="results">{"".join(body)}</table>'


def legend_html():
    """Colour legend for every kind of box."""
    entries = [("Red blood cells (RBC)", RBC_COLOR), ("Platelets", PLATELET_COLOR)]
    entries += [(info["label"], info["color"]) for info in wbc.WBC_TYPES.values()]
    entries.append(("Uncertain WBC", wbc.UNCERTAIN["color"]))
    items = "".join(
        f'<span><span class="swatch" style="border-color:{color}"></span>{html.escape(label)}</span>'
        for label, color in entries
    )
    return f'<div class="legend"><span class="legend-title">Box colours:</span>{items}</div>'


# ---------------------------------------------------------------------------
# 3. WHAT HAPPENS WHEN AN IMAGE IS ADDED (or the slider moves, or "Analyze" is clicked)
# ---------------------------------------------------------------------------

START_MESSAGE = "Upload, drag or paste (Cmd+V) a blood smear photo to start."


def analyze(image_path, conf):
    """Runs both models on the photo and prepares everything the page shows."""
    if not image_path:
        return None, START_MESSAGE, "", [], None

    # Pasted images can come in other formats (for example TIFF from Safari), so any format Pillow can
    # read is accepted, not only JPG and PNG.
    image = cells.open_image(image_path, check_extension=False)
    if image is None:
        raise gr.Error("I can't open that file as an image. Try a JPG or PNG photo.")

    try:
        detections = cells.detect_cells(image, conf)  # step 1: all cells, labelled RBC / WBC / Platelets
        wbc_detections = [d for d in detections if d[0] == "WBC"]
        white_cells = wbc.classify_white_cells(image, wbc_detections)  # step 2: type of each WBC
    except gr.Error:
        raise  # already a clear message (for example, a model file not found)
    except Exception as error:
        raise gr.Error(f"Something went wrong while analysing the image. Technical detail: {error}") from error

    rows = count_everything(detections, white_cells)

    # Files to download: the annotated image and a CSV with all the counts.
    out_dir = Path(tempfile.mkdtemp(prefix="blood_cells_"))
    annotated_path = out_dir / "annotated_cells.png"
    draw_results(image, detections, white_cells).save(annotated_path)
    csv_path = out_dir / "cell_counts.csv"
    pd.DataFrame(
        [{"category": label, "count": count, "percent_of_wbc": pct} for label, count, pct, _sub in rows]
        # written as text so the "count" column keeps whole numbers (25, not 25.0)
        + [{"category": "Detector confidence threshold", "count": f"{conf:.2f}", "percent_of_wbc": ""}]
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Message above the results.
    device = cells.DEVICE_NAMES[cells.DEVICE]
    n_unc = sum(c["uncertain"] for c in white_cells)
    if not detections:
        message = (
            "⚠️ **No cells were detected in this image.** Check that it is a microscope photo of a stained "
            "blood smear, or try lowering the confidence threshold."
        )
    elif not white_cells:
        message = (
            f"⚠️ **No white blood cells were found** (but {len(detections)} other cells were). Many smear "
            "photos simply have none. If you can see one, try lowering the confidence threshold."
        )
    else:
        message = f"✅ Analysed on {device}. The % on each white blood cell is how sure the classifier is."
        if n_unc:
            message += (
                f"  \n⚠️ {n_unc} white blood cell(s) marked **uncertain** (grey): the classifier was less than "
                f"{100 * wbc.UNCERTAIN_BELOW:.0f}% sure, or thought it was not a white blood cell. Check them in "
                "the gallery."
            )
    return str(annotated_path), message, results_html(rows), wbc.gallery_items(white_cells), [
        str(annotated_path), str(csv_path)]


# ---------------------------------------------------------------------------
# 4. THE WEB PAGE
# ---------------------------------------------------------------------------

CSS = """
.legend {display: flex; gap: 8px 18px; flex-wrap: wrap; align-items: center; padding: 4px 2px;}
.legend-title {font-weight: 600;}
.swatch {display: inline-block; width: 14px; height: 14px; border: 3px solid; border-radius: 2px;
         margin-right: 6px; vertical-align: -3px;}
.results {border-collapse: collapse; min-width: 340px; font-size: 1rem;}
.results td {padding: 5px 12px; border-bottom: 1px solid var(--border-color-primary);}
.results td:last-child {text-align: right; font-variant-numeric: tabular-nums; font-weight: 600;}
.results tr.sub td:first-child {padding-left: 36px; font-weight: 400;}
.results tr.sub td {color: var(--body-text-color-subdued);}
.results tr.main td:first-child, .results tr.total td {font-weight: 700;}
.results tr.total td {border-top: 2px solid var(--body-text-color); border-bottom: 0;}
.results .pct {font-weight: 400; margin-left: 6px;}
"""

# Lets you paste an image with Cmd+V (Ctrl+V on Windows). Gradio's own "paste" button needs a click, so
# this listens for the paste shortcut on the whole page and hands the pasted image to the upload box,
# exactly as if you had dropped a file on it.
PASTE_JS = """
() => {
  const box = () => document.querySelector("#smear-input");
  const fileInput = () => box()?.querySelector('input[type="file"]');
  const wait = ms => new Promise(r => setTimeout(r, ms));
  document.addEventListener("paste", async (event) => {
    const item = [...(event.clipboardData?.items || [])].find(i => i.kind === "file" && i.type.startsWith("image/"));
    if (!item || !box()) return;
    event.preventDefault();
    const blob = item.getAsFile();
    const ext = (blob.type.split("/")[1] || "png").replace("jpeg", "jpg").replace("+xml", "");
    const file = new File([blob], `pasted_image.${ext}`, { type: blob.type });
    // If a photo is already loaded, clear it first so the upload box appears again.
    if (!fileInput()) {
      const clear = [...box().querySelectorAll("button")].find(b =>
        /clear|remove/i.test((b.getAttribute("aria-label") || "") + (b.title || "")));
      clear?.click();
      for (let i = 0; i < 40 && !fileInput(); i++) await wait(50);
    }
    const input = fileInput();
    if (!input) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}
"""

with gr.Blocks(title="Blood cell analysis", analytics_enabled=False) as demo:
    gr.Markdown(
        "# 🩸 Blood cell analysis\n"
        "Add a photo of a **blood smear**. The first model finds every cell and labels it as a red blood cell, "
        "white blood cell or platelet. A second model then looks at each **white blood cell** and says its "
        "**type**. Everything runs on your computer; the image never leaves it.\n\n"
        "**Three ways to add a photo (JPG or PNG):** click the box, drag a file onto it, or copy an image from "
        "anywhere (a website, a screenshot...) and press **Cmd + V** on this page."
    )
    for path in (cells.MODEL_PATH, wbc.CLASSIFIER_PATH):
        if not path.exists():
            gr.Markdown(f"⚠️ **Model not found: `{path}`.** The app can't analyse images until it exists.")

    with gr.Row():
        image_input = gr.Image(
            label="Original image: click, drag a file here, or paste with Cmd+V",
            sources=["upload", "clipboard"],
            type="filepath",
            elem_id="smear-input",
            height=460,
        )
        annotated_output = gr.Image(label="Cells found", interactive=False, height=460)
    gr.HTML(legend_html())
    with gr.Row():
        conf_slider = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=cells.DEFAULT_CONF,
            step=0.05,
            label="Confidence threshold",
            info="The detector only marks cells it is at least this sure about: raise it if you see boxes on "
            "things that are not cells, lower it if cells are missed.",
            scale=3,
        )
        analyze_button = gr.Button("Analyze", variant="primary", scale=1)
    status = gr.Markdown(START_MESSAGE)
    results = gr.HTML()
    gallery = gr.Gallery(
        label="Each white blood cell as the classifier sees it (number · type · confidence)",
        columns=6,
        fit_columns=False,  # keep thumbnails small even with only 1 or 2 white blood cells
        height="auto",
        object_fit="contain",
        preview=False,  # start as a grid of thumbnails; click one to enlarge it
    )
    downloads = gr.File(label="Downloads: annotated image and CSV with all the counts", file_count="multiple",
                        interactive=False)

    # Analyse when an image is added (upload, drag or paste), when the slider is released,
    # and when "Analyze" is clicked.
    inputs = [image_input, conf_slider]
    outputs = [annotated_output, status, results, gallery, downloads]
    image_input.change(analyze, inputs, outputs)
    conf_slider.release(analyze, inputs, outputs)
    analyze_button.click(analyze, inputs, outputs, api_name="analyze")


if __name__ == "__main__":
    # Load both models and run a first prediction at start-up, so the first photo is not slow.
    blank = Image.new("RGB", (64, 64), "white")
    try:
        cells.detect_cells(blank, cells.DEFAULT_CONF)
        print(f"Detector loaded from {cells.MODEL_PATH} (running on {cells.DEVICE}).")
        wbc.classify([blank])
        print(f"White-blood-cell classifier loaded from {wbc.CLASSIFIER_PATH}.")
    except gr.Error as error:
        print(f"WARNING: {error.message}")

    # server_name="127.0.0.1" = the page can only be opened from this computer.
    # Gradio uses port 7860 and, if it is busy, the next free one (7861, 7862...).
    demo.launch(server_name="127.0.0.1", max_file_size="25mb", theme=gr.themes.Soft(), css=CSS, js=PASTE_JS)

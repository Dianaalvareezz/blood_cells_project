# Blood cell detection with YOLO26

Two YOLO26 models for blood smear images, trained on a Kaggle GPU, with a local HTML dashboard and a
local web app that combines them:

1. **Detector** (`models/best.pt`): finds every cell and labels it **WBC**, **RBC** or **Platelets**
   (TXL-PBC dataset). Report: [reports/REPORT.md](reports/REPORT.md)
2. **White-blood-cell classifier** (`models/wbc_classifier.pt`): says the type of a white blood cell
   (basophil, eosinophil, immature granulocyte, lymphocyte, monocyte, neutrophil, plus erythroblast as a
   "not a WBC" class). Trained on BloodMNIST. Reports: [reports/wbc_dataset/](reports/wbc_dataset/REPORT.md),
   [reports/wbc_model/](reports/wbc_model/REPORT.md)

## Layout

| Path | What it is |
|---|---|
| `data/TXL-PBC/` | Dataset config and README. Images and labels are not committed. |
| `data/bloodminst/` | BloodMNIST (64x64 px, `train/val/test/<class>/`). Not committed (in `.gitignore`). |
| `kaggle/gpu_check/` | Kaggle kernel that checks the GPU, CUDA, Ultralytics and dataset mount before training |
| `kaggle/train/` | Kaggle training kernel for the detector. Prints a `METRICS {json}` line each epoch. |
| `kaggle/train_wbc/` | Kaggle training kernel for the WBC classifier (YOLO26s-cls, early stopping) |
| `scripts/dataset_report.py` | Dataset statistics and figures, written to `reports/dataset/` |
| `scripts/wbc_dataset_report.py` | BloodMNIST statistics and figures, written to `reports/wbc_dataset/` |
| `scripts/monitor.py` | Polls a Kaggle kernel (`--job detector` or `--job wbc`), serves the dashboard, downloads outputs when the run finishes |
| `scripts/make_reports.py` | Builds `reports/training/`, `reports/model/`, `reports/REPORT.md` and `models/best.pt` |
| `scripts/make_wbc_reports.py` | Builds `reports/wbc_model/` and `models/wbc_classifier.pt` |
| `scripts/wbc_pipeline_check.py` | Tests detector + classifier together on TXL-PBC photos, written to `reports/wbc_pipeline/` |
| `dashboard/index.html` | Dashboard with Training, Model performance, Dataset and WBC classifier tabs |
| `app/` | Local web app combining both models (`app.py` page, `cells.py` detector, `wbc.py` classifier) |

## Setup

You need Python 3.11 or newer, because the Kaggle CLI 2.x requires it.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
echo "KAGGLE_API_TOKEN=<your token>" > .env          # a bare token on one line also works
```

## Running

```bash
# 1. Dataset report (needs data/TXL-PBC/images + labels locally)
.venv/bin/python scripts/dataset_report.py

# 2. Upload the dataset to Kaggle as a private dataset (first time only)
#    Stage the data in build/kaggle_dataset/ with a dataset-metadata.json, then run:
#    kaggle datasets create -p build/kaggle_dataset --dir-mode zip

# 3. Check the GPU, then train (the kaggle CLI reads KAGGLE_API_TOKEN from the environment)
kaggle kernels push -p kaggle/gpu_check --accelerator NvidiaTeslaT4
kaggle kernels push -p kaggle/train     --accelerator NvidiaTeslaT4

# 4. Monitor: open http://localhost:8000/dashboard/
.venv/bin/python scripts/monitor.py
```

When the kernel reaches `COMPLETE`, the monitor downloads the outputs and generates the reports. The
dashboard refreshes every 30 s.

Kaggle may publish a kernel's log only after the run ends. Until then, the dashboard shows the
kernel status and elapsed time. The per-epoch curves appear when the log becomes available.

### White-blood-cell classifier

```bash
# 1. Dataset report (needs data/bloodminst/ locally)
.venv/bin/python scripts/wbc_dataset_report.py

# 2. Upload the 7 classes used (platelet removed) as a private Kaggle dataset (first time only)
#    Stage them in build/kaggle_wbc_dataset/bloodmnist_wbc/{train,val,test}/<class>/ with a
#    dataset-metadata.json, then: kaggle datasets create -p build/kaggle_wbc_dataset --dir-mode zip

# 3. Train, then monitor (the WBC classifier tab of the dashboard)
kaggle kernels push -p kaggle/train_wbc --accelerator NvidiaTeslaT4
.venv/bin/python scripts/monitor.py --job wbc

# 4. Check detector + classifier together on TXL-PBC photos
.venv/bin/python scripts/wbc_pipeline_check.py
```

## How to use the web app

The app (`app/app.py`, built with Gradio) runs both models on a blood smear photo, in a single view.
Everything runs on your computer: it uses your Mac's GPU (MPS) if available, otherwise the CPU.

**The first time**, install the libraries in the virtual environment (from the project folder):

```bash
cd ~/Desktop/blood_cells_project
uv pip install --python .venv/bin/python -r requirements.txt
```

**To start it** (every time you want to use it):

```bash
cd ~/Desktop/blood_cells_project
.venv/bin/python app/app.py
```

Wait until the Terminal shows `Running on local URL:  http://127.0.0.1:7860` and open that address in
your browser. If port 7860 is busy, Gradio uses the next free one (7861, 7862...), so always open the
address the Terminal shows.

**To stop it**: go back to the Terminal window where it is running and press `Ctrl + C`.

On the page:

1. Add a photo: click the box, drag a JPG or PNG onto it, or copy an image from anywhere and press
   **Cmd + V**. The analysis starts by itself (or click **Analyze**).
2. Next to the original you see the boxes: **orange** = red blood cell, **green** = platelet, and each white
   blood cell in the colour of its type with a label "number · type %". The legend lists every colour.
3. Below: red blood cells, platelets, white blood cells (with each type's count and % of WBC) and the total.
4. White blood cells the classifier is unsure about (under 50% confidence, or classified as a non-WBC such as
   an erythroblast) are shown as **Uncertain** in grey, still counted as white blood cells.
5. The gallery shows each white blood cell crop with its type and confidence.
6. The **Confidence threshold** slider (0.40 by default) sets how sure the detector must be to mark a cell.
7. **Downloads**: the annotated image and a CSV with all the counts, including WBC types.

To count like the detector's validation, the app adds the same grey border around the image that YOLO adds
when it evaluates, so cells cut by the edge of the photo are not counted as extra cells.

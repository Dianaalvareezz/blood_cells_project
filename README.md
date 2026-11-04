# Blood cell detection with YOLO26

A YOLO26 model fine-tuned to detect and classify blood cells (**WBC**, **RBC**, **Platelets**) in
blood smear images from the TXL-PBC dataset. Training runs on a Kaggle GPU, and progress and results
show in a local HTML dashboard.

- Results summary: [reports/REPORT.md](reports/REPORT.md)
- Trained weights: `models/best.pt`

## Layout

| Path | What it is |
|---|---|
| `data/TXL-PBC/` | Dataset config and README. Images and labels are not committed. |
| `kaggle/gpu_check/` | Kaggle kernel that checks the GPU, CUDA, Ultralytics and dataset mount before training |
| `kaggle/train/` | Kaggle training kernel. Prints a `METRICS {json}` line each epoch. |
| `scripts/dataset_report.py` | Dataset statistics and figures, written to `reports/dataset/` |
| `scripts/monitor.py` | Polls the Kaggle kernel, serves the dashboard and downloads outputs when the run finishes |
| `scripts/make_reports.py` | Builds `reports/training/`, `reports/model/`, `reports/REPORT.md` and `models/best.pt` |
| `dashboard/index.html` | Dashboard with Training, Model performance and Dataset tabs |

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

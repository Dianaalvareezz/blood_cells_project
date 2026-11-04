"""Build training-progress and model-performance reports from the downloaded Kaggle outputs.

Inputs : runs/kaggle/ (downloaded by scripts/monitor.py)
Outputs: reports/training/, reports/model/, reports/REPORT.md, models/best.pt
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs" / "kaggle"
TR = ROOT / "reports" / "training"
MO = ROOT / "reports" / "model"
CLASSES = ["WBC", "RBC", "Platelets"]
COLORS = ["#7c3aed", "#dc2626", "#0891b2"]


def find(name):
    hits = sorted(SRC.rglob(name))
    return hits[0] if hits else None


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def training_report():
    TR.mkdir(parents=True, exist_ok=True)
    csv = find("results.csv")
    if csv is None:
        raise SystemExit("results.csv not found in runs/kaggle — has the kernel finished?")
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    df.to_csv(TR / "results.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, loss in zip(axes, ["box_loss", "cls_loss", "dfl_loss"]):
        for split, style in (("train", "-"), ("val", "--")):
            col = f"{split}/{loss}"
            if col in df:
                ax.plot(df["epoch"], df[col], style, label=split)
        ax.set_title(loss)
        ax.set_xlabel("epoch")
        ax.legend(frameon=False)
    fig.suptitle("Losses")
    save(fig, TR / "losses.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, lab in (("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"),
                     ("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")):
        if col in df:
            ax.plot(df["epoch"], df[col], label=lab)
    ax.set_ylim(0, 1)
    ax.set_xlabel("epoch")
    ax.set_title("Validation metrics per epoch")
    ax.legend(frameon=False)
    save(fig, TR / "val_metrics.png")

    for f in ("results.png", "labels.jpg", "train_batch0.jpg", "train_batch1.jpg", "args.yaml"):
        p = find(f)
        if p:
            shutil.copy(p, TR / f)
    return df


def model_report():
    MO.mkdir(parents=True, exist_ok=True)
    fm = find("final_metrics.json")
    final = json.loads(fm.read_text())
    (MO / "final_metrics.json").write_text(json.dumps(final, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        pc = final[split]["per_class"]
        metrics = ["precision", "recall", "mAP50", "mAP50-95"]
        w = 0.2
        for i, m in enumerate(metrics):
            vals = [pc[c][m] for c in CLASSES]
            b = ax.bar([j + (i - 1.5) * w for j in range(3)], vals, w, label=m)
            ax.bar_label(b, fmt="%.2f", fontsize=7)
        ax.set_xticks(range(3), CLASSES)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{split} split")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Per-class performance of best.pt")
    save(fig, MO / "per_class.png")

    for split in ("val", "test"):
        d = MO / split
        d.mkdir(exist_ok=True)
        ev = next((p for p in SRC.rglob(f"eval_{split}") if p.is_dir()), None)
        if ev:
            for p in ev.iterdir():
                if p.suffix in (".png", ".jpg"):
                    shutil.copy(p, d / p.name)
    pred = next((p for p in SRC.rglob("predictions") if p.is_dir()), None)
    if pred:
        d = MO / "predictions"
        d.mkdir(exist_ok=True)
        for p in sorted(pred.iterdir()):
            if p.suffix.lower() in (".png", ".jpg"):
                shutil.copy(p, d / p.name)

    manifest = {sub: sorted(p.name for p in (MO / sub).glob("*") if p.suffix.lower() in (".png", ".jpg"))
                for sub in ("val", "test", "predictions") if (MO / sub).exists()}
    (MO / "manifest.json").write_text(json.dumps(manifest, indent=2))

    best = find("best.pt")
    if best:
        (ROOT / "models").mkdir(exist_ok=True)
        shutil.copy(best, ROOT / "models" / "best.pt")
    return final


def write_md(df, final):
    ds = json.loads((ROOT / "reports" / "dataset" / "summary.json").read_text())
    cfg = final["cfg"]
    best_row = df.loc[df["metrics/mAP50-95(B)"].idxmax()]

    def table(split):
        r = final[split]
        lines = ["| Class | Precision | Recall | mAP50 | mAP50-95 |", "|---|---:|---:|---:|---:|"]
        for c in CLASSES:
            m = r["per_class"][c]
            lines.append(f"| {c} | {m['precision']:.3f} | {m['recall']:.3f} | {m['mAP50']:.3f} | {m['mAP50-95']:.3f} |")
        lines.append(f"| **All** | **{r['precision']:.3f}** | **{r['recall']:.3f}** | **{r['mAP50']:.3f}** | **{r['mAP50-95']:.3f}** |")
        return "\n".join(lines)

    split_rows = "\n".join(
        f"| {s} | {v['images']} | {v['per_class']['WBC']} | {v['per_class']['RBC']} | {v['per_class']['Platelets']} |"
        for s, v in ds["splits"].items())
    md = f"""# Blood-cell detection with YOLO26: report

## Dataset (TXL-PBC)

{ds['total_images']} images, {ds['total_boxes']} annotated cells, 3 classes. No missing or invalid labels.

| Split | Images | WBC | RBC | Platelets |
|---|---:|---:|---:|---:|
{split_rows}

RBCs make up about 90% of the annotations and platelets about 3%, so the classes are heavily imbalanced.
Figures: [reports/dataset/](dataset/).

## Training

- Model: `{cfg['model']}` (COCO-pretrained), fine-tuned on a Kaggle Tesla T4 GPU
- Image size {cfg['imgsz']}, batch {cfg['batch']}, up to {cfg['epochs']} epochs (early-stopping patience {cfg['patience']}), cosine LR
- Ran {len(df)} epochs in {final['train_time_s'] / 60:.1f} min, including the final evaluation
- Best validation mAP50-95: {best_row['metrics/mAP50-95(B)']:.3f}, at epoch {int(best_row['epoch'])}

Figures: [reports/training/](training/).

## Model performance (`models/best.pt`)

### Test split (held out, never used for training or model selection)

{table('test')}

### Validation split

{table('val')}

Inference speed on the T4: {final['test']['speed_ms'].get('inference', 0):.1f} ms per image.
Confusion matrices and PR curves: [reports/model/](model/).
"""
    (ROOT / "reports" / "REPORT.md").write_text(md)


if __name__ == "__main__":
    df = training_report()
    final = model_report()
    write_md(df, final)
    print("Reports written to reports/")

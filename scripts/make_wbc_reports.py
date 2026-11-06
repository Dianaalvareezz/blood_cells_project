"""Build the white-blood-cell classifier report from the downloaded Kaggle outputs.

Inputs : runs/kaggle_wbc/ (downloaded by `scripts/monitor.py --job wbc`)
Outputs: reports/wbc_model/ (metrics, confusion matrix, per-class precision/recall, training curves,
         examples of mistakes, REPORT.md) and models/wbc_classifier.pt
"""
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs" / "kaggle_wbc"
OUT = ROOT / "reports" / "wbc_model"
DATA = ROOT / "data" / "bloodminst"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3


def find(name):
    hits = sorted(SRC.rglob(name))
    return hits[0] if hits else None


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=120, facecolor="white")
    plt.close(fig)


def pretty(c):
    return c.replace("_", " ")


def training_curves():
    df = pd.read_csv(find("results.csv"))
    df.columns = [c.strip() for c in df.columns]
    df.to_csv(OUT / "results.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(df["epoch"], df["metrics/accuracy_top1"], color=BLUE, linewidth=2)
    best = df["metrics/accuracy_top1"].idxmax()
    ax.annotate(f"best {df['metrics/accuracy_top1'][best]:.3f} (epoch {int(df['epoch'][best])})",
                (df["epoch"][best], df["metrics/accuracy_top1"][best]), textcoords="offset points",
                xytext=(-10, -18), ha="right", fontsize=8, color=MUTED)
    ax.set_title("Validation accuracy per epoch", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("epoch", color=MUTED)
    style(ax)
    ax = axes[1]
    ax.plot(df["epoch"], df["train/loss"], color=BLUE, linewidth=2, label="train")
    ax.plot(df["epoch"], df["val/loss"], color=ORANGE, linewidth=2, label="validation")
    ax.set_title("Loss per epoch (lower is better)", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("epoch", color=MUTED)
    ax.legend(frameon=False)
    style(ax)
    save(fig, "training_curves.png")
    return df


def confusion(final, split="test"):
    r = final[split]
    cm = np.array(r["confusion_matrix"])
    classes = r["classes"]
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, f"{100 * norm[i, j]:.0f}%\n{cm[i, j]}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(classes)), [pretty(c) for c in classes], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), [pretty(c) for c in classes], fontsize=8)
    ax.set_xlabel("predicted class", color=MUTED)
    ax.set_ylabel("true class", color=MUTED)
    ax.set_title(f"Confusion matrix, {split} split (row %, image count)", loc="left", color=INK, fontsize=11)
    save(fig, f"confusion_matrix_{split}.png")


def per_class(final, split="test"):
    pc = final[split]["per_class"]
    classes = list(pc)
    x = np.arange(len(classes))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, (m, col) in enumerate((("precision", BLUE), ("recall", ORANGE), ("f1", AQUA))):
        b = ax.bar(x + (i - 1) * w, [pc[c][m] for c in classes], w - 0.02, color=col, label=m)
        ax.bar_label(b, fmt="%.2f", fontsize=7, color=MUTED, padding=2)
    ax.set_xticks(x, [pretty(c).replace(" ", "\n") for c in classes], fontsize=9, color=INK)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Precision, recall and F1 per class, {split} split", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, ncols=3, loc="lower left")
    style(ax)
    save(fig, f"per_class_{split}.png")


def top_confusions(final, split="test", n=5):
    r = final[split]
    cm = np.array(r["confusion_matrix"])
    classes = r["classes"]
    pairs = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j]
    return [p for p in sorted(pairs, reverse=True)[:n] if p[0] > 0]


def mistakes(split="test", per_pair=6):
    """Grid of wrongly classified test images for the most frequent confusions."""
    preds = find(f"{split}_predictions.csv")
    if preds is None or not DATA.exists():
        return
    wrong = defaultdict(list)
    with open(preds) as fh:
        for row in csv.DictReader(fh):
            if row["true"] != row["pred"]:
                wrong[(row["true"], row["pred"])].append(row)
    pairs = sorted(wrong, key=lambda k: -len(wrong[k]))[:5]
    if not pairs:
        return
    fig, axes = plt.subplots(len(pairs), per_pair, figsize=(per_pair * 1.3, len(pairs) * 1.55), squeeze=False)
    for r, key in enumerate(pairs):
        rows = sorted(wrong[key], key=lambda x: -float(x["conf"]))[:per_pair]
        for k in range(per_pair):
            ax = axes[r, k]
            ax.set_xticks([]), ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if k < len(rows):
                ax.imshow(Image.open(DATA / rows[k]["file"]))
                ax.set_title(f"{100 * float(rows[k]['conf']):.0f}%", fontsize=7, color=MUTED)
        axes[r, 0].set_ylabel(f"true: {pretty(key[0])}\npred: {pretty(key[1])}\n({len(wrong[key])} images)",
                              rotation=0, ha="right", va="center", fontsize=8, color=INK)
    fig.suptitle(f"Most frequent mistakes on the {split} split (model confidence above each image)",
                 color=INK, fontsize=10)
    save(fig, f"mistakes_{split}.png")


def write_md(df, final):
    t, v, cfg = final["test"], final["val"], final["cfg"]
    rows = "\n".join(f"| {pretty(c)} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
                     for c, m in t["per_class"].items())
    conf = "\n".join(f"- **{pretty(a)}** predicted as **{pretty(b)}**: {n} test images "
                     f"({100 * n / t['per_class'][a]['support']:.1f}% of the {pretty(a)}s)"
                     for n, a, b in top_confusions(final))
    best = df["metrics/accuracy_top1"].idxmax()
    md = f"""# White-blood-cell classifier: report

`models/wbc_classifier.pt`, a `{cfg['model']}` (ImageNet-pretrained) fine-tuned on BloodMNIST
(7 classes: 6 WBC types plus erythroblast as "not a WBC"). Dataset report: [../wbc_dataset/REPORT.md](../wbc_dataset/REPORT.md).

## Results

| | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| **Test** (never seen during training) | **{t['accuracy']:.3f}** | {t['macro_precision']:.3f} | {t['macro_recall']:.3f} | {t['macro_f1']:.3f} |
| Validation | {v['accuracy']:.3f} | {v['macro_precision']:.3f} | {v['macro_recall']:.3f} | {v['macro_f1']:.3f} |

Accuracy is the share of images whose class is predicted correctly. "Macro" means the average over the 7
classes, so small classes count as much as big ones.

### Per class, test split

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
{rows}

- **Precision**: when the model says "monocyte", how often is it really a monocyte?
- **Recall**: of all real monocytes, how many does the model find?
- **F1**: a single score that combines the two.

![Per-class metrics](per_class_test.png)

### What it confuses most

{conf}

![Confusion matrix](confusion_matrix_test.png)

![Examples of mistakes](mistakes_test.png)

## Training

- Kaggle Tesla T4 GPU, image size {cfg['imgsz']} (the 64 px images are upscaled), batch {cfg['batch']},
  up to {cfg['epochs']} epochs with early-stopping patience {cfg['patience']}, cosine learning rate, vertical and
  horizontal flips.
- Ran {len(df)} epochs in {final['train_time_s'] / 60:.1f} min. Best validation accuracy
  {df['metrics/accuracy_top1'][best]:.3f} at epoch {int(df['epoch'][best])}. That epoch's weights are the saved model.

![Training curves](training_curves.png)
"""
    (OUT / "REPORT.md").write_text(md)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fm = find("final_metrics.json")
    if fm is None:
        raise SystemExit("final_metrics.json not found in runs/kaggle_wbc. Has the kernel finished?")
    final = json.loads(fm.read_text())
    (OUT / "final_metrics.json").write_text(json.dumps(final, indent=2))
    df = training_curves()
    for split in ("test", "val"):
        confusion(final, split)
        per_class(final, split)
    mistakes("test")
    # Ultralytics' own plots, for reference.
    ul = OUT / "ultralytics"
    ul.mkdir(exist_ok=True)
    for name in ("results.png", "train_batch0.jpg"):
        p = find(name)
        if p:
            shutil.copy(p, ul / name)
    for split in ("val", "test"):
        ev = next((p for p in SRC.rglob(f"eval_{split}") if p.is_dir()), None)
        for p in (ev.glob("confusion_matrix*.png") if ev else []):
            shutil.copy(p, ul / f"{split}_{p.name}")
    model = find("wbc_classifier.pt") or find("best.pt")
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(model, ROOT / "models" / "wbc_classifier.pt")
    write_md(df, final)
    print(f"Test accuracy {final['test']['accuracy']:.3f}. Reports written to reports/wbc_model/, "
          "model copied to models/wbc_classifier.pt")


if __name__ == "__main__":
    main()

"""Analyse the TXL-PBC dataset and write figures + a JSON summary to reports/dataset/.

Usage: .venv/bin/python scripts/dataset_report.py
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "TXL-PBC"
OUT = ROOT / "reports" / "dataset"
SPLITS = ["train", "val", "test"]
NAMES = yaml.safe_load((DATA / "data.yaml").read_text())["names"]
COLORS = ["#7c3aed", "#dc2626", "#0891b2"]  # WBC, RBC, Platelets


def load():
    rows = []  # one dict per image
    for split in SPLITS:
        for img in sorted((DATA / "images" / split).glob("*.png")):
            lbl = DATA / "labels" / split / f"{img.stem}.txt"
            with Image.open(img) as im:
                w, h, mode = im.width, im.height, im.mode
            boxes = []
            if lbl.exists():
                for line in lbl.read_text().split("\n"):
                    p = line.split()
                    if len(p) == 5:
                        boxes.append((int(p[0]), *map(float, p[1:])))
            rows.append(dict(split=split, path=img, w=w, h=h, mode=mode, boxes=boxes,
                             has_label=lbl.exists()))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()

    per_split = {}
    for s in SPLITS:
        r = [x for x in rows if x["split"] == s]
        c = Counter(b[0] for x in r for b in x["boxes"])
        per_split[s] = {"images": len(r), "boxes": sum(c.values()),
                        "per_class": {NAMES[k]: c.get(k, 0) for k in range(len(NAMES))},
                        "images_without_boxes": sum(1 for x in r if not x["boxes"]),
                        "missing_label_files": sum(1 for x in r if not x["has_label"])}

    sizes = Counter((x["w"], x["h"]) for x in rows)
    modes = Counter(x["mode"] for x in rows)
    bad = [str(x["path"].name) for x in rows for b in x["boxes"]
           if not all(0 <= v <= 1 for v in b[1:]) or b[0] >= len(NAMES)]

    summary = {
        "dataset": "TXL-PBC", "classes": NAMES, "splits": per_split,
        "total_images": len(rows), "total_boxes": sum(len(x["boxes"]) for x in rows),
        "image_sizes_top": [{"w": w, "h": h, "count": n} for (w, h), n in sizes.most_common(10)],
        "distinct_image_sizes": len(sizes), "color_modes": dict(modes),
        "invalid_annotations": len(bad),
        "boxes_per_image": {s: _stats([len(x["boxes"]) for x in rows if x["split"] == s]) for s in SPLITS},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # 1. class distribution per split
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.27
    for i, name in enumerate(NAMES):
        vals = [per_split[s]["per_class"][name] for s in SPLITS]
        bars = ax.bar([j + (i - 1) * width for j in range(3)], vals, width, label=name, color=COLORS[i])
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(range(3), SPLITS)
    ax.set_ylabel("annotated cells")
    ax.set_title("Cells per class and split")
    ax.legend(frameon=False)
    _save(fig, "class_distribution.png")

    # 2. boxes per image histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([len(x["boxes"]) for x in rows], bins=30, color="#475569")
    ax.set_xlabel("cells per image")
    ax.set_ylabel("images")
    ax.set_title("Cells per image (all splits)")
    _save(fig, "boxes_per_image.png")

    # 3. box size (normalised w x h) per class
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    by_cls = defaultdict(list)
    for x in rows:
        for b in x["boxes"]:
            by_cls[b[0]].append((b[3], b[4]))
    for k, ax in enumerate(axes):
        pts = by_cls[k]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=3, alpha=0.3, color=COLORS[k])
        ax.set_title(f"{NAMES[k]} (n={len(pts)})")
        ax.set_xlabel("box width (fraction)")
    axes[0].set_ylabel("box height (fraction)")
    fig.suptitle("Bounding-box sizes by class")
    _save(fig, "box_sizes.png")

    # 4. box centre heatmap
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    xs = [b[1] for x in rows for b in x["boxes"]]
    ys = [b[2] for x in rows for b in x["boxes"]]
    ax.hist2d(xs, ys, bins=40, cmap="Blues")
    ax.invert_yaxis()
    ax.set_title("Box centre positions")
    _save(fig, "box_centres.png")

    # 5. sample grid with ground-truth boxes
    random.seed(0)
    samples = random.sample([x for x in rows if x["split"] == "train"], 8)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    for x, ax in zip(samples, axes.flat):
        im = Image.open(x["path"]).convert("RGB")
        d = ImageDraw.Draw(im)
        W, H = im.size
        for c, cx, cy, bw, bh in x["boxes"]:
            d.rectangle([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H],
                        outline=COLORS[c], width=max(2, W // 250))
        ax.imshow(im)
        ax.set_title(f"{x['w']}x{x['h']} · {len(x['boxes'])} cells", fontsize=9)
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c, lw=3, label=n) for c, n in zip(COLORS, NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Training samples with ground-truth boxes")
    _save(fig, "samples.png")

    print(json.dumps({k: summary[k] for k in ("total_images", "total_boxes", "distinct_image_sizes",
                                               "color_modes", "invalid_annotations")}, indent=2))
    print(json.dumps(per_split, indent=2))


def _stats(v):
    v = sorted(v)
    return {"min": v[0], "median": v[len(v) // 2], "mean": round(sum(v) / len(v), 2), "max": v[-1]}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()

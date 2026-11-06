"""Check the full white-blood-cell pipeline (detector -> crop -> classifier) on TXL-PBC photos.

Most of the 360x363 TXL-PBC photos come from the same source as BloodMNIST (the PBC dataset): the centre of
the photo, cut to 200x200 and shrunk to 64x64, is almost pixel-identical to a BloodMNIST image. That
"twin" gives us the true white-blood-cell type of the photo. Only twins in BloodMNIST's val and test splits
are used (the classifier never trained on them): val twins choose the crop size, test twins report the result.

Outputs: reports/wbc_pipeline/ (summary.json, crop_scale.png, crops_vs_bloodmnist.png, examples.png)

Usage: .venv/bin/python scripts/wbc_pipeline_check.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import cells  # noqa: E402
import wbc  # noqa: E402
import app  # noqa: E402  (the web app's drawing function)

BM = ROOT / "data" / "bloodminst"
TXL = ROOT / "data" / "TXL-PBC" / "images"
OUT = ROOT / "reports" / "wbc_pipeline"
SCALES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def center64(im):
    """What BloodMNIST did to a 360x363 photo: centre 200x200 crop, shrunk to 64x64."""
    W, H = im.size
    l, t = (W - 200) // 2, (H - 200) // 2
    return im.crop((l, t, l + 200, t + 200)).resize((64, 64), Image.LANCZOS)


def find_twins():
    bm = sorted(BM.rglob("*.png"))
    B = np.stack([np.asarray(Image.open(p), dtype=np.float32).ravel() / 255 for p in bm])
    photos = []
    for split in ("train", "val", "test"):
        for p in sorted((TXL / split).glob("*.png")):
            with Image.open(p) as im:
                if im.size in ((360, 363), (366, 369)):
                    photos.append(p)
    T = np.stack([np.asarray(center64(Image.open(p).convert("RGB")), dtype=np.float32).ravel() / 255 for p in photos])
    d = np.sqrt(np.maximum((T ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * T @ B.T, 0) / B.shape[1])
    order = np.argsort(d, 1)
    twins = []
    for i, p in enumerate(photos):
        best, second = d[i, order[i, 0]], d[i, order[i, 1]]
        rel = bm[order[i, 0]].relative_to(BM)
        # a confident match: very close, and much closer than the next candidate
        if best < 0.03 and second > 2 * best and rel.parts[0] in ("val", "test"):
            twins.append({"photo": p, "twin": bm[order[i, 0]], "split": rel.parts[0], "label": rel.parts[1],
                          "rms": float(best)})
    return twins, len(photos)


def central_wbc(image, conf):
    """The detected white blood cell closest to the centre of the photo (PBC photos centre the cell)."""
    dets = [d for d in cells.detect_cells(image, conf) if d[0] == "WBC"]
    if not dets:
        return None
    cx, cy = image.width / 2, image.height / 2
    return min(dets, key=lambda d: ((d[2][0] + d[2][2]) / 2 - cx) ** 2 + ((d[2][1] + d[2][3]) / 2 - cy) ** 2)


def classify_direct(crops):
    """Variant without shrinking to 64 px first (the crop goes straight to the classifier)."""
    model = wbc.get_classifier()
    return [model.names[int(r.probs.top1)] for r in cells.predict(model, crops, imgsz=wbc.CLASSIFIER_IMGSZ)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    twins, n_photos = find_twins()
    print(f"{len(twins)} twins in val/test out of {n_photos} PBC-style photos:", Counter(t["split"] for t in twins))
    model = wbc.get_classifier()

    rows = []
    for t in twins:
        im = Image.open(t["photo"]).convert("RGB")
        ref = model.names[int(cells.predict(model, Image.open(t["twin"]).convert("RGB"), imgsz=wbc.CLASSIFIER_IMGSZ)[0].probs.top1)]
        det = central_wbc(im, cells.DEFAULT_CONF)
        row = {"split": t["split"], "label": t["label"], "photo": t["photo"].name, "twin": str(t["twin"].relative_to(BM)),
               "reference": ref, "detected": det is not None, "shrink64": {}, "direct": {}}
        if det:
            crops = [wbc.crop_cell(im, det[2], s) for s in SCALES]
            for s, (pred, _c) in zip(SCALES, wbc.classify(crops)):
                row["shrink64"][str(s)] = pred
            for s, pred in zip(SCALES, classify_direct(crops)):
                row["direct"][str(s)] = pred
            row["box"] = det[2]
        rows.append(row)

    def acc(split, mode=None, scale=None):
        rs = [r for r in rows if r["split"] == split]
        if mode is None:
            return float(np.mean([r["reference"] == r["label"] for r in rs]))
        return float(np.mean([r["detected"] and r[mode][str(scale)] == r["label"] for r in rs]))

    table = {split: {"n": sum(r["split"] == split for r in rows),
                     "detected": sum(r["detected"] for r in rows if r["split"] == split),
                     "reference_bloodmnist_twin": acc(split),
                     "shrink64": {str(s): acc(split, "shrink64", s) for s in SCALES},
                     "direct": {str(s): acc(split, "direct", s) for s in SCALES}}
             for split in ("val", "test")}
    best_scale = max(SCALES, key=lambda s: (table["val"]["shrink64"][str(s)], -abs(s - 1.4)))
    summary = {"twins": len(rows), "chosen_scale_on_val": best_scale, "results": table,
               "test_at_chosen_scale": table["test"]["shrink64"][str(best_scale)]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "twin_predictions.json").write_text(json.dumps(rows, indent=1, default=float))
    print(json.dumps(summary, indent=1))

    # Chart: pipeline accuracy against crop size, with the classifier's accuracy on the twins as reference.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        r = table[split]
        ax.plot(SCALES, [r["shrink64"][str(s)] for s in SCALES], "-o", color=BLUE, lw=2, ms=6, label="crop shrunk to 64 px (app)")
        ax.plot(SCALES, [r["direct"][str(s)] for s in SCALES], "-o", color=ORANGE, lw=2, ms=6, label="crop not shrunk")
        ax.axhline(r["reference_bloodmnist_twin"], color=MUTED, ls="--", lw=1, label="classifier on the BloodMNIST twin")
        ax.axvline(best_scale, color=GRID, lw=6, zorder=0)
        ax.set_title(f"{split} twins (n={r['n']})", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("crop side / detected box side", color=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
        ax.yaxis.grid(True, color=GRID)
    axes[0].set_ylabel("correct type", color=MUTED)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "crop_scale.png", dpi=120, facecolor="white")
    plt.close(fig)

    # Figure: BloodMNIST twin next to the app's crop of the same cell (test twins).
    pick = [r for r in rows if r["split"] == "test" and r["detected"]][:8]
    fig, axes = plt.subplots(2, len(pick), figsize=(len(pick) * 1.5, 3.6))
    for k, r in enumerate(pick):
        im = Image.open(TXL.parent / "images" / next(s for s in ("train", "val", "test")
                                                      if (TXL / s / r["photo"]).exists()) / r["photo"]).convert("RGB")
        crop = wbc.crop_cell(im, r["box"], best_scale).resize((64, 64), Image.BICUBIC)
        for row_i, (img, cap) in enumerate(((Image.open(BM / r["twin"]), r["label"].replace("_", " ")),
                                            (crop, "pred: " + r["shrink64"][str(best_scale)].replace("_", " ")))):
            ax = axes[row_i, k]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(cap, fontsize=7, color=INK if row_i == 0 or r["shrink64"][str(best_scale)] == r["label"] else "#c0392b")
    axes[0, 0].set_ylabel("BloodMNIST", fontsize=8, color=MUTED)
    axes[1, 0].set_ylabel("app crop", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "crops_vs_bloodmnist.png", dpi=120, facecolor="white")
    plt.close(fig)

    # Figure: the app's output on photos from each source in TXL-PBC (test split), at the chosen crop size.
    wbc.CROP_SCALE = best_scale
    by_size = {}
    for p in sorted((TXL / "test").glob("*.png")):
        with Image.open(p) as im:
            by_size.setdefault(im.size, []).append(p)
    examples = [p for size in ((575, 575), (360, 363), (640, 480), (256, 256)) for p in by_size.get(size, [])[:2]]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    for ax, p in zip(axes.ravel(), examples):
        im = Image.open(p).convert("RGB")
        dets = cells.detect_cells(im, cells.DEFAULT_CONF)
        found = wbc.classify_white_cells(im, [d for d in dets if d[0] == "WBC"])
        ax.imshow(app.draw_results(im, dets, found))
        ax.set_title(f"{p.name[:8]}  ({im.width}x{im.height})", fontsize=9, color=MUTED)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=110, facecolor="white")
    plt.close(fig)
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()

"""Inspect the BloodMNIST dataset (data/bloodminst/) and write a report to reports/wbc_dataset/.

Checks the folder structure, counts images per class and split, image sizes and colour modes,
class imbalance, exact duplicates across splits, and how much detail is lost at 64x64.

Usage: .venv/bin/python scripts/wbc_dataset_report.py
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bloodminst"
OUT = ROOT / "reports" / "wbc_dataset"
SPLITS = ["train", "val", "test"]
CLASSES = ["basophil", "eosinophil", "erythroblast", "immature_granulocytes",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]
REMOVED = {"platelet"}  # not used for training (see REPORT.md)
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}  # validated categorical slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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


def scan():
    """Check the structure and collect counts, sizes, modes and file hashes."""
    problems = []
    for split in SPLITS:
        found = sorted(p.name for p in (DATA / split).iterdir() if p.is_dir())
        if found != CLASSES:
            problems.append(f"{split}: expected {CLASSES}, found {found}")
    counts = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    sizes, modes, broken = Counter(), Counter(), []
    hashes = defaultdict(list)
    for split in SPLITS:
        for c in CLASSES:
            for p in sorted((DATA / split / c).glob("*")):
                if p.suffix.lower() != ".png":
                    problems.append(f"non-PNG file {p.relative_to(DATA)}")
                    continue
                try:
                    with Image.open(p) as im:
                        sizes[f"{im.width}x{im.height}"] += 1
                        modes[im.mode] += 1
                except OSError:
                    broken.append(str(p.relative_to(DATA)))
                    continue
                counts[split][c] += 1
                hashes[hashlib.md5(p.read_bytes()).hexdigest()].append((split, c, str(p.relative_to(DATA))))
    dups = [v for v in hashes.values() if len({s for s, _, _ in v}) > 1]
    return counts, sizes, modes, broken, problems, dups


def plot_distribution(counts):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(CLASSES))
    w = 0.26
    for i, s in enumerate(SPLITS):
        vals = [counts[s][c] for c in CLASSES]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=SPLIT_COLORS[s], label=s)
        ax.bar_label(bars, fontsize=7, color=MUTED, padding=2)
    ax.set_xticks(x, [c.replace("_", "\n") + ("\n(removed)" if c in REMOVED else "") for c in CLASSES],
                  color=INK, fontsize=9)
    ax.set_ylabel("images", color=MUTED)
    ax.set_title("BloodMNIST: images per class and split", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    style(ax)
    save(fig, "class_distribution.png")


def plot_share(counts):
    """Share of each class in the training split, for the 7 classes the model learns."""
    used = [c for c in CLASSES if c not in REMOVED]
    vals = np.array([counts["train"][c] for c in used])
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.barh([used[i] for i in order], vals[order], color=SPLIT_COLORS["train"], height=0.6)
    ax.bar_label(bars, labels=[f"{vals[i]} ({100 * vals[i] / vals.sum():.0f}%)" for i in order],
                 fontsize=8, color=MUTED, padding=3)
    ax.axvline(vals.mean(), color=MUTED, linestyle="--", linewidth=1, label="equal share for every class")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title(f"Training images per class (largest / smallest = {vals.max() / vals.min():.1f}x)",
                 loc="left", color=INK, fontsize=11)
    ax.set_xlim(0, vals.max() * 1.25)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    save(fig, "class_balance.png")


def plot_samples():
    random.seed(0)
    n = 8
    fig, axes = plt.subplots(len(CLASSES), n, figsize=(n * 1.1, len(CLASSES) * 1.2))
    for r, c in enumerate(CLASSES):
        files = sorted((DATA / "train" / c).glob("*.png"))
        for k, p in enumerate(random.sample(files, n)):
            ax = axes[r, k]
            ax.imshow(Image.open(p))
            ax.set_xticks([]), ax.set_yticks([])
            ax.spines[:].set_visible(False)
        axes[r, 0].set_ylabel(c.replace("_", "\n"), rotation=0, ha="right", va="center", fontsize=8, color=INK)
    fig.suptitle("Random training images (64x64 px)", color=INK, fontsize=11)
    save(fig, "samples.png")


def plot_resolution():
    """Same cell at 64x64 (BloodMNIST), 28x28 and at the resolution of a TXL-PBC photo."""
    txl = sorted((ROOT / "data" / "TXL-PBC" / "images" / "train").glob("*.png"))
    pick = None
    for p in txl:
        with Image.open(p) as im:
            if im.size != (360, 363):
                continue
        lbl = ROOT / "data" / "TXL-PBC" / "labels" / "train" / f"{p.stem}.txt"
        rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        if any(r[0] == "0" for r in rows):
            pick = p
            break
    if pick is None:
        return
    im = Image.open(pick).convert("RGB")
    W, H = im.size
    crop = im.crop(((W - 200) // 2, (H - 200) // 2, (W + 200) // 2, (H + 200) // 2))
    views = [("Original photo crop\n200x200 px", crop), ("BloodMNIST size\n64x64 px", crop.resize((64, 64), Image.LANCZOS)),
             ("Smallest MedMNIST size\n28x28 px", crop.resize((28, 28), Image.LANCZOS))]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, (t, v) in zip(axes, views):
        ax.imshow(v.resize((200, 200), Image.NEAREST))
        ax.set_title(t, fontsize=9, color=INK)
        ax.set_xticks([]), ax.set_yticks([])
    save(fig, "resolution.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts, sizes, modes, broken, problems, dups = scan()
    plot_distribution(counts)
    plot_share(counts)
    plot_samples()
    plot_resolution()
    used = [c for c in CLASSES if c not in REMOVED]
    train_used = [counts["train"][c] for c in used]
    summary = {
        "dataset": "BloodMNIST (MedMNIST v2), 64x64",
        "folder": str(DATA.relative_to(ROOT)),
        "structure_ok": not problems,
        "problems": problems,
        "classes": CLASSES,
        "classes_used_for_training": used,
        "splits": {s: {"images": sum(counts[s].values()), "per_class": counts[s]} for s in SPLITS},
        "total_images": sum(sum(v.values()) for v in counts.values()),
        "image_sizes": dict(sizes),
        "color_modes": dict(modes),
        "broken_files": broken,
        "imbalance_ratio_train_used_classes": round(max(train_used) / min(train_used), 2),
        "duplicates_across_splits": [[f for _, _, f in g] for g in dups],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("splits", "duplicates_across_splits")}, indent=1))
    print("duplicate groups across splits:", len(dups))


if __name__ == "__main__":
    main()

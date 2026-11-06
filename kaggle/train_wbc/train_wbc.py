"""Fine-tune YOLO26-cls on BloodMNIST to classify white-blood-cell types, on a Kaggle GPU.

Runs as a Kaggle script kernel. Classes (one folder each): basophil, eosinophil, erythroblast
(kept as a "not a WBC" class), immature_granulocytes, lymphocyte, monocyte, neutrophil.
Each epoch prints one `METRICS {...}` line to stdout, which scripts/monitor.py parses for the
dashboard. Everything the reports need is written to /kaggle/working/.
"""
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"], check=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CFG = dict(
    model="yolo26s-cls.pt",  # ImageNet-pretrained YOLO26 small classifier, fine-tuned below
    epochs=40,
    patience=10,
    imgsz=224,  # the 64x64 images are upscaled: pretrained features work much better at 224
    batch=64,
    optimizer="auto",
    cos_lr=True,
    flipud=0.5,  # a cell has no "up", so vertical flips are valid augmentation (fliplr is 0.5 by default)
    seed=0,
)

WORK = Path("/kaggle/working")


def find_root():
    """The dataset folder is the one that holds train/, val/ and test/ (mount path varies)."""
    for d in Path("/kaggle/input").rglob("train"):
        if d.is_dir() and (d.parent / "val").is_dir() and (d.parent / "test").is_dir():
            return d.parent
    raise SystemExit("Dataset with train/val/test folders not found under /kaggle/input")


SRC = find_root()
CLASSES = sorted(p.name for p in (SRC / "train").iterdir() if p.is_dir())


def emit(tag, payload):
    print(f"{tag} {json.dumps(payload, default=float)}", flush=True)


emit("START", {"cfg": CFG, "classes": CLASSES, "ultralytics": ultralytics.__version__,
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "time": time.time(),
               "images": {s: sum(1 for _ in (SRC / s).rglob("*.png")) for s in ("train", "val", "test")}})

model = YOLO(CFG["model"])
t0 = time.time()


def on_fit_epoch_end(trainer):
    m = {k: round(float(v), 5) for k, v in trainer.metrics.items()}
    m.update({k: round(float(v), 5) for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()})
    lr = {k: round(float(v), 7) for k, v in trainer.lr.items()}
    emit("METRICS", {"epoch": trainer.epoch + 1, "epochs": trainer.epochs, **m, **lr,
                     "elapsed_s": round(time.time() - t0, 1),
                     "gpu_mem_gb": round(torch.cuda.memory_reserved() / 1e9, 2)})


model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
model.train(data=str(SRC), project=str(WORK), name="run", exist_ok=True, device=0, workers=4,
            plots=True, **{k: v for k, v in CFG.items() if k != "model"})

run = WORK / "run"
best = YOLO(run / "weights" / "best.pt")
names = best.names  # {0: 'basophil', ...}, alphabetical folder order


def evaluate(split):
    """Predict every image of a split; return metrics and write <split>_predictions.csv."""
    files = sorted((SRC / split).rglob("*.png"))
    n = len(names)
    cm = np.zeros((n, n), dtype=int)  # rows = true class, columns = predicted class
    idx = {v: k for k, v in names.items()}
    with open(WORK / f"{split}_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "true", "pred", "conf"])
        for i in range(0, len(files), 256):
            chunk = files[i:i + 256]
            for f, r in zip(chunk, best.predict([str(p) for p in chunk], imgsz=CFG["imgsz"], device=0,
                                                 verbose=False)):
                t, p = idx[f.parent.name], int(r.probs.top1)
                cm[t, p] += 1
                w.writerow([f"{split}/{f.parent.name}/{f.name}", names[t], names[p], round(float(r.probs.top1conf), 4)])
    per_class = {}
    for k in range(n):
        tp = cm[k, k]
        prec = tp / cm[:, k].sum() if cm[:, k].sum() else 0.0
        rec = tp / cm[k].sum() if cm[k].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[names[k]] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[k].sum())}
    return {"split": split, "accuracy": float(np.trace(cm) / cm.sum()),
            "macro_precision": float(np.mean([v["precision"] for v in per_class.values()])),
            "macro_recall": float(np.mean([v["recall"] for v in per_class.values()])),
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "per_class": per_class, "confusion_matrix": cm.tolist(), "classes": [names[k] for k in range(n)]}


final = {"cfg": CFG, "train_time_s": round(time.time() - t0, 1)}
for split in ("val", "test"):
    final[split] = evaluate(split)
    # Ultralytics' own evaluation also draws the confusion-matrix plots.
    best.val(data=str(SRC), split=split, imgsz=CFG["imgsz"], device=0, project=str(WORK),
             name=f"eval_{split}", exist_ok=True, plots=True)

(WORK / "final_metrics.json").write_text(json.dumps(final, indent=2, default=float))
shutil.copy(run / "weights" / "best.pt", WORK / "wbc_classifier.pt")
emit("FINAL", {k: final[k] if k == "cfg" else (
    {kk: vv for kk, vv in final[k].items() if kk not in ("confusion_matrix",)} if isinstance(final[k], dict) else final[k])
    for k in final})
print("TRAINING_DONE", flush=True)

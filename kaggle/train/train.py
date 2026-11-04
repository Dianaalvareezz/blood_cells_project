"""Fine-tune YOLO26 on TXL-PBC (WBC / RBC / Platelets) on a Kaggle GPU.

Runs as a Kaggle script kernel. Each epoch prints one `METRICS {...}` line to stdout,
which scripts/monitor.py parses from the kernel log for live monitoring.
Everything the reports need is written to /kaggle/working/.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"], check=True)

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CFG = dict(
    model="yolo26s.pt",  # COCO-pretrained YOLO26 small, fine-tuned below
    epochs=100,
    patience=25,
    imgsz=640,
    batch=16,
    optimizer="auto",
    cos_lr=True,
    close_mosaic=10,
    seed=0,
)

WORK = Path("/kaggle/working")
SRC = next(Path("/kaggle/input").rglob("data.yaml")).parent

# Write a data.yaml with absolute paths (the input mount is read-only, so we
# point at it rather than copying).
data_yaml = WORK / "data.yaml"
data_yaml.write_text(
    f"path: {SRC}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    "nc: 3\nnames: ['WBC', 'RBC', 'Platelets']\n"
)


def emit(tag, payload):
    print(f"{tag} {json.dumps(payload, default=float)}", flush=True)


emit("START", {"cfg": CFG, "ultralytics": ultralytics.__version__, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0), "time": time.time()})

model = YOLO(CFG["model"])
t0 = time.time()


def on_fit_epoch_end(trainer):
    m = {k.replace("(B)", ""): round(float(v), 5) for k, v in trainer.metrics.items()}
    lr = {k: round(float(v), 7) for k, v in trainer.lr.items()}
    emit("METRICS", {"epoch": trainer.epoch + 1, "epochs": trainer.epochs, **m, **lr,
                     "elapsed_s": round(time.time() - t0, 1),
                     "gpu_mem_gb": round(torch.cuda.memory_reserved() / 1e9, 2)})


model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

model.train(data=str(data_yaml), project=str(WORK), name="run", exist_ok=True, device=0,
            workers=4, plots=True, **{k: v for k, v in CFG.items() if k != "model"})

run = WORK / "run"
best = YOLO(run / "weights" / "best.pt")


def summarise(metrics, split):
    names = metrics.names
    box = metrics.box
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        per_class[names[int(c)]] = {
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "mAP50": float(box.ap50[i]), "mAP50-95": float(box.ap[i]),
        }
    return {"split": split, "precision": float(box.mp), "recall": float(box.mr),
            "mAP50": float(box.map50), "mAP50-95": float(box.map), "per_class": per_class,
            "speed_ms": metrics.speed}


final = {"cfg": CFG, "train_time_s": round(time.time() - t0, 1)}
for split in ("val", "test"):
    m = best.val(data=str(data_yaml), split=split, device=0, project=str(WORK),
                 name=f"eval_{split}", exist_ok=True, plots=True)
    final[split] = summarise(m, split)

# Predictions on a few test images for the qualitative report.
pred_dir = WORK / "predictions"
test_imgs = sorted((SRC / "images" / "test").glob("*.png"))[:12]
best.predict([str(p) for p in test_imgs], device=0, save=True, project=str(WORK),
             name="predictions", exist_ok=True, conf=0.25, line_width=2, verbose=False)

(WORK / "final_metrics.json").write_text(json.dumps(final, indent=2, default=float))
shutil.copy(run / "weights" / "best.pt", WORK / "best.pt")
emit("FINAL", final)
print("TRAINING_DONE", flush=True)

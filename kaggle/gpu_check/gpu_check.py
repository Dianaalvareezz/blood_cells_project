"""Kaggle GPU sanity check: run before launching the real training job."""
import json, os, subprocess, sys, time

report = {"ok": False, "steps": {}}

def step(name, fn):
    try:
        report["steps"][name] = {"ok": True, "result": fn()}
    except Exception as e:  # record and continue so we see every failure
        report["steps"][name] = {"ok": False, "error": repr(e)}
    print(f"[{name}]", json.dumps(report["steps"][name], default=str)[:2000], flush=True)

step("nvidia_smi", lambda: subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True, check=True).stdout.strip())

def torch_check():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "matmul_10x_4096_s": round(time.time() - t, 3), "finite": bool(torch.isfinite(y).all())}
step("torch_cuda", torch_check)

def ultra():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"], check=True)
    import ultralytics
    from ultralytics import YOLO
    m = YOLO("yolo26n.pt")
    r = m.predict("https://ultralytics.com/images/bus.jpg", device=0, verbose=False)
    return {"ultralytics": ultralytics.__version__, "boxes_on_bus.jpg": len(r[0].boxes)}
step("ultralytics_yolo26", ultra)

def find_data():
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if "data.yaml" in files:
            hits.append(root)
    assert hits, "data.yaml not found under /kaggle/input"
    return hits
step("dataset", find_data)

report["ok"] = all(s["ok"] for s in report["steps"].values())
json.dump(report, open("/kaggle/working/gpu_check.json", "w"), indent=2, default=str)
print("GPU_CHECK_OK" if report["ok"] else "GPU_CHECK_FAILED", flush=True)

"""
Per-frame latency benchmark across 1000 synthetic frames.
Runs all 9 models sequentially and saves results to latency_results.csv.
"""

import time
import csv
import numpy as np
from ultralytics import YOLO

BASE = "/home/amr/Dev/atlasracing-perception/yolo_models"

MODELS = [
    # (label,                                                         path,                                                                    imgsz)
    ("YOLOv11n-640",  f"{BASE}/YOLOv11_640x640/yolo11n_640/weights/best.engine",   640),
    ("YOLOv11s-640",  f"{BASE}/YOLOv11_640x640/yolo11s_640/weights/best.engine",   640),
    ("YOLOv11m-640",  f"{BASE}/YOLOv11_640x640/yolo11m_640/weights/best.engine",   640),
    ("YOLOv26n-640",  f"{BASE}/YOLOv26_640x640/yolo26n_640/weights/best.engine",   640),
    ("YOLOv26s-640",  f"{BASE}/YOLOv26_640x640/yolo26s_640/weights/best.engine",   640),
    ("YOLOv26m-640",  f"{BASE}/YOLOv26_640x640/yolo26m_640/weights/best.engine",   640),
    ("YOLOv26n-1280", f"{BASE}/YOLOv26_1280x1280/yolo26n_1280/weights/best.engine", 1280),
    ("YOLOv26s-1280", f"{BASE}/YOLOv26_1280x1280/yolo26s_1280/weights/best.engine", 1280),
    ("YOLOv26m-1280", f"{BASE}/YOLOv26_1280x1280/yolo26m_1280/weights/best.engine", 1280),
]

N_WARMUP = 20
N_FRAMES = 500

results = []  # (model_label, frame_id, latency_ms)

for label, path, imgsz in MODELS:
    print(f"\n{'='*60}")
    print(f"Benchmarking: {label}  ({imgsz}x{imgsz})")
    print(f"{'='*60}")

    model = YOLO(path, task="detect")

    # Generate synthetic frames once per resolution
    dummy  = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    frames = [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
              for _ in range(N_FRAMES)]

    # Warm-up — critical for TensorRT JIT compilation
    print(f"  Warming up ({N_WARMUP} frames)...")
    for _ in range(N_WARMUP):
        model(dummy, verbose=False)

    # Timed inference
    print(f"  Running {N_FRAMES} timed frames...")
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        model(frame, verbose=False)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        results.append((label, i, latency_ms))

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{N_FRAMES} frames done")

    # Free GPU memory before next model
    del model

# Save to CSV
out_path = "/home/amr/Dev/atlasracing-perception/latency_results.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "frame_id", "latency_ms"])
    writer.writerows(results)

print(f"\nDone. Results saved to: {out_path}")
print(f"Total rows: {len(results)}")
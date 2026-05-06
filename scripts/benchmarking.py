from ultralytics.utils.benchmarks import benchmark
from ultralytics import YOLO

DATA = "/home/amr/Dev/atlasracing-perception/YOLO_DATASET/data.yaml"
BASE = "/home/amr/Dev/atlasracing-perception/yolo_models"
BENCHMARK_MODELS = [
    (f"{BASE}/YOLOv11_640x640/yolo11n_640/weights/best.pt",    640),
    (f"{BASE}/YOLOv11_640x640/yolo11s_640/weights/best.pt",    640),
    (f"{BASE}/YOLOv11_640x640/yolo11m_640/weights/best.pt",    640),
    (f"{BASE}/YOLOv26_640x640/yolo26n_640/weights/best.pt",    640),
    (f"{BASE}/YOLOv26_640x640/yolo26s_640/weights/best.pt",    640),
    (f"{BASE}/YOLOv26_640x640/yolo26m_640/weights/best.pt",    640),
    (f"{BASE}/YOLOv26_1280x1280/yolo26n_1280/weights/best.pt", 1280),
    (f"{BASE}/YOLOv26_1280x1280/yolo26s_1280/weights/best.pt", 1280),
    (f"{BASE}/YOLOv26_1280x1280/yolo26m_1280/weights/best.pt", 1280),
]
VAL_MODELS = [
    (f"{BASE}/YOLOv11_640x640/yolo11n_640/weights/best.engine",    640),
    (f"{BASE}/YOLOv11_640x640/yolo11s_640/weights/best.engine",    640),
    (f"{BASE}/YOLOv11_640x640/yolo11m_640/weights/best.engine",    640),
    (f"{BASE}/YOLOv26_640x640/yolo26n_640/weights/best.engine",    640),
    (f"{BASE}/YOLOv26_640x640/yolo26s_640/weights/best.engine",    640),
    (f"{BASE}/YOLOv26_640x640/yolo26m_640/weights/best.engine",    640),
    (f"{BASE}/YOLOv26_1280x1280/yolo26n_1280/weights/best.engine", 1280),
    (f"{BASE}/YOLOv26_1280x1280/yolo26s_1280/weights/best.engine", 1280),
    (f"{BASE}/YOLOv26_1280x1280/yolo26m_1280/weights/best.engine", 1280),
]
# Benchmarking
for path, imgsz in BENCHMARK_MODELS:
  # Might have to change gpu args for jetson
  benchmark(model=path, data=DATA, imgsz=imgsz, half=False, device=0, format="engine")

# PR Curves & More (Saves in runs/detect somewhere on your pc)
for path, imgsz in VAL_MODELS:
    model = YOLO(path)
    model.val(data=DATA, imgsz=imgsz, plots=True)
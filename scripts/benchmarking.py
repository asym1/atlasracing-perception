from ultralytics.utils.benchmarks import benchmark
from ultralytics import YOLO
# Benchmarking On 640x640
# YOLOv11
benchmark(model="/home/amr/Dev/atlasracing/yolo_models/YOLOv11_640x640/yolo11n_640/weights/best.pt", data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", imgsz=640, half=False, device=0, format="engine")
benchmark(model="/home/amr/Dev/atlasracing/yolo_models/YOLOv11_640x640/yolo11s_640/weights/best.pt", data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", imgsz=640, half=False, device=0, format="engine")
benchmark(model="/home/amr/Dev/atlasracing/yolo_models/YOLOv11_640x640/yolo11m_640/weights/best.pt", data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", imgsz=640, half=False, device=0, format="engine")

# # YOLOv26
benchmark(model="/home/amr/Dev/atlasracing/yolo_models/YOLOv26_640x640/yolo26n_640/weights/best.pt", data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", imgsz=640, half=False, device=0, format="engine")
benchmark(model="/home/amr/Dev/atlasracing/yolo_models/YOLOv26_640x640/yolo26s_640/weights/best.pt", data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", imgsz=640, half=False, device=0, format="engine")
benchmark(model="/home/amr/Dev/atlasracing/yolo_models/YOLOv26_640x640/yolo26m_640/weights/best.pt", data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", imgsz=640, half=False, device=0, format="engine")

# ## Benchmarking on 1280x1280
# YOLOv26
benchmark(model="/home/amr/Dev/atlasracing/yolo_models/YOLOv26_1280x1280/yolo26n_1280/weights/best.pt", data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", imgsz=1280, half=False, device=0, format="engine")
benchmark(model="/home/amr/Dev/atlasracing/yolo_models/YOLOv26_1280x1280/yolo26s_1280/weights/best.pt", data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", imgsz=1280, half=False, device=0, format="engine")
benchmark(model="/home/amr/Dev/atlasracing/yolo_models/YOLOv26_1280x1280/yolo26m_1280/weights/best.pt", data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", imgsz=1280, half=False, device=0, format="engine")

# PR CURVES
# YOLOv11
model = YOLO("/home/amr/Dev/atlasracing/yolo_models/YOLOv11_640x640/yolo11n_640/weights/best.engine")  # or .engine for TensorRT
results = model.val(data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", plots=True)

model = YOLO("/home/amr/Dev/atlasracing/yolo_models/YOLOv11_640x640/yolo11s_640/weights/best.engine")  # or .engine for TensorRT
results = model.val(data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", plots=True)

model = YOLO("/home/amr/Dev/atlasracing/yolo_models/YOLOv11_640x640/yolo11m_640/weights/best.engine")  # or .engine for TensorRT
results = model.val(data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", plots=True)

# YOLOv26 640x640
model = YOLO("/home/amr/Dev/atlasracing/yolo_models/YOLOv26_640x640/yolo26n_640/weights/best.engine")  # or .engine for TensorRT
results = model.val(data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", plots=True)

model = YOLO("/home/amr/Dev/atlasracing/yolo_models/YOLOv26_640x640/yolo26s_640/weights/best.engine")  # or .engine for TensorRT
results = model.val(data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", plots=True)

model = YOLO("/home/amr/Dev/atlasracing/yolo_models/YOLOv26_640x640/yolo26m_640/weights/best.engine")  # or .engine for TensorRT
results = model.val(data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", plots=True)

# YOLOv26 1280x1280
model = YOLO("/home/amr/Dev/atlasracing/yolo_models/YOLOv26_1280x1280/yolo26n_1280/weights/best.engine")  # or .engine for TensorRT
results = model.val(data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", plots=True)

model = YOLO("/home/amr/Dev/atlasracing/yolo_models/YOLOv26_1280x1280/yolo26s_1280/weights/best.engine")  # or .engine for TensorRT
results = model.val(data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", plots=True)

model = YOLO("/home/amr/Dev/atlasracing/yolo_models/YOLOv26_1280x1280/yolo26m_1280/weights/best.engine")  # or .engine for TensorRT
results = model.val(data="/home/amr/Dev/atlasracing/YOLO_DATASET/data.yaml", plots=True)
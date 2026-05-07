from ultralytics import YOLO
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent

model = YOLO("yolo26s.pt")  # load a pretrained model (recommended for training)

results = model.train(
  data=str(REPO_DIR / "FS_Cone" / "yolo_dataset" / "data.yaml"),
  epochs=600,
  imgsz=1280, # or 640
  rect=True,
  patience=50,
  batch=0.85,
  save=True,
  save_period=30,
  exist_ok=True,
  profile=True, # For tensorRT & ONNX

  project=str(SCRIPT_DIR / 'training'),
  name='yolo11m_1280',
  device=0
)

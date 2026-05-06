from ultralytics import YOLO
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

results = model.train(
  data="YOLO_DATASET/data.yaml",
  epochs=600,
  imgsz=640, # or 640
  rect=True,
  patience=50,
  batch=0.90,
  save=True,
  save_period=30,
  exist_ok=True,
  profile=True, # For tensorRT & ONNX

  project=os.path.join(SCRIPT_DIR, 'training'),
  name='yolo11n_640',
  device=0
)

model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

results = model.train(
  data="YOLO_DATASET/data.yaml",
  epochs=600,
  imgsz=640, # or 640
  rect=True,
  patience=50,
  batch=0.90,
  save=True,
  save_period=30,
  exist_ok=True,
  profile=True, # For tensorRT & ONNX

  project=os.path.join(SCRIPT_DIR, 'training'),
  name='yolo11s_640',
  device=0
)

model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)

results = model.train(
  data="YOLO_DATASET/data.yaml",
  epochs=600,
  imgsz=640, # or 640
  rect=True,
  patience=50,
  batch=0.90,
  save=True,
  save_period=30,
  exist_ok=True,
  profile=True, # For tensorRT & ONNX

  project=os.path.join(SCRIPT_DIR, 'training'),
  name='yolo11m_640',
  device=0
)

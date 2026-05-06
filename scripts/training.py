from ultralytics import YOLO
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Make sure your ultrulytics json file has the right paths (the location of the file is different per distro)
# place YOLO_DATASET (the folder outputted by create_structure.py) in the directory root
# make sure paths in data.yaml are good

model = YOLO("yolo26s.pt")  # load a pretrained model (recommended for training)

results = model.train(
  data="YOLO_DATASET/data.yaml",
  epochs=600,
  imgsz=1280, # or 640
  rect=True,
  patience=50,
  batch=0.85,
  save=True,
  save_period=30,
  exist_ok=True,
  profile=True, # For tensorRT & ONNX

  project=os.path.join(SCRIPT_DIR, 'training'),
  name='yolo11m_1280',
  device=0
)

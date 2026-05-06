import matplotlib.pyplot as plt
import numpy as np

models = [
    "YOLO11n",
    "YOLO11s",
    "YOLO11m",
    "YOLO26n",
    "YOLO26s",
    "YOLO26m",
    "YOLO26n-1280",
    "YOLO26s-1280",
    "YOLO26m-1280"
]

fps = [
    1186.97,  # YOLO11n TensorRT
    642.43,   # YOLO11s TensorRT
    318.4,     # YOLO11m TensorRT
    1045.77,  # YOLO26n TensorRT
    571.88,   # YOLO26s TensorRT
    308.07,   # YOLO26m TensorRT
    1049.18,  # YOLO26n 1280x1280 TensorRT
    572.93,   # YOLO26s 1280x1280 TensorRT
    311.07    # YOLO26m 1280x1280 TensorRT
]

colors = [
    "#4C78A8", "#4C78A8", "#4C78A8",
    "#F58518", "#F58518", "#F58518",
    "#54A24B", "#54A24B", "#54A24B"
]

x = np.arange(len(models))

plt.figure(figsize=(14, 6))
bars = plt.bar(x, fps, color=colors, edgecolor="black")

for bar, val in zip(bars, fps):
    label = f"{val:.2f}" if val > 0 else "N/A"
    y = val + 15 if val > 0 else 15
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        y,
        label,
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xticks(x, models, rotation=25, ha="right")
plt.ylabel("FPS")
plt.xlabel("Model")
plt.title("FPS of YOLO Models Averaged Across 1000 Frames")
plt.tight_layout()
plt.show()
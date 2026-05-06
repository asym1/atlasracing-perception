import matplotlib.pyplot as plt
import numpy as np

models  = ["YOLO11n","YOLO11s","YOLO11m",
           "YOLO26n","YOLO26s","YOLO26m",
           "YOLO26n-1280","YOLO26s-1280","YOLO26m-1280"]
latency = [0.84, 1.56, 3.14,
           0.96, 1.75, 3.24,
           0.95, 1.74, 3.21]
palette = ["#4C78A8"]*3 + ["#F58518"]*3 + ["#54A24B"]*3

x = np.arange(len(models))

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(x, latency, color=palette, edgecolor="black", linewidth=0.7, width=0.55)

# value labels above bars
for bar, val in zip(bars, latency):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.04,
            f"{val} ms",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, rotation=45, ha="right")
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Latency (ms/im)", fontsize=12)
ax.set_ylim(0, 3.75)
ax.set_title("TensorRT Latency — Averaged over 1000 Frames",
             fontsize=14, fontweight="bold", pad=12)

# no grid, clean spines
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# family separators (dotted vertical lines between groups)
for xpos in [2.5, 5.5]:
    ax.axvline(x=xpos, color="gray", linestyle=":", linewidth=1.3, alpha=0.55)

plt.tight_layout()
plt.savefig("latency_bar.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
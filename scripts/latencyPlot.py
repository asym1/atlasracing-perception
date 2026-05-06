import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CSV = "/home/amr/Dev/atlasracing-perception/latency_results.csv"
OUT = "/home/amr/Dev/atlasracing-perception/latency_plot.png"

ROLLING_WINDOW = 30

COLORS = {
    "YOLOv11n-640":  "#E63946",
    "YOLOv11s-640":  "#9D0208",
    "YOLOv11m-640":  "#F4722B",
    "YOLOv26n-640":  "#2DC653",
    "YOLOv26s-640":  "#B5A100",
    "YOLOv26m-640":  "#9B2FC9",
    "YOLOv26n-1280": "#74B3CE",
    "YOLOv26s-1280": "#1A6FA8",
    "YOLOv26m-1280": "#0B3D6B",
}

df = pd.read_csv(CSV)

# Extra right margin to fit external legend
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for model_label, group in df.groupby("model", sort=False):
    color = COLORS.get(model_label, "#888888")
    frames = group["frame_id"].values
    lats   = group["latency_ms"].values

    ax.plot(frames, lats, color=color, alpha=0.30, linewidth=0.7, zorder=1)

    rolling = pd.Series(lats).rolling(ROLLING_WINDOW, center=True, min_periods=1).mean()
    ax.plot(frames, rolling, color=color, linewidth=2.0, label=model_label, zorder=2)

ax.set_xlabel("Frames", fontsize=12, color="#222222", labelpad=8)
ax.set_ylabel("Latency (ms / frame)", fontsize=12, color="#222222", labelpad=8)
ax.set_title(
    "TensorRT End-to-end Inference Latency — 1 000 Frames",
    fontsize=13, fontweight="bold", color="#111111", pad=12
)
ax.tick_params(colors="#444444", labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor("#cccccc")
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(which="major", color="#eeeeee", linewidth=0.8)
ax.grid(which="minor", color="#f5f5f5", linewidth=0.4)
ax.set_xlim(0, 999)

# Clip y-axis at 12.5 — hides YOLOv26m-1280 upper spikes, zooms into lower models
ax.set_ylim(bottom=0, top=12.5)

handles, labels = ax.get_legend_handles_labels()

h640, l640, h1280, l1280 = [], [], [], []
for h, l in zip(handles, labels):
    if "1280" in l:
        h1280.append(h); l1280.append(l)
    else:
        h640.append(h); l640.append(l)

# Both legends stacked on the right, outside the axes
leg640 = ax.legend(
    h640, l640, title="640 Resolution",
    loc="upper left", bbox_to_anchor=(1.02, 1.0),
    fontsize=8.5, title_fontsize=9.5,
    framealpha=0.93, edgecolor="#cccccc", frameon=True
)
ax.add_artist(leg640)

leg1280 = ax.legend(
    h1280, l1280, title="1280 Resolution",
    loc="upper left", bbox_to_anchor=(1.02, 0.52),
    fontsize=8.5, title_fontsize=9.5,
    framealpha=0.93, edgecolor="#cccccc", frameon=True
)

# bbox_inches="tight" ensures the external legend is not clipped
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
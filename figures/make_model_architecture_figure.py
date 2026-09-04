import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "savefig.dpi": 300,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 9))
for ax in (ax1, ax2):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")

BOX_STYLE = dict(boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=1.2)
INPUT_COLOR = "#e8e8e8"
LAYER_COLOR = "#dbe9f6"
ATTN_COLOR = "#fde8d8"
OUT_COLOR = "#dff0db"


def add_box(ax, x, y, w, h, text, color, fontsize=9, fontweight="normal"):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          facecolor=color, edgecolor="black", zorder=2, **BOX_STYLE)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, wrap=True, zorder=3)


def add_arrow(ax, x1, y1, x2, y2, mutation_scale=12):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                              mutation_scale=mutation_scale, linewidth=1.1,
                              color="black", zorder=1, shrinkA=0, shrinkB=0)
    ax.add_patch(arrow)


# ============================================================
# Panel (a): Convolutional-Transformer
# ============================================================
add_box(ax1, 5, 14.2, 8.2, 0.9,
        "Input window: 193 steps (\u00b148 h)\nweather + temporal features + ET history + mask\n(target ET value excluded)",
        INPUT_COLOR, fontsize=8.5)

add_arrow(ax1, 5, 13.75, 5, 13.05)
add_box(ax1, 5, 12.6, 6.5, 0.85, "1D Convolution\n(kernel size = 3, 64 channels)", LAYER_COLOR)

add_arrow(ax1, 5, 12.15, 5, 11.45)
add_box(ax1, 5, 11.0, 6.5, 0.8, "Linear projection + sinusoidal\npositional encoding", LAYER_COLOR)

add_arrow(ax1, 5, 10.6, 5, 9.9)
add_box(ax1, 5, 9.45, 7.2, 0.95,
        "Transformer encoder layer 1\nmulti-head self-attention (4 heads)\n+ feedforward (non-causal)",
        ATTN_COLOR, fontsize=8.5)

add_arrow(ax1, 5, 8.97, 5, 8.27)
add_box(ax1, 5, 7.8, 7.2, 0.95,
        "Transformer encoder layer 2\nmulti-head self-attention (4 heads)\n+ feedforward (non-causal)",
        ATTN_COLOR, fontsize=8.5)

add_arrow(ax1, 5, 7.32, 5, 6.62)
add_box(ax1, 5, 6.15, 6.8, 0.8, "Extract representation at\ntarget (center) position", LAYER_COLOR)

add_arrow(ax1, 5, 5.75, 5, 5.05)
add_box(ax1, 5, 4.6, 6.0, 0.85, "Fully connected head\n(Linear \u2192 ReLU \u2192 Dropout \u2192 Linear)", LAYER_COLOR)

add_arrow(ax1, 5, 4.17, 5, 3.47)
add_box(ax1, 5, 3.0, 4.5, 0.75, "Predicted ET value", OUT_COLOR, fontweight="bold")

ax1.text(5, 15.0, "(a) Convolutional-Transformer (TC)", ha="center", fontsize=11.5, fontweight="bold")
ax1.text(5, 1.7,
         "Non-causal attention: every position attends to every\nother position, both before and after the target.",
         ha="center", fontsize=8, style="italic")

# ============================================================
# Panel (b): Bidirectional LSTM
# ============================================================
add_box(ax2, 5, 14.2, 8.2, 0.9,
        "Input window: 193 steps (\u00b148 h)\nweather + temporal features + ET history + mask\n(target ET value excluded)",
        INPUT_COLOR, fontsize=8.5)

add_arrow(ax2, 3.3, 13.75, 3.3, 13.05)
add_arrow(ax2, 6.7, 13.75, 6.7, 13.05)
add_box(ax2, 3.3, 12.6, 3.6, 0.85, "Forward LSTM\nlayer 1 (64 units)", LAYER_COLOR, fontsize=8.5)
add_box(ax2, 6.7, 12.6, 3.6, 0.85, "Backward LSTM\nlayer 1 (64 units)", LAYER_COLOR, fontsize=8.5)

add_arrow(ax2, 3.3, 12.15, 3.3, 11.45)
add_arrow(ax2, 6.7, 12.15, 6.7, 11.45)
add_box(ax2, 3.3, 11.0, 3.6, 0.85, "Forward LSTM\nlayer 2 (64 units)", LAYER_COLOR, fontsize=8.5)
add_box(ax2, 6.7, 11.0, 3.6, 0.85, "Backward LSTM\nlayer 2 (64 units)", LAYER_COLOR, fontsize=8.5)

add_arrow(ax2, 3.3, 10.55, 5, 9.9)
add_arrow(ax2, 6.7, 10.55, 5, 9.9)
add_box(ax2, 5, 9.45, 7.0, 0.85,
        "Concatenate forward + backward hidden\nstates at target (center) position",
        ATTN_COLOR, fontsize=8.5)

add_arrow(ax2, 5, 9.0, 5, 8.3)
add_box(ax2, 5, 7.85, 6.0, 0.7, "Dropout (rate = 0.2)", LAYER_COLOR)

add_arrow(ax2, 5, 7.47, 5, 6.77)
add_box(ax2, 5, 6.3, 6.0, 0.85, "Fully connected head\n(Linear \u2192 ReLU \u2192 Dropout \u2192 Linear)", LAYER_COLOR)

add_arrow(ax2, 5, 5.87, 5, 5.17)
add_box(ax2, 5, 4.7, 4.5, 0.75, "Predicted ET value", OUT_COLOR, fontweight="bold")

ax2.text(5, 15.0, "(b) Bidirectional LSTM", ha="center", fontsize=11.5, fontweight="bold")
ax2.text(5, 2.9,
         "Forward pass carries information up to the target;\nbackward pass carries information from beyond it \u2014\ntogether giving bidirectional context.",
         ha="center", fontsize=8, style="italic")

fig.suptitle("Figure 4. Model architectures for bidirectional ET gap-filling",
             fontsize=13, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig("figs/fig_model_architectures.png", bbox_inches="tight")
fig.savefig("figs/fig_model_architectures.pdf", bbox_inches="tight")
print("Saved fig_model_architectures.png / .pdf")

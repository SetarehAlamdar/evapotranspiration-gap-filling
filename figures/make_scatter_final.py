import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.grid": False,
})

GAP_COLORS = {
    "Isolated point": "#2ca02c",
    "Short (3h)": "#ff7f0e",
    "Long (24h)": "#d62728",
}

df = pd.read_csv("final_all_predictions.csv")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

model_labels = {"lstm": "LSTM", "conv_transformer": "Conv-Transformer"}
datasets = ["Lysimeter", "EC"]
models = ["lstm", "conv_transformer"]

for row, model in enumerate(models):
    for col, dataset in enumerate(datasets):
        ax = axes[row, col]
        sub = df[(df["model"] == model) & (df["dataset"] == dataset)]

        all_vals = np.concatenate([sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()])
        lims = [all_vals.min(), all_vals.max()]
        ax.plot(lims, lims, "k--", linewidth=1, zorder=1)

        for gap_type, color in GAP_COLORS.items():
            g = sub[sub["gap_type"] == gap_type]
            if len(g) == 0:
                continue
            # Subsample very large groups for plot clarity; metrics are
            # unaffected since they're computed elsewhere from full data.
            if len(g) > 8000:
                g = g.sample(8000, random_state=0)
            ax.scatter(g["y_true"], g["y_pred"], s=5, alpha=0.2, color=color,
                        edgecolors="none", label=gap_type, zorder=2)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed ET (mm/half-hour)")
        ax.set_ylabel("Predicted ET (mm/half-hour)")
        ax.set_title(f"{model_labels[model]} \u2013 {dataset}", fontweight="bold")

        if row == 0 and col == 0:
            leg = ax.legend(loc="upper left", frameon=False, markerscale=3)
            for lh in leg.legend_handles:
                lh.set_alpha(1)

fig.suptitle("Figure 10. Predicted vs. Observed ET Across Gap-Duration Categories",
              fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("figs/fig_scatter_final.png", bbox_inches="tight")
fig.savefig("figs/fig_scatter_final.pdf", bbox_inches="tight")
print("Saved fig_scatter_final.png / .pdf")

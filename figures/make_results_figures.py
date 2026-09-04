import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl

os.makedirs("figs", exist_ok=True)
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.grid": False,
})

COLORS = {
    "LSTM": "#1f77b4",
    "Conv-Transformer": "#d62728",
    "Linear Interpolation": "#7f7f7f",
    "MDS": "#2ca02c",
}

# ============================================================
# Data: master results table (final configuration, window=96 + augmentation)
# ============================================================
master = pd.DataFrame([
    # gap_type, dataset, method, MAE, RMSE, NSE
    ("Isolated point", "Lysimeter", "LSTM", 0.012449, 0.023232, 0.926325),
    ("Isolated point", "Lysimeter", "Conv-Transformer", 0.013183, 0.023558, 0.924242),
    ("Isolated point", "Lysimeter", "Linear Interpolation", 0.013357, 0.025802, 0.908168),
    ("Isolated point", "Lysimeter", "MDS", 0.023259, 0.037436, 0.806660),
    ("Isolated point", "EC", "LSTM", 0.010118, 0.017765, 0.963202),
    ("Isolated point", "EC", "Conv-Transformer", 0.011802, 0.018413, 0.960470),
    ("Isolated point", "EC", "Linear Interpolation", 0.011193, 0.019813, 0.954030),
    ("Isolated point", "EC", "MDS", 0.019687, 0.032591, 0.875579),

    ("Short (3h)", "Lysimeter", "LSTM", 0.019484, 0.033748, 0.852271),
    ("Short (3h)", "Lysimeter", "Conv-Transformer", 0.017949, 0.030838, 0.876649),
    ("Short (3h)", "Lysimeter", "Linear Interpolation", 0.020829, 0.035773, 0.832439),
    ("Short (3h)", "Lysimeter", "MDS", 0.024558, 0.039487, 0.795795),
    ("Short (3h)", "EC", "LSTM", 0.013099, 0.020106, 0.951110),
    ("Short (3h)", "EC", "Conv-Transformer", 0.012983, 0.019452, 0.954237),
    ("Short (3h)", "EC", "Linear Interpolation", 0.017849, 0.027951, 0.905066),
    ("Short (3h)", "EC", "MDS", 0.021188, 0.032599, 0.870862),

    ("Long (24h)", "Lysimeter", "LSTM", 0.019252, 0.031628, 0.875537),
    ("Long (24h)", "Lysimeter", "Conv-Transformer", 0.023817, 0.038153, 0.818882),
    ("Long (24h)", "Lysimeter", "Linear Interpolation", 0.071725, 0.109534, -0.508976),
    ("Long (24h)", "Lysimeter", "MDS", 0.025177, 0.040915, 0.789452),
    ("Long (24h)", "EC", "LSTM", 0.023587, 0.040797, 0.781572),
    ("Long (24h)", "EC", "Conv-Transformer", 0.016988, 0.026305, 0.909191),
    ("Long (24h)", "EC", "Linear Interpolation", 0.077015, 0.113713, -0.696944),
    ("Long (24h)", "EC", "MDS", 0.017865, 0.029072, 0.889086),
], columns=["gap_type", "dataset", "method", "MAE", "RMSE", "NSE"])

master.to_csv("figs/master_results_table.csv", index=False)

# ============================================================
# Figure 5: Ablation (window size + augmentation), long24h only
# ============================================================
ablation = pd.DataFrame([
    ("Lysimeter", "LSTM", "w=24", 0.0632),
    ("Lysimeter", "LSTM", "w=96", 0.0561),
    ("Lysimeter", "LSTM", "w=96+aug", 0.0316),
    ("Lysimeter", "Conv-Transformer", "w=24", 0.0617),
    ("Lysimeter", "Conv-Transformer", "w=96", 0.0504),
    ("Lysimeter", "Conv-Transformer", "w=96+aug", 0.0382),
    ("EC", "LSTM", "w=24", 0.0513),
    ("EC", "LSTM", "w=96", 0.0504),
    ("EC", "LSTM", "w=96+aug", 0.0408),
    ("EC", "Conv-Transformer", "w=24", 0.0560),
    ("EC", "Conv-Transformer", "w=96", 0.0336),
    ("EC", "Conv-Transformer", "w=96+aug", 0.0263),
], columns=["dataset", "model", "config", "RMSE"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
configs = ["w=24", "w=96", "w=96+aug"]
x = np.arange(len(configs))
width = 0.35

for ax, ds in zip(axes, ["Lysimeter", "EC"]):
    for i, model in enumerate(["LSTM", "Conv-Transformer"]):
        sub = ablation[(ablation["dataset"] == ds) & (ablation["model"] == model)]
        vals = [sub[sub["config"] == c]["RMSE"].values[0] for c in configs]
        ax.bar(x + (i - 0.5) * width, vals, width, label=model, color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_title(ds, fontweight="bold")
    ax.set_xlabel("Configuration")

axes[0].set_ylabel("RMSE (mm/half-hour)")
axes[0].legend(frameon=False)
fig.suptitle("Figure 5. Effect of Context Window Size and Context-Dropout\nAugmentation on Long (24-hour) Gap Performance",
              fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("figs/fig_ablation.png", bbox_inches="tight")
fig.savefig("figs/fig_ablation.pdf", bbox_inches="tight")
print("Saved fig_ablation.png / .pdf")
plt.close(fig)

# ============================================================
# Figure 6: Master comparison, ALL THREE metrics, grouped bars
# ============================================================
methods = ["LSTM", "Conv-Transformer", "Linear Interpolation", "MDS"]
gap_types = ["Isolated point", "Short (3h)", "Long (24h)"]
metrics_to_plot = ["MAE", "RMSE", "NSE"]

fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex="col")
x = np.arange(len(gap_types))
width = 0.2

for row, metric in enumerate(metrics_to_plot):
    for col, ds in enumerate(["Lysimeter", "EC"]):
        ax = axes[row, col]
        for i, method in enumerate(methods):
            vals = []
            for gt in gap_types:
                r = master[(master["dataset"] == ds) & (master["gap_type"] == gt) & (master["method"] == method)]
                vals.append(r[metric].values[0] if len(r) else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width, label=method, color=COLORS[method])
        if metric == "NSE":
            ax.axhline(0, color="black", linewidth=0.8)
        if row == 0:
            ax.set_title(ds, fontweight="bold")
        if col == 0:
            ylabel = metric if metric != "MAE" and metric != "RMSE" else f"{metric} (mm/half-hour)"
            ax.set_ylabel(ylabel)
        if row == 2:
            ax.set_xticks(x)
            ax.set_xticklabels(gap_types)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])

axes[0, 0].legend(loc="upper right", frameon=False, fontsize=8)
fig.suptitle("Figure 6. Model and Baseline Performance Across Gap-Duration Categories\n(MAE, RMSE, and NSE)",
              fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("figs/fig_master_comparison_all_metrics.png", bbox_inches="tight")
fig.savefig("figs/fig_master_comparison_all_metrics.pdf", bbox_inches="tight")
print("Saved fig_master_comparison_all_metrics.png / .pdf")
plt.close(fig)

# ============================================================
# Figure 7: Performance vs. gap duration trend, ALL THREE metrics
# ============================================================
ci_data = pd.read_csv("dl_bootstrap_ci.csv")
CI_MODEL_MAP = {"lstm": "LSTM", "conv_transformer": "Conv-Transformer"}
ci_data["model"] = ci_data["model"].map(CI_MODEL_MAP)

fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex="col")
gap_x = np.arange(len(gap_types))

for row, metric in enumerate(metrics_to_plot):
    for col, ds in enumerate(["Lysimeter", "EC"]):
        ax = axes[row, col]
        for method in methods:
            vals = []
            for gt in gap_types:
                r = master[(master["dataset"] == ds) & (master["gap_type"] == gt) & (master["method"] == method)]
                vals.append(r[metric].values[0] if len(r) else np.nan)
            marker = "o" if method in ["LSTM", "Conv-Transformer"] else "s"
            linestyle = "-" if method in ["LSTM", "Conv-Transformer"] else "--"
            ax.plot(gap_x, vals, marker=marker, linestyle=linestyle, linewidth=2,
                     markersize=6, label=method, color=COLORS[method], zorder=3)

            # Shaded 95% CI band, LSTM/Conv-Transformer only (baselines
            # have no bootstrap CI computed)
            if method in CI_MODEL_MAP.values():
                lo_vals, hi_vals = [], []
                for gt in gap_types:
                    r = ci_data[(ci_data["dataset"] == ds) & (ci_data["gap_type"] == gt) & (ci_data["model"] == method)]
                    lo_vals.append(r[f"{metric}_lo"].values[0] if len(r) else np.nan)
                    hi_vals.append(r[f"{metric}_hi"].values[0] if len(r) else np.nan)
                ax.fill_between(gap_x, lo_vals, hi_vals, color=COLORS[method], alpha=0.15, zorder=1)

        if metric == "NSE":
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        if row == 0:
            ax.set_title(ds, fontweight="bold")
        if col == 0:
            ylabel = metric if metric == "NSE" else f"{metric} (mm/half-hour)"
            ax.set_ylabel(ylabel)
        if row == 2:
            ax.set_xticks(gap_x)
            ax.set_xticklabels(gap_types)

axes[0, 0].legend(loc="best", frameon=False, fontsize=8)
fig.suptitle("Figure 7. Gap-Filling Performance vs. Gap Duration\n(MAE, RMSE, and NSE; shaded bands show 95% CI for LSTM/Conv-Transformer)",
              fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("figs/fig_performance_vs_gap_duration_all_metrics.png", bbox_inches="tight")
fig.savefig("figs/fig_performance_vs_gap_duration_all_metrics.pdf", bbox_inches="tight")
print("Saved fig_performance_vs_gap_duration_all_metrics.png / .pdf")
plt.close(fig)

print("\nAll multi-metric figures generated successfully.")

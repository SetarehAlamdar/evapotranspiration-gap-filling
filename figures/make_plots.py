"""
make_plots.py

Generates publication-quality figures comparing LSTM, Conv-Transformer,
and the two baselines (linear interpolation, MDS) on the held-out
synthetic gap evaluation set.

Reads:
  predictions_lstm_lysimeter.csv, predictions_conv_transformer_lysimeter.csv
  predictions_lstm_ec.csv, predictions_conv_transformer_ec.csv
      (from generate_predictions.py)
  baseline_predictions_lysimeter.csv, baseline_predictions_ec.csv
      (from run_baselines.py)

Produces:
  fig_scatter_lysimeter.png / fig_scatter_ec.png
      2x2 panel of predicted vs. true ET scatter density plots
      (LSTM, Conv-Transformer, Linear Interp, MDS), each with 1:1 line,
      R^2, RMSE, MAE annotated.
  fig_residuals_lysimeter.png / fig_residuals_ec.png
      Residual (prediction - true) vs. true value, to check for
      systematic bias (e.g. underprediction at high ET).
  fig_summary_comparison.png
      Grouped bar chart of MAE/RMSE/NRMSE across all methods and both
      datasets, for a single at-a-glance comparison figure.

All figures are saved at 300 DPI as both .png (for quick viewing) and
.pdf (vector, for journal submission).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------- Journal-style formatting ----------------
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


def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    nrmse = 100 * rmse / np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "NRMSE_pct": nrmse, "R2": r2, "n": len(y_true)}


def scatter_panel(ax, y_true, y_pred, title, color, max_points_plotted=20000):
    """Density-aware scatter: subsamples for plotting speed/clarity if very
    large, but metrics are always computed on the FULL dataset first."""
    metrics = compute_metrics(y_true, y_pred)

    if len(y_true) > max_points_plotted:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y_true), size=max_points_plotted, replace=False)
        y_true_plot, y_pred_plot = y_true[idx], y_pred[idx]
    else:
        y_true_plot, y_pred_plot = y_true, y_pred

    ax.scatter(y_true_plot, y_pred_plot, s=4, alpha=0.15, color=color, edgecolors="none")

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Observed ET (mm/half-hour)")
    ax.set_ylabel("Predicted ET (mm/half-hour)")
    ax.set_title(title, fontweight="bold")

    text = (f"R² = {metrics['R2']:.3f}\n"
            f"RMSE = {metrics['RMSE']:.4f}\n"
            f"MAE = {metrics['MAE']:.4f}\n"
            f"NRMSE = {metrics['NRMSE_pct']:.1f}%\n"
            f"n = {metrics['n']:,}")
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))

    ax.set_aspect("equal", adjustable="box")
    return metrics


def make_scatter_figure(dataset_label, model_dfs, baseline_df, output_prefix):
    """
    model_dfs: dict {"LSTM": df, "Conv-Transformer": df} each with
               columns y_true, y_pred
    baseline_df: df with columns ET_true, linear_interp_pred, mds_pred
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    scatter_panel(axes[0, 0], model_dfs["LSTM"]["y_true"].to_numpy(),
                  model_dfs["LSTM"]["y_pred"].to_numpy(),
                  "LSTM", COLORS["LSTM"])
    scatter_panel(axes[0, 1], model_dfs["Conv-Transformer"]["y_true"].to_numpy(),
                  model_dfs["Conv-Transformer"]["y_pred"].to_numpy(),
                  "Conv-Transformer", COLORS["Conv-Transformer"])

    interp_valid = baseline_df.dropna(subset=["linear_interp_pred"])
    scatter_panel(axes[1, 0], interp_valid["ET_true"].to_numpy(),
                  interp_valid["linear_interp_pred"].to_numpy(),
                  "Linear Interpolation", COLORS["Linear Interpolation"])

    mds_valid = baseline_df.dropna(subset=["mds_pred"])
    scatter_panel(axes[1, 1], mds_valid["ET_true"].to_numpy(),
                  mds_valid["mds_pred"].to_numpy(),
                  "MDS", COLORS["MDS"])

    fig.suptitle(f"{dataset_label}: Predicted vs. Observed ET (held-out synthetic gaps)",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    print(f"Saved {output_prefix}.png / .pdf")
    plt.close(fig)


def make_residual_figure(dataset_label, model_dfs, baseline_df, output_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)

    panels = [
        ("LSTM", model_dfs["LSTM"]["y_true"].to_numpy(), model_dfs["LSTM"]["y_pred"].to_numpy()),
        ("Conv-Transformer", model_dfs["Conv-Transformer"]["y_true"].to_numpy(),
         model_dfs["Conv-Transformer"]["y_pred"].to_numpy()),
    ]
    interp_valid = baseline_df.dropna(subset=["linear_interp_pred"])
    mds_valid = baseline_df.dropna(subset=["mds_pred"])
    panels += [
        ("Linear Interpolation", interp_valid["ET_true"].to_numpy(), interp_valid["linear_interp_pred"].to_numpy()),
        ("MDS", mds_valid["ET_true"].to_numpy(), mds_valid["mds_pred"].to_numpy()),
    ]

    for ax, (name, y_true, y_pred) in zip(axes.flat, panels):
        residual = y_pred - y_true
        ax.scatter(y_true, residual, s=4, alpha=0.15, color=COLORS[name], edgecolors="none")
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Observed ET (mm/half-hour)")
        ax.set_ylabel("Residual (Predicted − Observed)")

    fig.suptitle(f"{dataset_label}: Prediction Residuals vs. Observed ET",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    print(f"Saved {output_prefix}.png / .pdf")
    plt.close(fig)


def make_summary_bar_figure(all_metrics_df, output_prefix="fig_summary_comparison"):
    """
    all_metrics_df: DataFrame with columns [dataset, method, MAE, RMSE, NRMSE_pct]
    """
    metrics_to_plot = ["MAE", "RMSE", "NRMSE_pct"]
    datasets = all_metrics_df["dataset"].unique()

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 4.5))

    methods = ["LSTM", "Conv-Transformer", "Linear Interpolation", "MDS"]
    x = np.arange(len(datasets))
    width = 0.2

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        for i, method in enumerate(methods):
            vals = []
            for ds in datasets:
                row = all_metrics_df[(all_metrics_df["dataset"] == ds) & (all_metrics_df["method"] == method)]
                vals.append(row[metric].values[0] if len(row) else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width, label=method, color=COLORS[method])

        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ylabel = metric if metric != "NRMSE_pct" else "NRMSE (%)"
        ax.set_ylabel(ylabel)
        ax.set_title(metric, fontweight="bold")

    axes[0].legend(loc="upper center", bbox_to_anchor=(1.7, -0.15), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    print(f"Saved {output_prefix}.png / .pdf")
    plt.close(fig)


if __name__ == "__main__":
    all_metrics_rows = []

    for dataset_label, prefix in [("Lysimeter", "lysimeter"), ("EC", "ec")]:
        lstm_df = pd.read_csv(f"predictions_lstm_{prefix}.csv")
        ct_df = pd.read_csv(f"predictions_conv_transformer_{prefix}.csv")
        baseline_df = pd.read_csv(f"baseline_predictions_{prefix}.csv")

        model_dfs = {"LSTM": lstm_df, "Conv-Transformer": ct_df}

        make_scatter_figure(dataset_label, model_dfs, baseline_df,
                             output_prefix=f"fig_scatter_{prefix}")
        make_residual_figure(dataset_label, model_dfs, baseline_df,
                              output_prefix=f"fig_residuals_{prefix}")

        for name, df in model_dfs.items():
            m = compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
            all_metrics_rows.append({"dataset": dataset_label, "method": name, **m})

        interp_valid = baseline_df.dropna(subset=["linear_interp_pred"])
        m = compute_metrics(interp_valid["ET_true"].to_numpy(), interp_valid["linear_interp_pred"].to_numpy())
        all_metrics_rows.append({"dataset": dataset_label, "method": "Linear Interpolation", **m})

        mds_valid = baseline_df.dropna(subset=["mds_pred"])
        m = compute_metrics(mds_valid["ET_true"].to_numpy(), mds_valid["mds_pred"].to_numpy())
        all_metrics_rows.append({"dataset": dataset_label, "method": "MDS", **m})

    all_metrics_df = pd.DataFrame(all_metrics_rows)
    all_metrics_df.to_csv("all_metrics_summary.csv", index=False)
    print("\nSaved all_metrics_summary.csv:")
    print(all_metrics_df.to_string(index=False))

    make_summary_bar_figure(all_metrics_df)

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
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.grid": False,
})

df = pd.read_csv("timeseries_case_data_2023.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# Isolate a single contiguous gap block (same logic as the main reconstruction figure)
is_synth = df["is_synthetic_gap"].to_numpy()
run_id = (is_synth != np.roll(is_synth, 1)).cumsum()
run_id[0] = run_id[1] if len(run_id) > 1 else run_id[0]
run_lengths = pd.Series(is_synth).groupby(run_id).transform("sum")
target_run_length = run_lengths[is_synth].max()
candidate_run_ids = run_id[is_synth & (run_lengths == target_run_length)]
chosen_run_id = candidate_run_ids.iloc[0] if hasattr(candidate_run_ids, "iloc") else candidate_run_ids[0]

block_mask = (run_id == chosen_run_id) & is_synth
block_idx = df.index[block_mask]
gap_start, gap_end = block_idx.min(), block_idx.max()
print(f"Isolated gap block: {gap_end - gap_start + 1} steps "
      f"({df.loc[gap_start, 'datetime']} to {df.loc[gap_end, 'datetime']})")

pad = 96
plot_start = max(0, gap_start - pad)
plot_end = min(len(df) - 1, gap_end + pad)
window_df = df.iloc[plot_start:plot_end + 1]

fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

gap_dates = window_df.loc[window_df["is_synthetic_gap"], "datetime"]

panels = [
    (axes[0], "pred_lstm", "LSTM", "#1f77b4", "--"),
    (axes[1], "pred_conv_transformer", "Conv-Transformer", "#d62728", "--"),
    (axes[2], "linear_interp_pred", "Linear Interpolation", "#7f7f7f", ":"),
    (axes[3], "mds_pred", "MDS", "#2ca02c", ":"),
]

for ax, model_col, model_name, color, linestyle in panels:
    if len(gap_dates) > 0:
        ax.axvspan(gap_dates.min(), gap_dates.max(), color="gold", alpha=0.25, zorder=0)
    ax.plot(window_df["datetime"], window_df["ET_true"], color="black", linewidth=1.6,
             label="Observed ET", zorder=3)
    ax.plot(window_df["datetime"], window_df[model_col], color=color, linewidth=1.4,
             linestyle=linestyle, label=f"{model_name} prediction", zorder=2)
    ax.set_ylabel("ET (mm/half-hour)")
    ax.set_title(model_name, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)

axes[3].set_xlabel("Date")
fig.suptitle("Figure 11. Reconstruction of a Withheld 24-Hour Gap by All Four\n"
             "Methods, 2023 Holdout Year (EC Dataset)",
             fontsize=13, fontweight="bold")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("figs/fig_2023_timeseries_comparison.png", bbox_inches="tight")
fig.savefig("figs/fig_2023_timeseries_comparison.pdf", bbox_inches="tight")
print("Saved fig_2023_timeseries_comparison.png / .pdf")

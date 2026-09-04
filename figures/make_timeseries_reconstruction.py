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
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.grid": False,
})

df = pd.read_csv("timeseries_case_data.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# The season contains MANY separate synthetic gap blocks scattered
# throughout it (not just one). We need to isolate a single contiguous
# block to display, not span from the first block to the last block
# across the whole record.
is_synth = df["is_synthetic_gap"].to_numpy()
if not is_synth.any():
    raise ValueError("No synthetic gap found in this data -- check the case selection.")

run_id = (is_synth != np.roll(is_synth, 1)).cumsum()
run_id[0] = run_id[1] if len(run_id) > 1 else run_id[0]  # avoid wraparound artifact at index 0
run_lengths = pd.Series(is_synth).groupby(run_id).transform("sum")

# Pick the longest single contiguous run (should be ~48 steps = 24 hours)
synth_run_lengths = run_lengths[is_synth]
target_run_length = synth_run_lengths.max()
# Get the run_id corresponding to that longest run
candidate_run_ids = run_id[is_synth & (run_lengths == target_run_length)]
chosen_run_id = candidate_run_ids.iloc[0] if hasattr(candidate_run_ids, "iloc") else candidate_run_ids[0]

block_mask = (run_id == chosen_run_id) & is_synth
block_idx = df.index[block_mask]
gap_start, gap_end = block_idx.min(), block_idx.max()
print(f"Isolated a single gap block: {gap_end - gap_start + 1} steps "
      f"({df.loc[gap_start, 'datetime']} to {df.loc[gap_end, 'datetime']})")

pad = 96  # ~2 days of context on each side, at half-hourly resolution
plot_start = max(0, gap_start - pad)
plot_end = min(len(df) - 1, gap_end + pad)

window_df = df.iloc[plot_start:plot_end + 1]

fig, ax = plt.subplots(figsize=(11, 4.5))

# Shade the synthetic (withheld) gap region
gap_dates = window_df.loc[window_df["is_synthetic_gap"], "datetime"]
if len(gap_dates) > 0:
    ax.axvspan(gap_dates.min(), gap_dates.max(), color="gold", alpha=0.25,
                label="Withheld gap (24 h)", zorder=0)

# Observed ET (the true value is known for every row here, including the
# withheld gap, since this is held-out ground truth, not a real unknown gap)
ax.plot(window_df["datetime"], window_df["ET_true"], color="black", linewidth=1.6,
         label="Observed ET", zorder=3)

ax.plot(window_df["datetime"], window_df["pred_lstm"], color="#1f77b4", linewidth=1.4,
         linestyle="--", label="LSTM prediction", zorder=2)
ax.plot(window_df["datetime"], window_df["pred_conv_transformer"], color="#d62728", linewidth=1.4,
         linestyle="--", label="Conv-Transformer prediction", zorder=2)

ax.set_xlabel("Date")
ax.set_ylabel("ET (mm/half-hour)")
ax.set_title("Figure 9. Time-Series Reconstruction of a Withheld 24-Hour Gap\n(EC dataset)",
              fontweight="bold")
ax.legend(loc="upper right", frameon=False)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("figs/fig_timeseries_reconstruction.png", bbox_inches="tight")
fig.savefig("figs/fig_timeseries_reconstruction.pdf", bbox_inches="tight")
print("Saved fig_timeseries_reconstruction.png / .pdf")
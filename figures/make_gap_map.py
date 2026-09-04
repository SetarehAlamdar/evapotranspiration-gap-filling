import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 7,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.grid": False,
})

lys = pd.read_csv("combined_lysimeter_data.csv", parse_dates=["datetime"])
ec = pd.read_csv("combined_ec_data.csv", parse_dates=["datetime"])

YEARS = sorted(lys["season_year"].unique())
LYS_IDS = sorted(lys["lysimeter_id"].unique())
MGMTS = sorted(ec["management"].unique())

# Friendly display labels: sequential numbering preserves the original
# nest-based grouping order (H1* group first, then H2*, then H3*), while
# being much easier to read in a figure than the raw instrument codes.
LYS_LABELS = {lid: f"Lysimeter {i+1}" for i, lid in enumerate(LYS_IDS)}


def get_missing_runs(is_missing: np.ndarray, day_offsets: np.ndarray, step_days: float):
    """Returns a list of (start_day, duration_days) tuples for each
    consecutive run of missing values, using the actual day-offset of each
    point so runs are drawn at the correct real duration even if a few
    timestamps are irregular."""
    runs = []
    n = len(is_missing)
    i = 0
    while i < n:
        if is_missing[i]:
            j = i
            while j < n and is_missing[j]:
                j += 1
            start = day_offsets[i]
            end = day_offsets[j - 1] + step_days
            runs.append((start, end - start))
            i = j
        else:
            i += 1
    return runs


fig, axes = plt.subplots(len(YEARS), 1, figsize=(10, 2.0 * len(YEARS)), sharex=True)
if len(YEARS) == 1:
    axes = [axes]

for ax, year in zip(axes, YEARS):
    y_pos = 0
    yticks, yticklabels = [], []

    for lid in LYS_IDS:
        sub = lys[(lys["lysimeter_id"] == lid) & (lys["season_year"] == year)].sort_values("datetime")
        if len(sub) > 0:
            day0 = sub["datetime"].iloc[0]
            day_offsets = (sub["datetime"] - day0).dt.total_seconds().to_numpy() / 86400.0
            is_missing = sub["ET"].isna().to_numpy()
            runs = get_missing_runs(is_missing, day_offsets, step_days=0.5 / 24)
            ax.broken_barh(runs, (y_pos - 0.4, 0.8), facecolors="firebrick")
        yticks.append(y_pos)
        yticklabels.append(LYS_LABELS[lid])
        y_pos += 1

    for m in MGMTS:
        sub = ec[(ec["management"] == m) & (ec["season_year"] == year)].sort_values("datetime")
        if len(sub) > 0:
            day0 = sub["datetime"].iloc[0]
            day_offsets = (sub["datetime"] - day0).dt.total_seconds().to_numpy() / 86400.0
            is_missing = sub["ET"].isna().to_numpy()
            runs = get_missing_runs(is_missing, day_offsets, step_days=0.5 / 24)
            ax.broken_barh(runs, (y_pos - 0.4, 0.8), facecolors="steelblue")
        yticks.append(y_pos)
        yticklabels.append(f"EC-{m}")
        y_pos += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(y_pos - 0.5, -0.5)  # inverted so first unit is on top
    ax.set_title(str(year), loc="left", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 185)

axes[-1].set_xlabel("Days since May 1 (growing season)")

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color="firebrick", label="Lysimeter gap"),
    plt.Rectangle((0, 0), 1, 1, color="steelblue", label="EC gap"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

fig.suptitle("Gap Map: Timing and Duration of Missing ET Periods by Year",
             fontsize=13, fontweight="bold", y=1.05)
fig.tight_layout()
fig.savefig("fig_gap_map.png", bbox_inches="tight")
fig.savefig("fig_gap_map.pdf", bbox_inches="tight")
print("Saved fig_gap_map.png / .pdf")
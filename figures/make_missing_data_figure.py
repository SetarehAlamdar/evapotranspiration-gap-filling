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
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.grid": False,
})

# ---------------- Load real (unmasked) combined data ----------------
lys = pd.read_csv("combined_lysimeter_data.csv", parse_dates=["datetime"])
ec = pd.read_csv("combined_ec_data.csv", parse_dates=["datetime"])

# ---------------- Compute % missing ET by year ----------------
lys_missing = (
    lys.groupby(["lysimeter_id", "season_year"])["ET"]
    .apply(lambda s: 100 * s.isna().mean())
    .reset_index(name="pct_missing")
)
lys_summary = lys_missing.groupby("season_year")["pct_missing"].agg(["mean", "min", "max"])

ec_missing = (
    ec.groupby(["management", "season_year"])["ET"]
    .apply(lambda s: 100 * s.isna().mean())
    .reset_index(name="pct_missing")
)
ec_pivot = ec_missing.pivot(index="season_year", columns="management", values="pct_missing")

print("=== Lysimeter: mean/min/max % missing across 9 lysimeters, by year ===")
print(lys_summary.round(1))
print("\n=== EC: % missing by management and year ===")
print(ec_pivot.round(1))

# ---------------- Build the two-panel figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

years = ec_pivot.index.to_numpy()
x = np.arange(len(years))
width = 0.35

ax = axes[0]
managements = ec_pivot.columns.tolist()
colors_ec = {"Conventional": "#d62728", "Diversified": "#2ca02c"}
for i, mgmt in enumerate(managements):
    ax.bar(x + (i - 0.5) * width, ec_pivot[mgmt].to_numpy(), width,
           label=mgmt, color=colors_ec.get(mgmt, None))
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.set_xlabel("Year")
ax.set_ylabel("Missing ET (%)")
ax.set_title("(a) EC System", fontweight="bold")
ax.legend(frameon=False)
ax.set_ylim(0, 100)

ax = axes[1]
lys_years = lys_summary.index.to_numpy()
means = lys_summary["mean"].to_numpy()
mins = lys_summary["min"].to_numpy()
maxs = lys_summary["max"].to_numpy()
yerr = np.vstack([means - mins, maxs - means])
ax.bar(np.arange(len(lys_years)), means, color="#1f77b4",
       yerr=yerr, capsize=4, error_kw={"linewidth": 1})
ax.set_xticks(np.arange(len(lys_years)))
ax.set_xticklabels(lys_years)
ax.set_xlabel("Year")
ax.set_ylabel("Missing ET (%)")
ax.set_title("(b) Lysimeters (mean \u00b1 range across 9 units)", fontweight="bold")
ax.set_ylim(0, 100)

fig.suptitle("Percentage of Missing ET Values by Year", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("fig_missing_data_by_year.png", bbox_inches="tight")
fig.savefig("fig_missing_data_by_year.pdf", bbox_inches="tight")
print("\nSaved fig_missing_data_by_year.png / .pdf")

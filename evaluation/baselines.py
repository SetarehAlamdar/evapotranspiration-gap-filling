"""
baselines.py

Two standard gap-filling baselines, evaluated on the EXACT SAME held-out
synthetic gaps (is_synthetic_gap == True) as the LSTM/Conv-Transformer
models, using the exact same MAE/RMSE/NRMSE(mean-based) definitions, so
the comparison is fair and apples-to-apples.

1. Linear interpolation: the simplest possible baseline. Uses only the
   nearest real observed neighbors on either side (no tolerance/window
   logic).

2. Marginal Distribution Sampling (MDS), following Reichstein et al.
   (2005) -- the standard gap-filling method in eddy-covariance
   literature. For each gap, search an expanding time window (+/- 7, 14,
   21, ... days) for OTHER real observed points with similar net
   radiation (Rg), air temperature (Ta), and vapor pressure deficit (VPD)
   -- if enough similar points ("analogs") are found, average their ET.
   If not enough are found even at the widest window, relax the
   similarity requirements (drop VPD, then drop Ta) as a fallback, matching
   the original MDS strategy of progressively relaxing constraints rather
   than leaving a gap unfilled.

Both baselines are computed strictly from ET_input (which already has
both real gaps AND synthetic evaluation gaps set to NaN) as the pool of
usable analog/interpolation points -- never from ET_true -- so no
held-out evaluation value can leak into another gap's fill via either
method.
"""

import numpy as np
import pandas as pd


def compute_vpd(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """
    Vapor pressure deficit (kPa) via the Tetens formula, a standard,
    widely-used approximation of saturation vapor pressure.
    """
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))  # saturation vapor pressure, kPa
    ea = es * (rh_pct / 100.0)                                 # actual vapor pressure, kPa
    return es - ea


def linear_interpolation_baseline(
    df: pd.DataFrame,
    group_cols: list,
    et_input_col: str = "ET_input",
    datetime_col: str = "datetime",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adds a 'linear_interp_pred' column: linear interpolation using only
    ET_input (real gaps + synthetic gaps both NaN), computed independently
    within each group, with NO gap-length cap (unlike the earlier
    context-filling interpolation) -- this is meant to represent the
    simplest reasonable baseline someone might use, not our model's input.
    """
    df = df.copy()
    df["linear_interp_pred"] = np.nan

    for group_key, g in df.groupby(group_cols):
        g_sorted = g.sort_values(datetime_col)
        interp = g_sorted[et_input_col].interpolate(method="linear", limit_area="inside")
        df.loc[g_sorted.index, "linear_interp_pred"] = interp.values

    if verbose:
        n_eval_filled = df.loc[df.get("is_synthetic_gap", False), "linear_interp_pred"].notna().sum()
        print(f"Linear interpolation: filled {n_eval_filled} eval points "
              f"(some may remain NaN if too close to a group's edge).")

    return df


def mds_fill(
    df: pd.DataFrame,
    group_cols: list,
    et_input_col: str = "ET_input",
    is_synth_col: str = "is_synthetic_gap",
    datetime_col: str = "datetime",
    temp_col: str = "Temp",
    netrad_col: str = "NetRad",
    relhum_col: str = "RelHum",
    rg_tol: float = 50.0,       # W/m^2, standard MDS tolerance
    ta_tol: float = 2.5,        # deg C, standard MDS tolerance
    vpd_tol: float = 0.5,       # kPa (~5 hPa), standard MDS tolerance
    window_days_list=(7, 14, 21, 28, 35, 42, 49, 56),
    min_analogs: int = 2,
    steps_per_day: int = 48,    # half-hourly resolution
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adds an 'mds_pred' column using a simplified Marginal Distribution
    Sampling approach. Only applies MDS to rows where is_synth_col is True
    (i.e. only fills the held-out evaluation gaps -- this is an evaluation
    baseline, not meant to fill every real gap in the dataset).

    Strategy per gap (matching the tiered relaxation in the original MDS
    method):
      1. Try each window size in window_days_list, requiring similarity in
         Rg, Ta, AND VPD. Stop at the first window with >= min_analogs
         matching points.
      2. If no window (even the widest) found enough analogs under all
         three conditions, relax to Rg + Ta only (drop VPD) at the widest
         window.
      3. If still nothing, fall back to the mean of all real observed
         points in the widest window (no similarity filtering at all).
    """
    df = df.copy()
    df["VPD"] = compute_vpd(df[temp_col].to_numpy(), df[relhum_col].to_numpy())
    df["mds_pred"] = np.nan

    total_eval = 0
    filled_tier1 = 0
    filled_tier2 = 0
    filled_tier3 = 0

    for group_key, g in df.groupby(group_cols):
        g_sorted = g.sort_values(datetime_col)
        row_idx = g_sorted.index.to_numpy()
        n = len(g_sorted)

        et = g_sorted[et_input_col].to_numpy()
        rg = g_sorted[netrad_col].to_numpy()
        ta = g_sorted[temp_col].to_numpy()
        vpd = g_sorted["VPD"].to_numpy()
        is_synth = g_sorted[is_synth_col].to_numpy()

        candidate_base = ~np.isnan(et)  # only real observed points can be analogs
        eval_positions = np.where(is_synth)[0]
        total_eval += len(eval_positions)

        widest_window_steps = window_days_list[-1] * steps_per_day

        for pos in eval_positions:
            found = False

            for window_days in window_days_list:
                w = window_days * steps_per_day
                lo, hi = max(0, pos - w), min(n, pos + w + 1)
                local_pos = pos - lo

                cand = candidate_base[lo:hi].copy()
                if 0 <= local_pos < len(cand):
                    cand[local_pos] = False

                mask = (cand
                        & (np.abs(rg[lo:hi] - rg[pos]) <= rg_tol)
                        & (np.abs(ta[lo:hi] - ta[pos]) <= ta_tol)
                        & (np.abs(vpd[lo:hi] - vpd[pos]) <= vpd_tol))

                if mask.sum() >= min_analogs:
                    df.loc[row_idx[pos], "mds_pred"] = et[lo:hi][mask].mean()
                    filled_tier1 += 1
                    found = True
                    break

            if found:
                continue

            # Tier 2: relax to Rg + Ta only, widest window
            lo, hi = max(0, pos - widest_window_steps), min(n, pos + widest_window_steps + 1)
            local_pos = pos - lo
            cand = candidate_base[lo:hi].copy()
            if 0 <= local_pos < len(cand):
                cand[local_pos] = False
            mask = (cand
                    & (np.abs(rg[lo:hi] - rg[pos]) <= rg_tol)
                    & (np.abs(ta[lo:hi] - ta[pos]) <= ta_tol))
            if mask.sum() >= 1:
                df.loc[row_idx[pos], "mds_pred"] = et[lo:hi][mask].mean()
                filled_tier2 += 1
                continue

            # Tier 3: no similarity filtering, just mean of all real
            # observations in the widest window
            if cand.sum() > 0:
                df.loc[row_idx[pos], "mds_pred"] = et[lo:hi][cand].mean()
                filled_tier3 += 1

    if verbose:
        total_filled = filled_tier1 + filled_tier2 + filled_tier3
        print(f"MDS: {total_filled} / {total_eval} eval points filled "
              f"(tier1 Rg+Ta+VPD: {filled_tier1}, tier2 Rg+Ta: {filled_tier2}, "
              f"tier3 window-mean fallback: {filled_tier3})")

    return df


def evaluate_baseline(df: pd.DataFrame, pred_col: str, true_col: str = "ET_true",
                       is_synth_col: str = "is_synthetic_gap") -> dict:
    """Computes MAE/RMSE/NRMSE(mean-based)/NSE on the held-out eval rows
    only, using the exact same definitions as the model evaluation, for a
    fair side-by-side comparison."""
    eval_rows = df[is_synth_col] & df[pred_col].notna()
    y_true = df.loc[eval_rows, true_col].to_numpy()
    y_pred = df.loc[eval_rows, pred_col].to_numpy()

    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    nrmse = 100 * rmse / np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - ss_res / ss_tot

    return {"MAE": mae, "RMSE": rmse, "NRMSE_pct": nrmse, "NSE": nse, "n_samples": int(eval_rows.sum())}


if __name__ == "__main__":
    # Self-test on synthetic data with a known analog structure
    rng = np.random.default_rng(0)
    n = 500
    dt = pd.date_range("2018-05-01", periods=n, freq="30min")
    temp = 15 + 10 * np.sin(np.linspace(0, 20 * np.pi, n)) + rng.normal(0, 0.5, n)
    netrad = 200 + 400 * np.sin(np.linspace(0, 20 * np.pi, n)) + rng.normal(0, 10, n)
    relhum = 60 + 20 * np.cos(np.linspace(0, 20 * np.pi, n)) + rng.normal(0, 2, n)
    et_true = 0.05 + 0.02 * np.sin(np.linspace(0, 20 * np.pi, n)) + rng.normal(0, 0.005, n)
    et_true = np.clip(et_true, 0, None)

    test_df = pd.DataFrame({
        "lysimeter_id": ["H1L1"] * n, "season_year": [2018] * n, "datetime": dt,
        "Temp": temp, "NetRad": netrad, "RelHum": relhum,
        "ET_true": et_true, "ET_input": et_true.copy(),
    })

    synth_idx = rng.choice(n, size=40, replace=False)
    test_df["is_synthetic_gap"] = False
    test_df.loc[synth_idx, "is_synthetic_gap"] = True
    test_df.loc[synth_idx, "ET_input"] = np.nan

    test_df = linear_interpolation_baseline(test_df, group_cols=["lysimeter_id", "season_year"])
    test_df = mds_fill(test_df, group_cols=["lysimeter_id", "season_year"])

    interp_metrics = evaluate_baseline(test_df, "linear_interp_pred")
    mds_metrics = evaluate_baseline(test_df, "mds_pred")

    print("\nLinear interpolation:", interp_metrics)
    print("MDS:", mds_metrics)

    assert interp_metrics["n_samples"] > 0
    assert mds_metrics["n_samples"] > 0
    print("\nSelf-test passed: both baselines produced usable predictions.")

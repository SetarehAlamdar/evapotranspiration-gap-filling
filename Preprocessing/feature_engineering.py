"""
feature_engineering.py

Three responsibilities:

1. add_temporal_features(): adds cyclical (sin/cos) day-of-year, day-of-week,
   and month encodings, so the model sees smooth periodic time signals
   rather than raw integers with artificial jumps (e.g. day 365 -> day 1).

2. add_interpolated_et_context(): (optional, recommended) linearly
   interpolates ET_input within each group, to be used ONLY for filling
   missing CONTEXT points inside a window -- never the target/center point.
   Computed strictly from ET_input (which already has both real gaps AND
   synthetic evaluation gaps set to NaN), never from ET_true, so a
   held-out evaluation value can never leak into another window's context
   via interpolation. This gives the model a physically reasonable filled
   value instead of a flat 0 wherever context is missing.

3. build_windows(): builds bidirectional context windows around each
   candidate timestep, for gap-filling (not forecasting) style training.
   For a window half-size of W, each example uses W steps before and W
   steps after the target timestep as context.

   Two critical correctness rules enforced here:
   - The TARGET timestep's own ET value is NEVER included in the input
     window, even for training examples where the true value exists --
     otherwise the model would trivially see the answer.
   - Windows never cross a group boundary (lysimeter_id/management AND
     season_year), so no window blends two replicates or spans the winter
     gap between growing seasons.

   Missing ET values inside the surrounding context (not just at the
   center) are filled either with the interpolated value from
   add_interpolated_et_context() (if et_context_col is passed in), or with
   a placeholder (0) otherwise -- in both cases accompanied by a binary
   mask channel (1 = observed, 0 = missing/filled) so the model can always
   distinguish a real reading from a filled one, a standard approach in
   gap-filling models (e.g. GRU-D, BRITS).

Usage:
    from feature_engineering import (
        add_temporal_features, add_interpolated_et_context, build_windows,
    )

    df = add_temporal_features(df)
    df = add_interpolated_et_context(df, group_cols=["lysimeter_id", "season_year"])

    windows = build_windows(
        df,
        group_cols=["lysimeter_id", "season_year"],
        weather_cols=["Temp", "Precip", "RelHum", "WindSpd", "NetRad"],
        temporal_cols=["doy_sin", "doy_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"],
        window_half_size=24,  # 24 steps = 12 hours each side at half-hourly resolution
        et_context_col="ET_context_filled",  # omit this argument to fall back to zero-fill
    )
"""

import numpy as np
import pandas as pd


def fill_weather_gaps(
    df: pd.DataFrame,
    group_cols: list,
    weather_cols: list,
    datetime_col: str = "datetime",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fills missing values in weather columns (Temp, Precip, RelHum, WindSpd,
    NetRad, etc.) within each group. Unlike ET, weather is always treated
    as a known input (never the prediction target), so we don't need a
    mask channel here -- we just need every weather value to be non-NaN
    before it reaches the model.

    Strategy per group: linear interpolation first (weather changes
    smoothly, so this is a reasonable fill), then forward-fill/back-fill
    for any remaining edge gaps (a run of missing values at the very start
    or end of a group, where interpolation has no neighbor on one side).
    This guarantees zero NaN remain in these columns afterward.
    """
    df = df.copy()
    n_before = df[weather_cols].isna().sum().sum()

    filled_frames = []
    for group_key, g in df.groupby(group_cols):
        g_sorted = g.sort_values(datetime_col).copy()
        g_sorted[weather_cols] = g_sorted[weather_cols].interpolate(method="linear", limit_area="inside")
        g_sorted[weather_cols] = g_sorted[weather_cols].ffill().bfill()
        filled_frames.append(g_sorted)

    df = pd.concat(filled_frames).sort_index()
    n_after = df[weather_cols].isna().sum().sum()

    if verbose:
        print(f"fill_weather_gaps: {n_before} missing weather values before, "
              f"{n_after} remaining after (should be 0).")
        if n_after > 0:
            print(f"  WARNING: {n_after} weather values still NaN -- this would mean "
                  f"an entire group had no real weather data at all for that column.")

    return df


def add_temporal_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """
    Adds cyclical sin/cos encodings for day-of-year, day-of-week, and month.
    """
    df = df.copy()
    dt = df[datetime_col]

    doy = dt.dt.dayofyear
    doy_max = 366.0  # accounts for leap years consistently
    df["doy_sin"] = np.sin(2 * np.pi * doy / doy_max)
    df["doy_cos"] = np.cos(2 * np.pi * doy / doy_max)

    dow = dt.dt.dayofweek  # 0=Monday .. 6=Sunday
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    month = dt.dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return df


TEMPORAL_FEATURE_COLS = ["doy_sin", "doy_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]


def add_interpolated_et_context(
    df: pd.DataFrame,
    group_cols: list,
    et_input_col: str = "ET_input",
    datetime_col: str = "datetime",
    new_col: str = "ET_context_filled",
    max_gap_steps: int = 12,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adds a linearly-interpolated version of ET, computed ONLY from
    et_input_col (which already has both real gaps AND synthetic gaps set
    to NaN). This is used purely to fill CONTEXT points inside a window
    (never the target/center point, and never derived from ET_true) --
    using ET_true here would leak held-out evaluation values into other
    windows' context.

    Interpolation is done independently within each group (never crossing
    lysimeter/management or season_year boundaries).

    max_gap_steps caps how long a consecutive run of missing values can be
    before we give up on interpolating it. ET has a strong daily cycle, so
    a straight-line interpolation across a long gap (e.g. multiple days)
    would draw a smooth, physically wrong signal that hides the diurnal
    pattern entirely -- arguably worse than being honest that we have no
    information there. So:
      - Runs of missing values with length <= max_gap_steps ARE interpolated
      - Runs longer than max_gap_steps are left as NaN (falls back to
        zero-fill with mask=0 in build_windows, same as leading/trailing
        edge gaps with no interior neighbor)

    Default of 12 steps = 6 hours at half-hourly resolution.
    """
    df = df.copy()
    df[new_col] = np.nan

    n_short_gap_filled = 0
    n_long_gap_left = 0
    n_edge_left = 0

    for group_key, g in df.groupby(group_cols):
        g_sorted = g.sort_values(datetime_col)
        s = g_sorted[et_input_col]
        is_na = s.isna()

        # Full interpolation (no length limit) -- our candidate fill values.
        # limit_area='inside' means leading/trailing gaps (no real neighbor
        # on one side) are never extrapolated and stay NaN here regardless.
        interp_full = s.interpolate(method="linear", limit_area="inside")

        # Identify consecutive NaN runs and their lengths.
        run_id = (is_na != is_na.shift()).cumsum()
        run_length = is_na.groupby(run_id).transform("sum")  # 0 for non-NaN runs

        eligible = is_na & (run_length <= max_gap_steps) & interp_full.notna()
        too_long = is_na & (run_length > max_gap_steps)
        edge_case = is_na & (~eligible) & (~too_long)  # inside limit but interp_full still NaN (edge)

        filled = s.copy()
        filled[eligible] = interp_full[eligible]

        df.loc[g_sorted.index, new_col] = filled.values

        n_short_gap_filled += eligible.sum()
        n_long_gap_left += too_long.sum()
        n_edge_left += edge_case.sum()

    if verbose:
        n_total_missing = df[et_input_col].isna().sum()
        print(f"Interpolation (max_gap_steps={max_gap_steps}, "
              f"= {max_gap_steps * 0.5:.1f} hours at half-hourly resolution):")
        print(f"  Short gaps filled:        {n_short_gap_filled} / {n_total_missing}")
        print(f"  Long gaps left as NaN:    {n_long_gap_left} / {n_total_missing} "
              f"(run length > {max_gap_steps} steps -- will zero-fill with mask=0)")
        print(f"  Edge cases left as NaN:   {n_edge_left} / {n_total_missing} "
              f"(no interior neighbor on one side -- will zero-fill with mask=0)")

    return df


def build_windows(
    df: pd.DataFrame,
    group_cols: list,
    weather_cols: list,
    temporal_cols: list = None,
    window_half_size: int = 24,
    et_input_col: str = "ET_input",
    et_context_col: str = None,
    et_true_col: str = "ET_true",
    is_synth_col: str = "is_synthetic_gap",
    datetime_col: str = "datetime",
    verbose: bool = True,
):
    """
    Builds bidirectional context windows for every valid candidate center
    timestep within each group.

    et_context_col: if provided (e.g. "ET_context_filled" from
    add_interpolated_et_context), context points use this interpolated
    value instead of raw zero-fill wherever available; any still-missing
    points (edge cases with no interior neighbor) fall back to 0. The mask
    channel (X_et_mask) always reflects TRUE observed status from
    et_input_col, regardless of which fill method is used, so the model
    always knows which context points are real vs. filled.
    If et_context_col is None (default), behaves exactly as before:
    raw et_input_col values, missing points zero-filled.

    Returns a dict of numpy arrays:
      X_weather   : (N, 2*W+1, n_weather_features)  -- weather/temporal features
                    for the full window, including the center step (weather
                    is always allowed to be seen at the center; only ET is hidden)
      X_et        : (N, 2*W, 1)  -- ET history for the window EXCLUDING the
                    center step (W steps before + W steps after), missing
                    values filled per et_context_col (or 0 if not provided)
      X_et_mask   : (N, 2*W, 1)  -- 1 = observed, 0 = missing, aligned with X_et
      y           : (N,)  -- true ET value at the center step
      sample_type : (N,)  array of strings: "train", "eval", or "real_gap"
      meta        : DataFrame with group_cols + datetime for each sample,
                    for traceability back to the original data

    "real_gap" samples have no usable y  (it's NaN) -- these are the actual
    production gaps you'd fill in the final deliverable, but they cannot be
    used for training or evaluation since there's no ground truth.
    """
    if temporal_cols is None:
        temporal_cols = TEMPORAL_FEATURE_COLS

    feature_cols = weather_cols + temporal_cols
    W = window_half_size

    X_weather_list, X_et_list, X_et_mask_list, y_list, type_list, meta_list = [], [], [], [], [], []

    n_groups = 0
    n_skipped_edge = 0

    for group_key, g in df.groupby(group_cols):
        g = g.sort_values(datetime_col).reset_index(drop=True)
        n = len(g)
        n_groups += 1

        if n < (2 * W + 1):
            n_skipped_edge += n  # whole group too short to build even one window
            continue

        feat_matrix = g[feature_cols].to_numpy(dtype=float)
        et_input = g[et_input_col].to_numpy(dtype=float)
        # Values used to FILL missing context points: interpolated if
        # provided, otherwise same as raw et_input (will be zero-filled below)
        et_fill_source = (g[et_context_col].to_numpy(dtype=float)
                          if et_context_col is not None else et_input)
        et_true = g[et_true_col].to_numpy(dtype=float)
        is_synth = g[is_synth_col].to_numpy(dtype=bool)
        dt_vals = g[datetime_col].to_numpy()

        # Valid centers: need W steps of history available on both sides
        for center in range(W, n - W):
            window_feat = feat_matrix[center - W: center + W + 1]  # includes center

            et_window_idx = list(range(center - W, center)) + list(range(center + 1, center + W + 1))
            et_context_raw = et_input[et_window_idx]  # true observed status comes from here
            et_context_fill = et_fill_source[et_window_idx]  # values to actually feed the model
            et_mask = (~np.isnan(et_context_raw)).astype(float)
            # Wherever the fill source is itself still NaN (e.g. interpolation
            # edge case with no interior neighbor), fall back to 0.
            et_context_filled = np.where(np.isnan(et_context_fill), 0.0, et_context_fill)

            true_val = et_true[center]
            if is_synth[center]:
                s_type = "eval"
            elif not np.isnan(true_val):
                s_type = "train"
            else:
                s_type = "real_gap"

            X_weather_list.append(window_feat)
            X_et_list.append(et_context_filled.reshape(-1, 1))
            X_et_mask_list.append(et_mask.reshape(-1, 1))
            y_list.append(true_val)
            type_list.append(s_type)

            meta_row = dict(zip(group_cols, group_key)) if isinstance(group_key, tuple) \
                else {group_cols[0]: group_key}
            meta_row[datetime_col] = dt_vals[center]
            meta_list.append(meta_row)

    X_weather = np.stack(X_weather_list) if X_weather_list else np.empty((0, 2 * W + 1, len(feature_cols)))
    X_et = np.stack(X_et_list) if X_et_list else np.empty((0, 2 * W, 1))
    X_et_mask = np.stack(X_et_mask_list) if X_et_mask_list else np.empty((0, 2 * W, 1))
    y = np.array(y_list, dtype=float)
    sample_type = np.array(type_list, dtype=object)
    meta = pd.DataFrame(meta_list)

    if verbose:
        print(f"Groups processed: {n_groups}")
        if n_skipped_edge > 0:
            print(f"WARNING: {n_skipped_edge} rows belonged to groups too short "
                  f"for even one window (need >= {2*W+1} rows); those groups were skipped entirely.")
        print(f"\nTotal windows built: {len(y)}")
        for t in ["train", "eval", "real_gap"]:
            n_t = (sample_type == t).sum()
            print(f"  {t}: {n_t} ({100*n_t/len(y):.1f}%)" if len(y) else f"  {t}: 0")
        print(f"\nShapes: X_weather={X_weather.shape}, X_et={X_et.shape}, "
              f"X_et_mask={X_et_mask.shape}, y={y.shape}")

    return {
        "X_weather": X_weather,
        "X_et": X_et,
        "X_et_mask": X_et_mask,
        "y": y,
        "sample_type": sample_type,
        "meta": meta,
    }


def add_management_encoding(df: pd.DataFrame, management_col: str = "management",
                             positive_label: str = "Conventional",
                             new_col: str = "is_conventional") -> pd.DataFrame:
    """
    Encodes the EC 'management' column as a binary feature (1 = positive_label,
    0 = other), so the model can distinguish Conventional vs Diversified
    windows. Include this column name in the weather_cols list passed to
    build_windows() for the EC dataset (not needed for lysimeter, which has
    no management/replicate-identity feature by design).
    """
    df = df.copy()
    df[new_col] = (df[management_col] == positive_label).astype(float)
    return df


if __name__ == "__main__":
    # Self-test on a small synthetic group
    rng = np.random.default_rng(0)
    n = 200
    dt = pd.date_range("2018-05-01", periods=n, freq="30min")
    test_df = pd.DataFrame({
        "lysimeter_id": ["H1L1"] * n,
        "season_year": [2018] * n,
        "datetime": dt,
        "Temp": rng.normal(20, 5, n),
        "Precip": np.zeros(n),
        "RelHum": rng.normal(60, 10, n),
        "WindSpd": rng.normal(10, 3, n),
        "NetRad": rng.normal(100, 50, n),
        "ET_true": rng.normal(0.05, 0.02, n),
    })
    test_df["ET_input"] = test_df["ET_true"]
    test_df["is_synthetic_gap"] = False

    # Simulate some real missing + some synthetic gaps
    real_missing = rng.choice(n, size=15, replace=False)
    test_df.loc[real_missing, ["ET_input", "ET_true"]] = np.nan

    # Add one deliberately LONG consecutive gap (20 steps = 10 hours,
    # longer than the default max_gap_steps=12) to verify the cap works
    long_gap_start = 100
    test_df.loc[long_gap_start:long_gap_start + 19, "ET_input"] = np.nan
    test_df.loc[long_gap_start:long_gap_start + 19, "ET_true"] = np.nan

    # And one deliberately SHORT gap (5 steps) to verify it DOES get filled
    short_gap_start = 50
    test_df.loc[short_gap_start:short_gap_start + 4, "ET_input"] = np.nan
    test_df.loc[short_gap_start:short_gap_start + 4, "ET_true"] = np.nan

    remaining_observed = test_df.index[test_df["ET_true"].notna()]
    synth_gaps = rng.choice(remaining_observed, size=20, replace=False)
    test_df.loc[synth_gaps, "ET_input"] = np.nan
    test_df.loc[synth_gaps, "is_synthetic_gap"] = True

    test_df = add_temporal_features(test_df)
    test_df = add_interpolated_et_context(
        test_df, group_cols=["lysimeter_id", "season_year"], max_gap_steps=12,
    )

    # Verify the cap worked directly on the dataframe before windowing:
    short_gap_filled = test_df.loc[short_gap_start:short_gap_start + 4, "ET_context_filled"]
    long_gap_filled = test_df.loc[long_gap_start:long_gap_start + 19, "ET_context_filled"]
    assert short_gap_filled.notna().all(), "Short gap (5 steps) should have been fully interpolated!"
    assert long_gap_filled.isna().all(), "Long gap (20 steps) should have been left as NaN (exceeds cap)!"
    print("\nGap-length cap verified: short gap filled, long gap correctly left for zero-fill.")

    result = build_windows(
        test_df,
        group_cols=["lysimeter_id", "season_year"],
        weather_cols=["Temp", "Precip", "RelHum", "WindSpd", "NetRad"],
        window_half_size=10,
        et_context_col="ET_context_filled",
    )

    # Sanity checks
    assert result["X_weather"].shape[1] == 21  # 2*10+1
    assert result["X_et"].shape[1] == 20        # 2*10, center excluded
    assert not np.isnan(result["y"][result["sample_type"] == "train"]).any()
    assert not np.isnan(result["y"][result["sample_type"] == "eval"]).any()
    assert np.isnan(result["y"][result["sample_type"] == "real_gap"]).all()
    print("\nAll sanity checks passed.")

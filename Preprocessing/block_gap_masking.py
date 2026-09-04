"""
block_gap_masking.py

Creates synthetic BLOCK gaps (contiguous runs of missing values), as
opposed to gap_masking.py's isolated single-point masking. This tests a
harder, more realistic scenario: real sensor outages and EC data gaps
often come in multi-hour or multi-day runs, not scattered single points.

Only selects blocks that sit entirely within a run of REAL, actually
observed ET values (never overlapping an already-missing stretch), so
every masked position still has real ground truth to evaluate against --
same honesty principle as the point-masking approach.

Run this on the CLEANED, COMBINED data (e.g. combined_lysimeter.csv),
NOT the already-point-masked file -- this is an independent experiment,
not a modification of the point-masking results.

Usage:
    from block_gap_masking import add_block_synthetic_gaps

    masked = add_block_synthetic_gaps(
        combined_lysimeter_df,
        group_cols=["lysimeter_id", "season_year"],
        value_col="ET",
        block_length_steps=6,     # e.g. 6 steps = 3 hours at half-hourly res
        mask_fraction=0.15,
        random_seed=42,
    )
"""

import numpy as np
import pandas as pd


def _find_eligible_block_starts(observed: np.ndarray, block_length: int) -> np.ndarray:
    """
    Given a boolean array (True = observed), returns all starting indices
    where a block of `block_length` consecutive observed values exists
    (i.e. observed[start:start+block_length] is all True).
    """
    n = len(observed)
    if block_length > n:
        return np.array([], dtype=int)

    # Rolling-window "all True" check via cumulative sum trick
    cum = np.concatenate([[0], np.cumsum(observed.astype(int))])
    window_sums = cum[block_length:] - cum[:-block_length]
    starts = np.where(window_sums == block_length)[0]
    return starts


def add_block_synthetic_gaps(
    df: pd.DataFrame,
    group_cols: list,
    value_col: str = "ET",
    block_length_steps: int = 6,
    mask_fraction: float = 0.15,
    random_seed: int = 42,
    datetime_col: str = "datetime",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adds block-style synthetic gaps, analogous to gap_masking.add_synthetic_gaps
    but masking contiguous runs of `block_length_steps` instead of isolated
    points. Selected blocks never overlap each other or any pre-existing
    real gap.

    Adds the same three columns as the point-masking version:
      - f"{value_col}_true": original value (unchanged)
      - f"{value_col}_input": NaN wherever a synthetic block gap was placed
      - "is_synthetic_gap": True only for the artificially hidden rows

    mask_fraction is applied per-group, same interpretation as before:
    approximately this fraction of each group's OBSERVED points end up
    inside a masked block (the exact fraction may be slightly lower if a
    group doesn't have enough eligible non-overlapping block positions).
    """
    df = df.copy()
    df[f"{value_col}_true"] = df[value_col]
    df[f"{value_col}_input"] = df[value_col]
    df["is_synthetic_gap"] = False

    rng = np.random.default_rng(random_seed)
    summary_rows = []

    for group_key, g in df.groupby(group_cols):
        g_sorted = g.sort_values(datetime_col)
        idx = g_sorted.index.to_numpy()
        n = len(g_sorted)
        observed = g_sorted[value_col].notna().to_numpy()
        n_observed = observed.sum()

        target_masked = int(round(n_observed * mask_fraction))
        n_blocks_target = max(1, target_masked // block_length_steps)

        eligible_starts = _find_eligible_block_starts(observed, block_length_steps)
        rng.shuffle(eligible_starts)

        selected_starts = []
        occupied = np.zeros(n, dtype=bool)

        for start in eligible_starts:
            if len(selected_starts) >= n_blocks_target:
                break
            end = start + block_length_steps
            if occupied[start:end].any():
                continue  # overlaps a previously selected block
            selected_starts.append(start)
            occupied[start:end] = True

        n_masked = len(selected_starts) * block_length_steps
        for start in selected_starts:
            end = start + block_length_steps
            block_idx = idx[start:end]
            df.loc[block_idx, f"{value_col}_input"] = np.nan
            df.loc[block_idx, "is_synthetic_gap"] = True

        summary_rows.append({
            **(dict(zip(group_cols, group_key)) if isinstance(group_key, tuple)
               else {group_cols[0]: group_key}),
            "n_observed": n_observed,
            "n_blocks_placed": len(selected_starts),
            "n_masked": n_masked,
            "pct_masked": round(100 * n_masked / n_observed, 1) if n_observed else 0,
        })

    if verbose:
        summary_df = pd.DataFrame(summary_rows)
        print(f"Block synthetic gap masking (block_length_steps={block_length_steps}, "
              f"= {block_length_steps * 0.5:.1f} hours, mask_fraction={mask_fraction}):")
        print(summary_df.to_string(index=False))
        total_observed = df[value_col].notna().sum()
        total_masked = df["is_synthetic_gap"].sum()
        print(f"\nTotal observed points: {total_observed}")
        print(f"Total synthetic block-gap points created: {total_masked} "
              f"({100 * total_masked / total_observed:.1f}% of observed points)")
        print(f"Real missing (untouched): {df[value_col].isna().sum()}")

    return df


if __name__ == "__main__":
    # Self-test: verify blocks are contiguous, non-overlapping, and never
    # touch already-missing data
    rng = np.random.default_rng(0)
    n = 500
    test_df = pd.DataFrame({
        "lysimeter_id": ["H1L1"] * n,
        "season_year": [2018] * n,
        "datetime": pd.date_range("2018-05-01", periods=n, freq="30min"),
        "ET": rng.normal(0.05, 0.02, size=n),
    })
    # Simulate some real missing data (10% of rows)
    real_missing_idx = rng.choice(n, size=50, replace=False)
    test_df.loc[real_missing_idx, "ET"] = np.nan

    masked = add_block_synthetic_gaps(
        test_df, group_cols=["lysimeter_id", "season_year"],
        value_col="ET", block_length_steps=6, mask_fraction=0.15, random_seed=1,
    )

    # Check: every masked position must have had a REAL true value (never
    # overlapping a pre-existing real gap)
    masked_rows = masked[masked["is_synthetic_gap"]]
    assert masked_rows["ET_true"].notna().all(), "A synthetic gap overlapped a real missing value!"

    # Check: masked positions come in contiguous runs of exactly
    # block_length_steps (verify via run-length check)
    is_synth = masked["is_synthetic_gap"].to_numpy()
    run_id = (is_synth != np.roll(is_synth, 1)).cumsum()
    run_lengths = pd.Series(is_synth).groupby(run_id).transform("sum")
    synth_run_lengths = run_lengths[is_synth].unique()
    assert set(synth_run_lengths) <= {6}, f"Found block runs of unexpected length: {synth_run_lengths}"

    print(f"\nAll sanity checks passed: {masked_rows.shape[0]} points masked in "
          f"contiguous blocks of exactly 6 steps, none overlapping real gaps.")

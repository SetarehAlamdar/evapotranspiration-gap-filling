"""
gap_masking.py

Creates synthetic (MCAR) gaps in ET data for honest gap-filling evaluation.

Core principle: we NEVER evaluate a model against a value that was already
missing in the real data (that's what the imputation would be guessing at
anyway -- there's no ground truth to check against). Instead, we take rows
where ET WAS actually observed, hide a fraction of them, and keep the real
value aside so we can score the model's reconstruction against real data.

Masking is done independently within each (lysimeter_id/management,
season_year) group, so:
  - We never "borrow" observed points across different replicates/systems
  - We never treat a data point from one growing season as informing a
    masking decision in another season

This is the MCAR (missing completely at random) version: single, isolated
half-hourly points are hidden at random, independent of gap length or
timing. (A future extension can add block-gap masking -- short/long
consecutive runs -- to study the gap-length sensitivity noted in the
project's abstract.)

Usage:
    from gap_masking import add_synthetic_gaps

    masked = add_synthetic_gaps(
        combined_lysimeter_df,
        group_cols=["lysimeter_id", "season_year"],
        value_col="ET",
        mask_fraction=0.15,
        random_seed=42,
    )

    # masked now has:
    #   ET_true          -- original observed value (NaN where real data was
    #                        already missing)
    #   ET_input          -- what the model is allowed to see: real gaps AND
    #                        synthetic gaps are both NaN here
    #   is_synthetic_gap  -- True only for the artificially hidden points;
    #                        this is what you evaluate MAE/RMSE against
"""

import numpy as np
import pandas as pd


def add_synthetic_gaps(
    df: pd.DataFrame,
    group_cols: list,
    value_col: str = "ET",
    mask_fraction: float = 0.15,
    random_seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adds synthetic MCAR gaps to `value_col`, computed independently within
    each group defined by `group_cols`.

    Returns a copy of df with three new/changed columns:
      - f"{value_col}_true": the original value (unchanged)
      - f"{value_col}_input": same as original, except synthetic-gap rows
        are set to NaN (this is what a model should train/predict on as input)
      - "is_synthetic_gap": boolean, True only for the artificially hidden rows

    Real missing values (already NaN in `value_col`) are left untouched and
    are never selected for masking, and are not flagged as synthetic gaps.
    """
    if not (0 < mask_fraction < 1):
        raise ValueError(f"mask_fraction must be between 0 and 1, got {mask_fraction}")

    df = df.copy()
    df[f"{value_col}_true"] = df[value_col]
    df[f"{value_col}_input"] = df[value_col]
    df["is_synthetic_gap"] = False

    rng = np.random.default_rng(random_seed)

    summary_rows = []
    for group_key, group_idx in df.groupby(group_cols).groups.items():
        group_idx = pd.Index(group_idx)
        observed_idx = group_idx[df.loc[group_idx, value_col].notna()]
        n_observed = len(observed_idx)
        n_to_mask = int(round(n_observed * mask_fraction))

        if n_observed == 0:
            continue

        chosen = rng.choice(observed_idx.to_numpy(), size=n_to_mask, replace=False)
        df.loc[chosen, f"{value_col}_input"] = np.nan
        df.loc[chosen, "is_synthetic_gap"] = True

        summary_rows.append({
            **(dict(zip(group_cols, group_key)) if isinstance(group_key, tuple)
               else {group_cols[0]: group_key}),
            "n_observed": n_observed,
            "n_masked": n_to_mask,
            "pct_masked": round(100 * n_to_mask / n_observed, 1) if n_observed else 0,
        })

    if verbose:
        summary_df = pd.DataFrame(summary_rows)
        print(f"Synthetic gap masking summary (mask_fraction={mask_fraction}):")
        print(summary_df.to_string(index=False))
        total_observed = df[value_col].notna().sum()
        total_masked = df["is_synthetic_gap"].sum()
        print(f"\nTotal observed points: {total_observed}")
        print(f"Total synthetic gaps created: {total_masked} "
              f"({100 * total_masked / total_observed:.1f}% of observed points)")
        print(f"Real missing (untouched): {df[value_col].isna().sum()}")

    return df


def train_test_masks(df: pd.DataFrame, value_col: str = "ET"):
    """
    Convenience helper: returns boolean masks for the three relevant
    row categories after add_synthetic_gaps has been applied.

    - train_mask: rows the model can train on (real observed, not held out)
    - eval_mask: rows to evaluate MAE/RMSE against (synthetic gaps only)
    - real_gap_mask: rows that were genuinely missing in the raw data
      (not usable for training OR evaluation, but the model still needs to
      predict something for these in the final gap-filled product)
    """
    is_synth = df["is_synthetic_gap"]
    was_observed = df[f"{value_col}_true"].notna()

    train_mask = was_observed & (~is_synth)
    eval_mask = is_synth
    real_gap_mask = (~was_observed)

    return train_mask, eval_mask, real_gap_mask


if __name__ == "__main__":
    # Self-test on a small synthetic dataframe
    rng = np.random.default_rng(0)
    n = 200
    test_df = pd.DataFrame({
        "lysimeter_id": ["H1L1"] * n,
        "season_year": [2018] * n,
        "datetime": pd.date_range("2018-05-01", periods=n, freq="30min"),
        "ET": rng.normal(0.05, 0.02, size=n),
    })
    # Simulate some real missing data (10% of rows)
    real_missing_idx = rng.choice(test_df.index, size=20, replace=False)
    test_df.loc[real_missing_idx, "ET"] = np.nan

    masked = add_synthetic_gaps(
        test_df, group_cols=["lysimeter_id", "season_year"],
        value_col="ET", mask_fraction=0.15, random_seed=1,
    )

    train_mask, eval_mask, real_gap_mask = train_test_masks(masked, value_col="ET")
    print(f"\nTrain rows: {train_mask.sum()}, Eval rows: {eval_mask.sum()}, "
          f"Real gap rows: {real_gap_mask.sum()}")
    assert not (train_mask & eval_mask).any(), "Train and eval overlap!"
    assert not (train_mask & real_gap_mask).any(), "Train and real-gap overlap!"
    assert not (eval_mask & real_gap_mask).any(), "Eval and real-gap overlap!"
    print("All mask-overlap sanity checks passed.")

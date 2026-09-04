import pandas as pd
from ec_loader import load_ec_year
from Gap_masking import add_synthetic_gaps
from block_gap_masking import add_block_synthetic_gaps

# ---------------- Load 2023 data ----------------
# Lysimeter: read the CORRECTED file (properly summed from 1-minute
# readings), NOT re-derived from the original file, which has the ~30x
# scale bug baked into its ET values from whoever created it originally.
lys_2023 = pd.read_csv("combined_lysimeter_2023_CORRECTED.csv", parse_dates=["datetime"])

# EC data was never affected by the scale bug, so this stays as before.
ec_2023 = load_ec_year("C:\\Users\\setar\\OneDrive\\Desktop\\Forecasting Crop Evapotranspiration-original\\Forecasting Crop Evapotranspiration\\data2\\E26\\2023.csv", known_year=2023)

ec_2023.to_csv("combined_ec_2023.csv", index=False)
print(f"Using combined_lysimeter_2023_CORRECTED.csv directly ({len(lys_2023)} rows)")
print(f"Saved combined_ec_2023.csv ({len(ec_2023)} rows)")

# ---------------- Apply the SAME three masking schemes ----------------
# Isolated point (same fractions as originally used: 15% lysimeter, 10% EC)
lys_point = add_synthetic_gaps(lys_2023, group_cols=["lysimeter_id", "season_year"],
                                 value_col="ET", mask_fraction=0.15, random_seed=42)
lys_point.to_csv("lysimeter_2023_point_masked.csv", index=False)

ec_point = add_synthetic_gaps(ec_2023, group_cols=["management", "season_year"],
                                value_col="ET", mask_fraction=0.10, random_seed=42)
ec_point.to_csv("ec_2023_point_masked.csv", index=False)

# Short (3h) blocks
lys_short = add_block_synthetic_gaps(lys_2023, group_cols=["lysimeter_id", "season_year"],
                                       value_col="ET", block_length_steps=6,
                                       mask_fraction=0.15, random_seed=42)
lys_short.to_csv("lysimeter_2023_block_short3h_masked.csv", index=False)

ec_short = add_block_synthetic_gaps(ec_2023, group_cols=["management", "season_year"],
                                      value_col="ET", block_length_steps=6,
                                      mask_fraction=0.10, random_seed=42)
ec_short.to_csv("ec_2023_block_short3h_masked.csv", index=False)

# Long (24h) blocks
lys_long = add_block_synthetic_gaps(lys_2023, group_cols=["lysimeter_id", "season_year"],
                                      value_col="ET", block_length_steps=48,
                                      mask_fraction=0.15, random_seed=42)
lys_long.to_csv("lysimeter_2023_block_long24h_masked.csv", index=False)

ec_long = add_block_synthetic_gaps(ec_2023, group_cols=["management", "season_year"],
                                     value_col="ET", block_length_steps=48,
                                     mask_fraction=0.10, random_seed=42)
ec_long.to_csv("ec_2023_block_long24h_masked.csv", index=False)

print("\nAll 6 masked 2023 datasets saved (isolated point, short3h, long24h x lysimeter, EC).")
print("This 2023 data was never used in any part of model training, validation, or "
      "hyperparameter selection -- it is a genuinely held-out generalization test.")
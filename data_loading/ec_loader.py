"""
ec_loader.py

Loads and standardizes yearly eddy covariance (EC) ET CSV files -- which,
like the lysimeter files, have inconsistent column names, ordering, and
encoding across years -- into one clean combined long-format dataframe.

Known quirks handled:
  - A "units row" in some years (e.g. 2018, 2021) where Year/DoY/Hour show
    "-" and CONV_ET/DIV_ET show the literal string "mm/minutes" instead of
    a value. We don't need to specially detect/drop this row: we never
    trust Year/DoY/Hour anyway (same policy as the lysimeter loader), and
    coercing CONV_ET/DIV_ET to numeric turns "mm/minutes" into NaN, which
    correctly represents "ET not available at this timestamp".
  - "#VALUE!" Excel error strings in CONV_ET/DIV_ET -> coerced to NaN.
  - A corrupted year-column header in 2019 ("#NAME?") -- irrelevant, since
    we rebuild datetime from a known per-file year + the timestamp column,
    not from any Year column, whatever it's named.
  - An extra, non-ET "Lv" column in 2019 with large raw values -- simply
    not selected into the standardized output (not silently kept).
  - "timestamp" vs "Timestamp" capitalization -- matched case-insensitively.
  - Net radiation column naming differences, reusing the same regex
    approach as the lysimeter loader ('Net_radiation', 'Net_Radiation',
    'Net Radiation', 'Net Radiation (Wm^2)').
  - CONV_ET / DIV_ET confirmed to already be half-hourly ET depths (the
    "mm/minutes" unit label is a mislabeling artifact, not an actual rate)
    -- so NO unit conversion is applied. If this assumption changes,
    update EC_VALUE_COLS handling below.

Restricts to the 2018-2022 period (EC data doesn't exist before 2018) and
to the May 1 - Nov 1 growing season, consistent with the lysimeter loader.

Usage:
    from ec_loader import combine_ec_years

    year_files = {
        2018: "path/to/ec_2018.csv",
        2019: "path/to/ec_2019.csv",
        ...
    }
    combined = combine_ec_years(year_files)
"""

import re
import pandas as pd
import numpy as np

# Maps raw ET column name -> standardized management label.
EC_VALUE_COLS = {
    "CONV_ET": "Conventional",
    "DIV_ET": "Diversified",
}

GROWING_SEASON_START = (5, 1)   # May 1
GROWING_SEASON_END = (11, 1)    # Nov 1

MIN_YEAR = 2018  # EC record starts in 2018
MAX_YEAR = 2023  # extended to include 2023, used as a fully held-out generalization test year

WEATHER_COLUMN_PATTERNS = {
    "Temp": re.compile(r"^\s*Temp", re.IGNORECASE),
    "Precip": re.compile(r"^\s*Precip", re.IGNORECASE),
    "RelHum": re.compile(r"^\s*Rel\s*Hum", re.IGNORECASE),
    "WindSpd": re.compile(r"^\s*Wind\s*Spd", re.IGNORECASE),
    "NetRad": re.compile(r"net[\s_]*rad", re.IGNORECASE),
}

TIMESTAMP_PATTERN = re.compile(r"^\s*timestamp\s*$", re.IGNORECASE)


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_timestamp_column(df: pd.DataFrame) -> str:
    matches = [c for c in df.columns if TIMESTAMP_PATTERN.search(c)]
    if len(matches) == 0:
        raise ValueError(f"No timestamp column found. Columns present: {list(df.columns)}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous timestamp column match: {matches}")
    return matches[0]


def _map_weather_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    rename_map = {}
    for standard_name, pattern in WEATHER_COLUMN_PATTERNS.items():
        matches = [c for c in df.columns if pattern.search(c)]
        if len(matches) == 0:
            raise ValueError(
                f"Could not find a column matching '{standard_name}' "
                f"(pattern: {pattern.pattern}). Columns present: {list(df.columns)}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous match for '{standard_name}': multiple columns matched "
                f"{matches}. Please resolve manually."
            )
        rename_map[matches[0]] = standard_name
        if verbose:
            print(f"  Mapped '{matches[0]}' -> '{standard_name}'")
    return df.rename(columns=rename_map)


def _coerce_et_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Convert CONV_ET / DIV_ET to numeric, turning Excel error strings
    ('#VALUE!') and mislabeled unit strings ('mm/minutes') into NaN.
    """
    df = df.copy()
    for raw_col in EC_VALUE_COLS:
        if raw_col not in df.columns:
            raise ValueError(f"Expected ET column '{raw_col}' not found. "
                              f"Columns present: {list(df.columns)}")
        before_numeric = df[raw_col].copy()
        df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
        n_coerced = (before_numeric.notna() & df[raw_col].isna()).sum()
        if verbose and n_coerced > 0:
            examples = before_numeric[before_numeric.notna() & df[raw_col].isna()].unique()[:3]
            print(f"  '{raw_col}': {n_coerced} non-numeric values coerced to NaN "
                  f"(examples: {list(examples)})")
    return df


def _build_datetime(df: pd.DataFrame, known_year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Same policy as the lysimeter loader: never trust an embedded Year
    column (in 2019 it's not even named 'Year' -- it's the corrupted
    '#NAME?'). Always override the year with the known year for this file,
    keeping month/day/time from the timestamp column.

    Fallback: the 2023 EC file has NO timestamp column at all -- only
    Year, DoY (day-of-year), and Hour. In that case, datetime is
    constructed directly from DoY and Hour (using known_year, per the
    same never-trust-the-embedded-Year policy). Based on cross-checking
    Hour against real timestamps in other years' files, the convention is:
    clock time = (Hour - 0.5) hours past midnight (e.g. Hour=0.5 -> 0:00,
    Hour=1.0 -> 0:30, Hour=1.5 -> 1:00).
    """
    df = df.copy()

    try:
        ts_col = _find_timestamp_column(df)
    except ValueError:
        ts_col = None

    if ts_col is not None:
        parsed = pd.to_datetime(df[ts_col], errors="coerce")
        n_bad = parsed.isna().sum()
        if n_bad > 0 and verbose:
            print(f"  WARNING: {n_bad} rows had unparseable timestamps and will be dropped.")

        embedded_years = parsed.dropna().dt.year.unique()
        if verbose and len(embedded_years) > 0 and set(embedded_years) != {known_year}:
            print(f"  NOTE: timestamp column's embedded year(s) {sorted(embedded_years)} "
                  f"differ from known file year {known_year}. Overriding with {known_year}.")

        df["datetime"] = parsed.apply(
            lambda ts: ts.replace(year=known_year) if pd.notna(ts) else pd.NaT
        )
    else:
        if "DoY" not in df.columns or "Hour" not in df.columns:
            raise ValueError(
                "No timestamp column found, and no DoY/Hour columns available as a "
                f"fallback. Columns present: {list(df.columns)}"
            )
        if verbose:
            print(f"  No timestamp column found; constructing datetime from DoY + Hour "
                  f"(clock time = Hour - 0.5), using known_year={known_year}.")
        jan1 = pd.Timestamp(year=known_year, month=1, day=1)
        df["datetime"] = jan1 + pd.to_timedelta(df["DoY"] - 1, unit="D") \
                               + pd.to_timedelta(df["Hour"] - 0.5, unit="h")

    df = df.dropna(subset=["datetime"])
    df["season_year"] = known_year
    return df


def _filter_growing_season(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    year = df["season_year"].iloc[0]
    start = pd.Timestamp(year=year, month=GROWING_SEASON_START[0], day=GROWING_SEASON_START[1])
    end = pd.Timestamp(year=year, month=GROWING_SEASON_END[0], day=GROWING_SEASON_END[1])
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    if verbose:
        print(f"  Growing season filter: {mask.sum()} / {len(df)} rows kept "
              f"({start.date()} to {end.date()})")
    return df[mask]


def _reshape_to_long(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reshape CONV_ET / DIV_ET (wide, one column per management system) into
    long format: one row per (datetime, management), with an 'ET' column
    and 'management' set to the human-readable label.
    """
    id_vars = [c for c in df.columns if c not in EC_VALUE_COLS]
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=list(EC_VALUE_COLS.keys()),
        var_name="management_raw",
        value_name="ET",
    )
    long_df["management"] = long_df["management_raw"].map(EC_VALUE_COLS)
    long_df = long_df.drop(columns=["management_raw"])
    if verbose:
        print(f"  Reshaped to long format: {len(df)} rows x {len(EC_VALUE_COLS)} "
              f"management systems -> {len(long_df)} rows")
    return long_df


def load_ec_year(filepath: str, known_year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Full pipeline for a single yearly EC CSV file. Returns a long-format
    dataframe with columns:
      datetime, season_year, management, ET, Temp, Precip, RelHum, WindSpd, NetRad
    """
    if not (MIN_YEAR <= known_year <= MAX_YEAR):
        raise ValueError(f"Year {known_year} is outside the supported EC range "
                          f"({MIN_YEAR}-{MAX_YEAR}).")

    if verbose:
        print(f"\n=== Loading {filepath} (year={known_year}) ===")

    df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    df = _clean_column_names(df)
    df = _map_weather_columns(df, verbose=verbose)
    df = _coerce_et_columns(df, verbose=verbose)
    df = _build_datetime(df, known_year=known_year, verbose=verbose)
    df = _filter_growing_season(df, verbose=verbose)

    keep_cols = (["datetime", "season_year"] + list(EC_VALUE_COLS.keys())
                 + list(WEATHER_COLUMN_PATTERNS.keys()))
    df = df[keep_cols]

    long_df = _reshape_to_long(df, verbose=verbose)
    long_df = long_df.sort_values(["management", "datetime"]).reset_index(drop=True)
    return long_df


def combine_ec_years(year_files: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Load and combine multiple yearly EC files (2018-2022) into one
    long-format dataframe, sorted by management then datetime.
    """
    out_of_range = [y for y in year_files if not (MIN_YEAR <= y <= MAX_YEAR)]
    if out_of_range:
        raise ValueError(f"Years {out_of_range} are outside the supported EC range "
                          f"({MIN_YEAR}-{MAX_YEAR}); EC data doesn't exist before 2018.")

    frames = []
    for year, path in sorted(year_files.items()):
        frames.append(load_ec_year(path, known_year=year, verbose=verbose))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["management", "datetime"]).reset_index(drop=True)

    if verbose:
        print(f"\n=== Combined EC dataset ===")
        print(f"Total rows: {len(combined)}")
        print(f"Years covered: {sorted(combined['season_year'].unique())}")
        print(f"Management systems: {sorted(combined['management'].unique())}")
        print(f"ET missing: {combined['ET'].isna().sum()} / {len(combined)} "
              f"({100 * combined['ET'].isna().mean():.1f}%)")

    return combined


if __name__ == "__main__":
    test_files = {
        2018: "test_data/ec_2018.csv",
        2019: "test_data/ec_2019.csv",
    }
    result = combine_ec_years(test_files)
    print("\n--- Sample of combined output ---")
    print(result.head(10))
    print("\n--- dtypes ---")
    print(result.dtypes)
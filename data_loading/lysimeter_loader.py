"""
lysimeter_loader.py

Loads and standardizes yearly lysimeter CSV files that have inconsistent
column names, ordering, and encoding across years, into one clean combined
long-format dataframe ready for gap-filling model training.

Handles, per the known quirks in the source files:
  - Header whitespace (leading/trailing spaces on column names)
  - Encoding artifacts on the Temp column (single or double mis-encoded '°')
  - Net radiation column appearing under several different names/positions
    ('Net_radiation', 'Net_Radiation', 'Net Radiation', 'Net Radiation (Wm^2)')
  - Year/DOY columns present in some years (e.g. 2017) but absent in others
    (2018-2022 only have a 'timestamp' column)
  - A 'timestamp' column whose YEAR can be wrong (confirmed in the 2017 file,
    where timestamp said 2019 while Year/DOY said 2017), but whose month/day/
    time-of-day appear reliable. We therefore NEVER trust the year embedded
    in 'timestamp' -- we always override it with the year the file is known
    to represent (passed in explicitly), and only take month/day/hour/minute
    from the timestamp column.
  - A large block of soil-moisture-by-depth columns (e.g. 'H1L1_005_(%vol)')
    that are not used in this analysis and are dropped.

Usage:
    from lysimeter_loader import load_lysimeter_year, combine_lysimeter_years

    year_files = {
        2017: "path/to/lysimeter_2017.csv",
        2018: "path/to/lysimeter_2018.csv",
        ...
    }
    combined = combine_lysimeter_years(year_files, lysimeter_cols=LYSIMETER_COLS)
"""

import re
import pandas as pd
import numpy as np

# The 9 lysimeter ID columns for the matched soil type.
LYSIMETER_COLS = ["H1L1", "H1L2", "H1L6", "H2L1", "H2L3", "H2L5", "H3L2", "H3L3", "H3L4"]

GROWING_SEASON_START = (5, 1)   # May 1
GROWING_SEASON_END = (11, 1)    # Nov 1

# Regex patterns used to identify weather columns regardless of naming
# inconsistencies / encoding artifacts across years.
WEATHER_COLUMN_PATTERNS = {
    "Temp": re.compile(r"^\s*Temp", re.IGNORECASE),
    "Precip": re.compile(r"^\s*Precip", re.IGNORECASE),
    "RelHum": re.compile(r"^\s*Rel\s*Hum", re.IGNORECASE),
    "WindSpd": re.compile(r"^\s*Wind\s*Spd", re.IGNORECASE),
    "NetRad": re.compile(r"net[\s_]*rad", re.IGNORECASE),
}

# Columns to always drop (soil moisture by depth, and the unreliable Year/DOY
# columns -- we rebuild datetime ourselves instead of trusting these).
DROP_PATTERNS = [
    re.compile(r"%vol", re.IGNORECASE),
    re.compile(r"^\s*Year\s*$", re.IGNORECASE),
    re.compile(r"^\s*DOY\s*$", re.IGNORECASE),
]


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _drop_unwanted_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop soil-moisture and Year/DOY columns."""
    to_drop = []
    for col in df.columns:
        if any(pat.search(col) for pat in DROP_PATTERNS):
            to_drop.append(col)
        elif col.lower().startswith("unnamed") and df[col].isna().all():
            to_drop.append(col)
    if verbose and to_drop:
        print(f"  Dropping {len(to_drop)} columns (soil moisture / Year / DOY / empty): "
              f"{to_drop[:5]}{'...' if len(to_drop) > 5 else ''}")
    return df.drop(columns=to_drop)


def _map_weather_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Rename weather columns to a standard schema using regex matching,
    since exact names/encodings vary by year. Raises if a required
    weather variable can't be found, or if more than one column matches
    (ambiguous), so problems surface immediately rather than silently
    picking the wrong column.
    """
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


def _build_datetime(df: pd.DataFrame, known_year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Build a reliable datetime column from the 'timestamp' column, but
    override the year with the known_year the file represents (since the
    embedded year in 'timestamp' has been shown to be wrong in at least
    one file). Month/day/hour/minute are taken from 'timestamp' as-is.
    """
    df = df.copy()
    parsed = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad = parsed.isna().sum()
    if n_bad > 0 and verbose:
        print(f"  WARNING: {n_bad} rows had unparseable timestamps and will be dropped.")

    # Detect mismatched years before overriding, so it's visible in the log.
    embedded_years = parsed.dropna().dt.year.unique()
    if verbose and len(embedded_years) > 0 and set(embedded_years) != {known_year}:
        print(f"  NOTE: timestamp column's embedded year(s) {sorted(embedded_years)} "
              f"differ from known file year {known_year}. Overriding with {known_year}, "
              f"keeping month/day/time from timestamp.")

    df["datetime"] = parsed.apply(
        lambda ts: ts.replace(year=known_year) if pd.notna(ts) else pd.NaT
    )
    df = df.dropna(subset=["datetime"])
    df["season_year"] = known_year
    return df


def _filter_growing_season(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Keep only rows within May 1 - Nov 1 of the row's season_year."""
    year = df["season_year"].iloc[0]
    start = pd.Timestamp(year=year, month=GROWING_SEASON_START[0], day=GROWING_SEASON_START[1])
    end = pd.Timestamp(year=year, month=GROWING_SEASON_END[0], day=GROWING_SEASON_END[1])
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    if verbose:
        print(f"  Growing season filter: {mask.sum()} / {len(df)} rows kept "
              f"({start.date()} to {end.date()})")
    return df[mask]


def _coerce_lysimeter_columns(df: pd.DataFrame, lysimeter_cols: list, verbose: bool = True) -> pd.DataFrame:
    """
    Convert lysimeter ET columns to numeric. Some years contain the literal
    string ' NaN' (with a leading space) instead of a true missing marker,
    which pandas reads as text rather than recognizing as NaN -- this
    coerces those (and any other stray non-numeric text) to proper NaN.
    """
    df = df.copy()
    for col in lysimeter_cols:
        before = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        newly_bad = before.notna() & df[col].isna() & (before.astype(str).str.strip() != "")
        n_coerced = newly_bad.sum()
        if verbose and n_coerced > 0:
            examples = before[newly_bad].unique()[:3]
            print(f"  '{col}': {n_coerced} non-numeric values coerced to NaN "
                  f"(examples: {list(examples)})")
    return df


def _convert_et_sign(df: pd.DataFrame, lysimeter_cols: list, verbose: bool = True) -> pd.DataFrame:
    """
    The raw lysimeter signal is a mass-loss delta (<=0 = water leaving the
    lysimeter via ET). We convert to a positive ET depth via absolute value,
    to be consistent with how ET is reported elsewhere (e.g. EC, and the
    paper's existing RMSE/MAE figures, which are positive magnitudes).
    """
    df = df.copy()
    for col in lysimeter_cols:
        n_positive_before = (df[col] > 0).sum()
        if verbose and n_positive_before > 0:
            print(f"  '{col}': {n_positive_before} values were already >0 before "
                  f"sign conversion (e.g. rain/irrigation mass gain) -- abs() leaves these unchanged in magnitude.")
        df[col] = df[col].abs()
    return df


def _reshape_to_long(df: pd.DataFrame, lysimeter_cols: list, verbose: bool = True) -> pd.DataFrame:
    """
    Reshape from wide (one column per lysimeter replicate) to long format:
    one row per (datetime, lysimeter_id), with an 'ET' column, so replicates
    can be pooled for one soil-type model.
    """
    id_vars = [c for c in df.columns if c not in lysimeter_cols]
    missing = [c for c in lysimeter_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Expected lysimeter columns not found in file: {missing}")

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=lysimeter_cols,
        var_name="lysimeter_id",
        value_name="ET",
    )
    if verbose:
        print(f"  Reshaped to long format: {len(df)} rows x {len(lysimeter_cols)} "
              f"lysimeters -> {len(long_df)} rows")
    return long_df


def load_lysimeter_year(
    filepath: str,
    known_year: int,
    lysimeter_cols: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline for a single yearly lysimeter CSV file:
      1. Load + clean column names
      2. Drop soil moisture / Year / DOY columns
      3. Map weather columns to a standard schema (regex-based, encoding-safe)
      4. Rebuild a reliable datetime (overriding the file's embedded year)
      5. Filter to the May 1 - Nov 1 growing season
      6. Reshape lysimeter replicate columns from wide to long format

    Returns a long-format dataframe with columns:
      datetime, season_year, lysimeter_id, ET, Temp, Precip, RelHum, WindSpd, NetRad
    """
    if lysimeter_cols is None:
        lysimeter_cols = LYSIMETER_COLS

    if verbose:
        print(f"\n=== Loading {filepath} (year={known_year}) ===")

    df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    df = _clean_column_names(df)
    df = _drop_unwanted_columns(df, verbose=verbose)
    df = _map_weather_columns(df, verbose=verbose)
    df = _build_datetime(df, known_year=known_year, verbose=verbose)
    df = _filter_growing_season(df, verbose=verbose)

    keep_cols = ["datetime", "season_year"] + lysimeter_cols + list(WEATHER_COLUMN_PATTERNS.keys())
    df = df[keep_cols]
    df = _coerce_lysimeter_columns(df, lysimeter_cols=lysimeter_cols, verbose=verbose)
    df = _convert_et_sign(df, lysimeter_cols=lysimeter_cols, verbose=verbose)

    long_df = _reshape_to_long(df, lysimeter_cols=lysimeter_cols, verbose=verbose)
    long_df = long_df.sort_values(["lysimeter_id", "datetime"]).reset_index(drop=True)
    return long_df


def combine_lysimeter_years(
    year_files: dict,
    lysimeter_cols: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and combine multiple yearly lysimeter files into one long-format
    dataframe, sorted by lysimeter_id then datetime.

    year_files: dict mapping {year_int: filepath_str}
    """
    frames = []
    for year, path in sorted(year_files.items()):
        frames.append(load_lysimeter_year(path, known_year=year,
                                           lysimeter_cols=lysimeter_cols, verbose=verbose))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["lysimeter_id", "datetime"]).reset_index(drop=True)

    if verbose:
        print(f"\n=== Combined dataset ===")
        print(f"Total rows: {len(combined)}")
        print(f"Years covered: {sorted(combined['season_year'].unique())}")
        print(f"Lysimeters: {sorted(combined['lysimeter_id'].unique())}")
        print(f"ET missing: {combined['ET'].isna().sum()} / {len(combined)} "
              f"({100 * combined['ET'].isna().mean():.1f}%)")

    return combined


if __name__ == "__main__":
    # Quick self-test against synthetic files mimicking the real header quirks.
    test_files = {
        2017: "test_data/lysimeter_2017.csv",
        2018: "test_data/lysimeter_2018.csv",
    }
    result = combine_lysimeter_years(test_files)
    print("\n--- Sample of combined output ---")
    print(result.head(10))
    print("\n--- dtypes ---")
    print(result.dtypes)
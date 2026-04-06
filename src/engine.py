"""
YieldSense Statistical Engine
==============================
Provides data loading/cleaning and statistical utilities for the
Climate Sensitivity Index dashboard.

Lag direction convention
------------------------
Rainfall *leads* price:  corr( rainfall(t),  price(t + lag) )
Implemented by:          corr( rainfall,      price.shift(-lag) )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_and_clean_data(filepath: str = "data/agri_data_master.csv") -> pd.DataFrame:
    """Load master CSV, parse dates, and apply the grouped interpolation policy.

    Missing-value policy (applied within each (country, crop) group):
    - Gaps of **<= 2 consecutive months**: filled by linear interpolation.
    - Gaps of **> 2 consecutive months**: those rows are dropped so that
      downstream statistics do not rely on long imputed runs.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame sorted by yyyy_mm with a proper DatetimeTZUnaware
        period column.
    """
    df = pd.read_csv(filepath)

    # Parse yyyy_mm → Period then to Timestamp for arithmetic convenience
    df["yyyy_mm"] = pd.to_datetime(df["yyyy_mm"], format="%Y-%m")
    df = df.sort_values("yyyy_mm").reset_index(drop=True)

    numeric_cols = ["rainfall_mm", "temp_c", "price_usd"]

    cleaned_groups: list[pd.DataFrame] = []
    for (country, crop), grp in df.groupby(["country", "crop"], sort=False):
        grp = grp.sort_values("yyyy_mm").copy()

        for col in numeric_cols:
            if grp[col].isna().any():
                grp = _interpolate_with_gap_policy(grp, col, max_gap=2)

        cleaned_groups.append(grp)

    result = pd.concat(cleaned_groups, ignore_index=True)
    return result.sort_values("yyyy_mm").reset_index(drop=True)


def _interpolate_with_gap_policy(
    grp: pd.DataFrame, col: str, max_gap: int = 2
) -> pd.DataFrame:
    """Linearly interpolate gaps of <= max_gap; drop rows in longer gaps."""
    series = grp[col].copy()

    # Identify consecutive NaN run lengths
    is_na = series.isna()
    # Compute run-length encoding for NaN stretches
    run_ids = is_na.ne(is_na.shift()).cumsum()
    run_lengths = is_na.groupby(run_ids).transform("sum")

    # Rows that are NaN AND belong to a run longer than max_gap
    drop_mask = is_na & (run_lengths > max_gap)

    # Interpolate the short gaps first
    series_interpolated = series.copy()
    series_interpolated[~drop_mask] = (
        series_interpolated[~drop_mask].interpolate(method="linear")
    )

    grp = grp.copy()
    grp[col] = series_interpolated

    # Drop rows that belong to long-gap NaN runs (still NaN after interpolation)
    grp = grp[~drop_mask].reset_index(drop=True)
    return grp


# ---------------------------------------------------------------------------
# Subset helper
# ---------------------------------------------------------------------------

def get_subset(
    df: pd.DataFrame, country: str, crop: str
) -> pd.DataFrame:
    """Return rows for the given country and crop, sorted by yyyy_mm."""
    mask = (df["country"] == country) & (df["crop"] == crop)
    return df[mask].sort_values("yyyy_mm").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Outlier capping
# ---------------------------------------------------------------------------

def cap_outliers(series: pd.Series, percentile: float = 99) -> pd.Series:
    """Cap values at the given upper percentile (handles price spikes)."""
    upper = np.nanpercentile(series.values, percentile)
    return series.clip(upper=upper)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def min_max_normalize(series: pd.Series) -> pd.Series:
    """Return min–max scaled series in [0, 1].  Returns NaN series if constant."""
    lo = series.min()
    hi = series.max()
    if hi == lo:
        return pd.Series(np.nan, index=series.index, name=series.name)
    return (series - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Lag correlation
# ---------------------------------------------------------------------------

def compute_lag_correlation(
    subset: pd.DataFrame, lag: int
) -> dict:
    """Compute Pearson r between rainfall(t) and price(t + lag).

    Rainfall *leads* price:
        corr( subset["rainfall_mm"],  subset["price_usd"].shift(-lag) )

    Parameters
    ----------
    subset : pd.DataFrame
        Rows for a single (country, crop) pair sorted by yyyy_mm.
    lag : int
        Number of months price is shifted forward (must be >= 1).

    Returns
    -------
    dict with keys:
        lag        – the lag used
        r          – signed Pearson r (float or nan)
        p_value    – two-tailed p-value (float or nan)
        n          – number of paired observations used
        reason     – None, or a string explaining why r is nan
    """
    rainfall = subset["rainfall_mm"]
    price_shifted = subset["price_usd"].shift(-lag)

    # Align valid pairs
    combined = pd.concat(
        [rainfall.rename("r"), price_shifted.rename("p")], axis=1
    ).dropna()

    n = len(combined)

    if n < 3:
        return {"lag": lag, "r": np.nan, "p_value": np.nan, "n": n,
                "reason": "insufficient data after alignment"}

    r_var = combined["r"].var(ddof=1)
    p_var = combined["p"].var(ddof=1)

    if r_var == 0:
        return {"lag": lag, "r": np.nan, "p_value": np.nan, "n": n,
                "reason": "zero variance in rainfall"}
    if p_var == 0:
        return {"lag": lag, "r": np.nan, "p_value": np.nan, "n": n,
                "reason": "zero variance in price"}

    r_val, p_val = scipy_stats.pearsonr(combined["r"], combined["p"])
    return {"lag": lag, "r": float(r_val), "p_value": float(p_val),
            "n": n, "reason": None}


# ---------------------------------------------------------------------------
# Sensitivity Index
# ---------------------------------------------------------------------------

def compute_sensitivity_index(subset: pd.DataFrame) -> dict:
    """Compute Climate Sensitivity Index = max(|r_lag|) for lag in 1..6.

    Parameters
    ----------
    subset : pd.DataFrame
        Rows for a single (country, crop) pair.

    Returns
    -------
    dict with keys:
        sensitivity_index  – max absolute correlation across lags 1–6 (float or nan)
        best_lag           – lag that achieves sensitivity_index (int or nan)
        r_at_best_lag      – signed r at best_lag (float or nan)
        lag_results        – list of per-lag dicts from compute_lag_correlation
    """
    lag_results = [compute_lag_correlation(subset, lag) for lag in range(1, 7)]

    valid = [(res["lag"], abs(res["r"]), res["r"])
             for res in lag_results if not np.isnan(res["r"])]

    if not valid:
        return {
            "sensitivity_index": np.nan,
            "best_lag": np.nan,
            "r_at_best_lag": np.nan,
            "lag_results": lag_results,
        }

    best_lag, max_abs_r, best_r = max(valid, key=lambda x: x[1])
    return {
        "sensitivity_index": float(max_abs_r),
        "best_lag": int(best_lag),
        "r_at_best_lag": float(best_r),
        "lag_results": lag_results,
    }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    subset: pd.DataFrame, threshold: float = 2.0
) -> pd.DataFrame:
    """Z-score anomaly detection for rainfall within the given subset.

    Uses the full-period mean and std of rainfall_mm in the subset
    (representing the full 2000–2024 window for that country/crop).

    Parameters
    ----------
    subset : pd.DataFrame
        Rows for a single (country, crop) pair.
    threshold : float
        Z-score threshold above which a month is flagged as a 'climate shock'.

    Returns
    -------
    pd.DataFrame
        Input subset with two additional columns:
        - ``rainfall_zscore``  – z-score of rainfall_mm
        - ``is_shock``         – boolean flag (|z| > threshold)
    """
    out = subset.copy()
    mu = subset["rainfall_mm"].mean()
    sigma = subset["rainfall_mm"].std(ddof=1)

    if sigma == 0 or np.isnan(sigma):
        out["rainfall_zscore"] = np.nan
        out["is_shock"] = False
    else:
        out["rainfall_zscore"] = (subset["rainfall_mm"] - mu) / sigma
        out["is_shock"] = out["rainfall_zscore"].abs() > threshold

    return out


# ---------------------------------------------------------------------------
# Summary report helper
# ---------------------------------------------------------------------------

def build_summary_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary report DataFrame for all (country, crop) pairs.

    Columns: country, crop, best_lag, sensitivity_index, r_at_best_lag,
             count_points_used, shocks_count.
    """
    rows = []
    for (country, crop), grp in df.groupby(["country", "crop"]):
        subset = grp.sort_values("yyyy_mm").reset_index(drop=True)
        idx = compute_sensitivity_index(subset)
        anomalies = detect_anomalies(subset)
        rows.append({
            "country": country,
            "crop": crop,
            "best_lag": idx["best_lag"],
            "sensitivity_index": idx["sensitivity_index"],
            "r_at_best_lag": idx["r_at_best_lag"],
            "count_points_used": len(subset),
            "shocks_count": int(anomalies["is_shock"].sum()),
        })
    return pd.DataFrame(rows)

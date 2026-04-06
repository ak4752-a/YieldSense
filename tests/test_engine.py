"""
Tests for src/engine.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine import (
    _interpolate_with_gap_policy,
    build_summary_report,
    cap_outliers,
    compute_lag_correlation,
    compute_sensitivity_index,
    detect_anomalies,
    get_subset,
    load_and_clean_data,
    min_max_normalize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_subset(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Create a minimal (country, crop) subset with no missing values."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    rainfall = 50 + 30 * np.sin(np.arange(n) * 2 * np.pi / 12) + rng.normal(0, 5, n)
    price = 200 + np.cumsum(rng.normal(0, 3, n))
    return pd.DataFrame({
        "yyyy_mm": dates,
        "country": "TestCountry",
        "crop": "TestCrop",
        "rainfall_mm": rainfall,
        "temp_c": 20.0,
        "price_usd": price,
    })


def _make_df_with_groups() -> pd.DataFrame:
    """Two (country, crop) groups."""
    a = _make_subset(60, seed=1)
    b = _make_subset(60, seed=2)
    b["country"] = "OtherCountry"
    return pd.concat([a, b], ignore_index=True)


# ---------------------------------------------------------------------------
# _interpolate_with_gap_policy
# ---------------------------------------------------------------------------

class TestInterpolateGapPolicy:
    def test_short_gap_filled(self):
        s = pd.Series([1.0, np.nan, 3.0])
        df = pd.DataFrame({"rainfall_mm": s, "temp_c": 0.0, "price_usd": 0.0})
        result = _interpolate_with_gap_policy(df, "rainfall_mm", max_gap=2)
        assert result["rainfall_mm"].isna().sum() == 0
        assert pytest.approx(result["rainfall_mm"].iloc[1], rel=1e-3) == 2.0

    def test_exactly_two_gap_filled(self):
        s = pd.Series([0.0, np.nan, np.nan, 6.0])
        df = pd.DataFrame({"rainfall_mm": s, "temp_c": 0.0, "price_usd": 0.0})
        result = _interpolate_with_gap_policy(df, "rainfall_mm", max_gap=2)
        assert result["rainfall_mm"].isna().sum() == 0
        assert pytest.approx(result["rainfall_mm"].iloc[1], rel=1e-3) == 2.0
        assert pytest.approx(result["rainfall_mm"].iloc[2], rel=1e-3) == 4.0

    def test_long_gap_rows_dropped(self):
        s = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0])
        df = pd.DataFrame({"rainfall_mm": s, "temp_c": 0.0, "price_usd": 0.0})
        result = _interpolate_with_gap_policy(df, "rainfall_mm", max_gap=2)
        # Rows 1, 2, 3 (the 3-length NaN run) should be dropped
        assert len(result) == 2
        assert result["rainfall_mm"].isna().sum() == 0

    def test_no_nans_unchanged(self):
        s = pd.Series([1.0, 2.0, 3.0])
        df = pd.DataFrame({"rainfall_mm": s, "temp_c": 0.0, "price_usd": 0.0})
        result = _interpolate_with_gap_policy(df, "rainfall_mm", max_gap=2)
        assert list(result["rainfall_mm"]) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# load_and_clean_data
# ---------------------------------------------------------------------------

class TestLoadAndCleanData:
    def test_loads_csv(self, tmp_path):
        csv_content = (
            "yyyy_mm,country,crop,rainfall_mm,temp_c,price_usd\n"
            "2000-01,India,Rice,50.0,28.0,450.0\n"
            "2000-02,India,Rice,55.0,29.0,460.0\n"
            "2000-03,India,Rice,,30.0,470.0\n"   # single NaN gap
            "2000-04,India,Rice,60.0,31.0,480.0\n"
        )
        p = tmp_path / "test.csv"
        p.write_text(csv_content)
        df = load_and_clean_data(str(p))
        assert len(df) == 4
        assert df["rainfall_mm"].isna().sum() == 0

    def test_long_gap_drops_rows(self, tmp_path):
        rows = ["yyyy_mm,country,crop,rainfall_mm,temp_c,price_usd"]
        rows.append("2000-01,India,Rice,50.0,28.0,450.0")
        rows.append("2000-02,India,Rice,,28.0,450.0")
        rows.append("2000-03,India,Rice,,28.0,450.0")
        rows.append("2000-04,India,Rice,,28.0,450.0")  # 3-row NaN run → dropped
        rows.append("2000-05,India,Rice,60.0,28.0,450.0")
        p = tmp_path / "test.csv"
        p.write_text("\n".join(rows))
        df = load_and_clean_data(str(p))
        assert len(df) == 2  # only the two non-NaN endpoints remain

    def test_yyyy_mm_is_datetime(self, tmp_path):
        csv_content = (
            "yyyy_mm,country,crop,rainfall_mm,temp_c,price_usd\n"
            "2000-01,India,Rice,50.0,28.0,450.0\n"
        )
        p = tmp_path / "test.csv"
        p.write_text(csv_content)
        df = load_and_clean_data(str(p))
        assert pd.api.types.is_datetime64_any_dtype(df["yyyy_mm"])


# ---------------------------------------------------------------------------
# get_subset
# ---------------------------------------------------------------------------

class TestGetSubset:
    def test_returns_correct_rows(self):
        df = _make_df_with_groups()
        sub = get_subset(df, "TestCountry", "TestCrop")
        assert (sub["country"] == "TestCountry").all()
        assert (sub["crop"] == "TestCrop").all()

    def test_empty_for_missing(self):
        df = _make_df_with_groups()
        sub = get_subset(df, "Nowhere", "NoCrop")
        assert sub.empty


# ---------------------------------------------------------------------------
# cap_outliers
# ---------------------------------------------------------------------------

class TestCapOutliers:
    def test_values_capped(self):
        s = pd.Series([1.0, 2.0, 3.0, 1000.0])
        result = cap_outliers(s, percentile=99)
        assert result.max() < 1000.0

    def test_normal_values_unchanged(self):
        s = pd.Series(range(1, 101), dtype=float)
        result = cap_outliers(s, percentile=99)
        # Only extreme top should be clipped
        assert result[:98].equals(s[:98])


# ---------------------------------------------------------------------------
# min_max_normalize
# ---------------------------------------------------------------------------

class TestMinMaxNormalize:
    def test_range(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = min_max_normalize(s)
        assert pytest.approx(result.min(), abs=1e-9) == 0.0
        assert pytest.approx(result.max(), abs=1e-9) == 1.0

    def test_constant_returns_nan(self):
        s = pd.Series([5.0, 5.0, 5.0])
        result = min_max_normalize(s)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# compute_lag_correlation
# ---------------------------------------------------------------------------

class TestComputeLagCorrelation:
    def test_returns_r(self):
        subset = _make_subset(100)
        result = compute_lag_correlation(subset, lag=1)
        assert "r" in result
        assert "lag" in result
        assert result["lag"] == 1

    def test_valid_r_range(self):
        subset = _make_subset(100)
        result = compute_lag_correlation(subset, lag=3)
        if not np.isnan(result["r"]):
            assert -1.0 <= result["r"] <= 1.0

    def test_zero_variance_rainfall_returns_nan(self):
        subset = _make_subset(50)
        subset = subset.copy()
        subset["rainfall_mm"] = 42.0   # constant → zero variance
        result = compute_lag_correlation(subset, lag=2)
        assert np.isnan(result["r"])
        assert result["reason"] is not None

    def test_zero_variance_price_returns_nan(self):
        subset = _make_subset(50)
        subset = subset.copy()
        subset["price_usd"] = 100.0
        result = compute_lag_correlation(subset, lag=2)
        assert np.isnan(result["r"])

    def test_lag_direction(self):
        """Rainfall(t) should predict price(t + lag), not the reverse."""
        n = 100
        dates = pd.date_range("2000-01-01", periods=n, freq="MS")
        rainfall = np.zeros(n)
        rainfall[10] = 100.0   # spike at t=10
        price = np.zeros(n)
        price[13] = 100.0      # spike at t=13 (lag=3)
        subset = pd.DataFrame({
            "yyyy_mm": dates, "country": "X", "crop": "Y",
            "rainfall_mm": rainfall, "temp_c": 20.0, "price_usd": price,
        })
        r_lag3 = compute_lag_correlation(subset, lag=3)
        r_lag1 = compute_lag_correlation(subset, lag=1)
        # |r| at lag=3 should be stronger than at lag=1
        if not (np.isnan(r_lag3["r"]) or np.isnan(r_lag1["r"])):
            assert abs(r_lag3["r"]) >= abs(r_lag1["r"])

    def test_insufficient_data(self):
        subset = _make_subset(2)  # only 2 rows → not enough after shift
        result = compute_lag_correlation(subset, lag=1)
        assert np.isnan(result["r"])


# ---------------------------------------------------------------------------
# compute_sensitivity_index
# ---------------------------------------------------------------------------

class TestComputeSensitivityIndex:
    def test_returns_expected_keys(self):
        subset = _make_subset(100)
        result = compute_sensitivity_index(subset)
        assert "sensitivity_index" in result
        assert "best_lag" in result
        assert "r_at_best_lag" in result
        assert "lag_results" in result

    def test_sensitivity_index_is_max_abs_r(self):
        subset = _make_subset(100)
        result = compute_sensitivity_index(subset)
        lag_r_abs = [abs(r["r"]) for r in result["lag_results"] if not np.isnan(r["r"])]
        if lag_r_abs and not np.isnan(result["sensitivity_index"]):
            assert pytest.approx(result["sensitivity_index"], abs=1e-9) == max(lag_r_abs)

    def test_best_lag_in_1_to_6(self):
        subset = _make_subset(100)
        result = compute_sensitivity_index(subset)
        if not np.isnan(float(result["best_lag"])):
            assert result["best_lag"] in range(1, 7)

    def test_nan_when_all_zero_variance(self):
        subset = _make_subset(50)
        subset = subset.copy()
        subset["rainfall_mm"] = 0.0
        result = compute_sensitivity_index(subset)
        assert np.isnan(result["sensitivity_index"])


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------

class TestDetectAnomalies:
    def test_adds_columns(self):
        subset = _make_subset(60)
        result = detect_anomalies(subset)
        assert "rainfall_zscore" in result.columns
        assert "is_shock" in result.columns

    def test_shock_flag_correct(self):
        subset = _make_subset(60)
        subset = subset.copy()
        # Manually inject an extreme value
        subset.at[0, "rainfall_mm"] = 1e6
        result = detect_anomalies(subset)
        assert result.at[0, "is_shock"] is True or result.at[0, "is_shock"] == True  # noqa: E712

    def test_constant_rainfall_no_shock(self):
        subset = _make_subset(60)
        subset = subset.copy()
        subset["rainfall_mm"] = 42.0
        result = detect_anomalies(subset)
        assert result["is_shock"].sum() == 0

    def test_zscore_near_zero_for_mean(self):
        subset = _make_subset(60)
        result = detect_anomalies(subset)
        # The mean of z-scores should be near 0
        if not result["rainfall_zscore"].isna().all():
            assert pytest.approx(result["rainfall_zscore"].mean(), abs=1e-9) == 0.0


# ---------------------------------------------------------------------------
# build_summary_report
# ---------------------------------------------------------------------------

class TestBuildSummaryReport:
    def test_columns(self):
        df = _make_df_with_groups()
        report = build_summary_report(df)
        expected_cols = {
            "country", "crop", "best_lag", "sensitivity_index",
            "r_at_best_lag", "count_points_used", "shocks_count",
        }
        assert expected_cols.issubset(set(report.columns))

    def test_row_count(self):
        df = _make_df_with_groups()
        report = build_summary_report(df)
        # Two unique (country, crop) combinations
        assert len(report) == 2

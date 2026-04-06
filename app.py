"""
YieldSense – Climate-Driven Global Food Security Dashboard
===========================================================
Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.engine import (
    REQUIRED_COLUMNS,
    build_summary_report,
    cap_outliers,
    compute_sensitivity_index,
    detect_anomalies,
    get_subset,
    load_and_clean_data,
    load_and_clean_uploaded_csv,
    min_max_normalize,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YieldSense",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.environ.get(
    "DATA_PATH",
    os.path.join(os.path.dirname(__file__), "data", "agri_data_master.csv"),
)

# ---------------------------------------------------------------------------
# Cached data-loading helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading & cleaning data …")
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.error(
            f"CSV not found at **{DATA_PATH}**.  "
            "Please place `agri_data_master.csv` in the `data/` folder and restart.  "
            "You can also upload a CSV directly in the sidebar."
        )
        st.stop()
    return load_and_clean_data(DATA_PATH)


@st.cache_data(show_spinner="Parsing & cleaning uploaded CSV …")
def load_uploaded_data(file_bytes: bytes) -> tuple[pd.DataFrame | None, list[str]]:
    """Parse and clean an uploaded CSV (bytes). Cached by content hash."""
    return load_and_clean_uploaded_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Computing global sensitivity index …")
def get_summary(df: pd.DataFrame) -> pd.DataFrame:
    return build_summary_report(df)


# ---------------------------------------------------------------------------
# Template CSV for download
# ---------------------------------------------------------------------------

_TEMPLATE_CSV = (
    "yyyy_mm,country,crop,rainfall_mm,temp_c,price_usd\n"
    "2024-01,India,Rice,15.2,22.4,450.50\n"
    "2024-01,Brazil,Maize,120.5,26.1,210.20\n"
    "2024-01,USA,Wheat,45.3,5.2,240.10\n"
    "2024-02,India,Rice,18.1,23.0,450.50\n"
    "2024-02,Brazil,Maize,115.0,25.8,210.20\n"
    "2024-02,USA,Wheat,38.7,6.1,240.10\n"
)

# ---------------------------------------------------------------------------
# Sidebar – branding & upload
# ---------------------------------------------------------------------------
st.sidebar.title("🌾 YieldSense")
st.sidebar.markdown("Climate-Driven Food Security Dashboard")
st.sidebar.divider()

# ── CSV upload ──────────────────────────────────────────────────────────────
st.sidebar.subheader("📂 Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload master CSV (optional)",
    type=["csv"],
    help=(
        "Upload your own master CSV to override the built-in dataset for this "
        "session.  The file is used in-memory only — nothing is saved to disk."
    ),
)

st.sidebar.download_button(
    label="⬇️ Download template CSV",
    data=_TEMPLATE_CSV,
    file_name="yieldsense_template.csv",
    mime="text/csv",
    help="Download a minimal example CSV showing the required format.",
)

# ── Data format / instructions expander ─────────────────────────────────────
with st.sidebar.expander("ℹ️ Data format / Upload instructions"):
    st.markdown(
        f"""
**Required columns** (exact names):

| Column | Type | Description |
|--------|------|-------------|
| `yyyy_mm` | string | Month in `YYYY-MM` format, e.g. `2024-01` |
| `country` | string | Country name, e.g. `India`, `USA` |
| `crop` | string | Crop name, e.g. `Wheat`, `Rice`, `Maize`, `Soybeans` |
| `rainfall_mm` | float | Monthly cumulative rainfall (mm) |
| `temp_c` | float | Mean monthly temperature (°C) |
| `price_usd` | float | Global commodity price (USD / metric tonne) |

**Example rows:**
```
{", ".join(REQUIRED_COLUMNS)}
2024-01,India,Rice,15.2,22.4,450.50
2024-01,Brazil,Maize,120.5,26.1,210.20
```

**Note on `price_usd`:** this is a *global reference price* for the
crop-month combination (e.g. from the World Bank Pink Sheet).  It is
the same value for every country in the same crop-month row.

**Suggested data sources:**
- **Prices** – [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets)
  (monthly commodity prices, free download)
- **Climate** – [World Bank Climate Knowledge Portal](https://climateknowledgeportal.worldbank.org/)
  (monthly rainfall + temperature by country, free)

Merge the two sources on `yyyy_mm` + `crop` to produce the master CSV.
"""
    )

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Load data (upload takes priority over local CSV)
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    result_df, messages = load_uploaded_data(file_bytes)

    if result_df is None:
        # Fatal validation errors
        for msg in messages:
            st.sidebar.error(msg)
        st.error(
            "The uploaded CSV could not be loaded.  "
            "See the sidebar for details and refer to the **Data format / Upload "
            "instructions** section for the required schema."
        )
        st.stop()

    for msg in messages:
        st.sidebar.warning(msg)

    df = result_df
    st.sidebar.success("✅ Using uploaded CSV for this session.")
else:
    df = load_data()

summary_df = get_summary(df)

# ---------------------------------------------------------------------------
# Sidebar – country / crop selectors (need df to be loaded first)
# ---------------------------------------------------------------------------
countries = sorted(df["country"].unique().tolist())
crops = sorted(df["crop"].unique().tolist())

selected_country = st.sidebar.selectbox("🗺️ Country", countries, index=0)
selected_crop = st.sidebar.selectbox("🌱 Crop", crops, index=0)

st.sidebar.divider()
st.sidebar.caption("Data: 2000–2024 | Monthly granularity")

# ---------------------------------------------------------------------------
# Subset & stats
# ---------------------------------------------------------------------------
subset = get_subset(df, selected_country, selected_crop)

if subset.empty:
    st.warning("No data available for the selected combination.")
    st.stop()

# Cap price outliers for display / normalisation
subset = subset.copy()
subset["price_usd"] = cap_outliers(subset["price_usd"])

# Compute sensitivity index
idx_result = compute_sensitivity_index(subset)
sensitivity_index = idx_result["sensitivity_index"]
best_lag = idx_result["best_lag"]
r_at_best_lag = idx_result["r_at_best_lag"]

# Anomaly detection
subset_with_shocks = detect_anomalies(subset)
shocks_count = int(subset_with_shocks["is_shock"].sum())

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title(f"🌾 YieldSense — {selected_country} / {selected_crop}")
st.markdown(
    "Statistical analysis of how **local rainfall** predicts "
    "**global commodity prices** (Market Response Latency model)."
)

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    si_display = f"{sensitivity_index:.3f}" if not np.isnan(sensitivity_index) else "N/A"
    st.metric("🎯 Sensitivity Index", si_display,
              help="max |Pearson r| across lags 1–6 months")

with col2:
    bl_display = f"{best_lag} month(s)" if not np.isnan(float(best_lag)) else "N/A"
    st.metric("⏱️ Best Lag", bl_display,
              help="Lag at which |r| is maximised")

with col3:
    r_display = f"{r_at_best_lag:.3f}" if not np.isnan(float(r_at_best_lag)) else "N/A"
    st.metric("📊 r at Best Lag", r_display,
              help="Signed Pearson correlation at the best lag")

with col4:
    st.metric("⚡ Climate Shocks", shocks_count,
              help="Months where |z-score of rainfall| > 2")

st.divider()

# ---------------------------------------------------------------------------
# Dual-axis chart: Rainfall (bars) + Price (line)
# ---------------------------------------------------------------------------
st.subheader("📈 Rainfall vs. Commodity Price (Dual-Axis)")

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=subset_with_shocks["yyyy_mm"],
        y=subset_with_shocks["rainfall_mm"],
        name="Rainfall (mm)",
        marker_color=np.where(
            subset_with_shocks["is_shock"], "rgba(220,53,69,0.7)", "rgba(30,144,255,0.55)"
        ).tolist(),
        yaxis="y1",
        hovertemplate="<b>%{x|%Y-%m}</b><br>Rainfall: %{y:.1f} mm<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=subset_with_shocks["yyyy_mm"],
        y=subset_with_shocks["price_usd"],
        name="Price (USD)",
        line=dict(color="darkorange", width=2),
        yaxis="y2",
        hovertemplate="<b>%{x|%Y-%m}</b><br>Price: $%{y:.2f}<extra></extra>",
    )
)

fig.update_layout(
    xaxis=dict(title="Date"),
    yaxis=dict(title="Rainfall (mm)", showgrid=False),
    yaxis2=dict(title="Price (USD)", overlaying="y", side="right", showgrid=False),
    legend=dict(x=0.01, y=0.99),
    hovermode="x unified",
    height=420,
    margin=dict(l=50, r=50, t=30, b=50),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "🔴 Red bars = climate shocks (|z-score| > 2).  "
    f"Lag analysis: rainfall today predicts price {best_lag} month(s) later."
)

# ---------------------------------------------------------------------------
# Lag correlation bar chart
# ---------------------------------------------------------------------------
st.subheader("🔗 Lag Correlation Profile (Lags 1–6 months)")

lag_data = idx_result["lag_results"]
lag_vals = [r["lag"] for r in lag_data]
r_vals = [r["r"] if not np.isnan(r["r"]) else 0 for r in lag_data]
colors = [
    "rgba(220,53,69,0.8)" if lag == best_lag else "rgba(30,144,255,0.6)"
    for lag in lag_vals
]

fig_lag = go.Figure(
    go.Bar(
        x=[f"Lag {l}" for l in lag_vals],
        y=r_vals,
        marker_color=colors,
        text=[f"{v:.3f}" for v in r_vals],
        textposition="outside",
    )
)
fig_lag.update_layout(
    yaxis=dict(title="Pearson r", range=[-1, 1]),
    height=300,
    margin=dict(l=40, r=20, t=20, b=40),
)
fig_lag.add_hline(y=0, line_dash="dot", line_color="grey")
st.plotly_chart(fig_lag, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Choropleth: Sensitivity Index by country for selected crop
# ---------------------------------------------------------------------------
st.subheader(f"🗺️ Climate Sensitivity Index — {selected_crop} (all countries)")

crop_summary = summary_df[summary_df["crop"] == selected_crop].copy()
crop_summary["si_display"] = crop_summary["sensitivity_index"].round(3)

iso_map = {
    "India": "IND", "USA": "USA", "Brazil": "BRA", "China": "CHN",
    "Ukraine": "UKR", "Argentina": "ARG", "Australia": "AUS",
    "Canada": "CAN", "Russia": "RUS", "France": "FRA",
}
crop_summary["iso_alpha"] = crop_summary["country"].map(iso_map)

fig_map = go.Figure(
    go.Choropleth(
        locations=crop_summary["iso_alpha"],
        z=crop_summary["sensitivity_index"].fillna(0),
        text=crop_summary["country"],
        colorscale="Reds",
        zmin=0, zmax=1,
        colorbar_title="Sensitivity<br>Index",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Sensitivity Index: %{z:.3f}<extra></extra>"
        ),
    )
)
fig_map.update_layout(
    geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
    height=400,
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_map, use_container_width=True)
st.caption(
    "Darker red = stronger statistical link between local rainfall and global price.  "
    "Computed as max(|r|) across lags 1–6 months."
)

st.divider()

# ---------------------------------------------------------------------------
# Normalised trend (min-max)
# ---------------------------------------------------------------------------
st.subheader("📐 Min–Max Normalised Trends")

norm_rain = min_max_normalize(subset["rainfall_mm"])
norm_price = min_max_normalize(cap_outliers(subset["price_usd"]))

fig_norm = go.Figure()
fig_norm.add_trace(
    go.Scatter(x=subset["yyyy_mm"], y=norm_rain, name="Rainfall (norm)",
               line=dict(color="steelblue", width=1.5))
)
fig_norm.add_trace(
    go.Scatter(x=subset["yyyy_mm"], y=norm_price, name="Price (norm)",
               line=dict(color="darkorange", width=1.5))
)
fig_norm.update_layout(
    yaxis=dict(title="Normalised value [0–1]"),
    height=300, margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig_norm, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# CSV download – full summary report
# ---------------------------------------------------------------------------
st.subheader("⬇️ Download Summary Report")

# Build the per-selection row
selection_row = summary_df[
    (summary_df["country"] == selected_country) & (summary_df["crop"] == selected_crop)
].copy()

st.dataframe(selection_row, use_container_width=True, hide_index=True)

# Full-report download
csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Full Summary (all countries & crops)",
    data=csv_bytes,
    file_name="yieldsense_summary_report.csv",
    mime="text/csv",
)

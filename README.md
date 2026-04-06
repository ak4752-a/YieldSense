# 🌾 YieldSense

**YieldSense** is an interactive data-science dashboard that quantifies the statistical relationship between environmental volatility (rainfall) and global food commodity prices. It implements a *Market Response Latency* model: given a country and crop, it finds the lag (1–6 months) at which local rainfall most strongly predicts the global commodity price.

---

## Features

- **Climate Sensitivity Index** – max |Pearson r| across lags 1–6 months between rainfall and price.
- **Best-lag detection** – automatically identifies the delay at which rainfall signal is strongest.
- **Anomaly / shock detection** – flags months where |z-score| of rainfall exceeds 2 standard deviations.
- **Interactive choropleth map** – compare sensitivity across all countries for a selected crop.
- **Dual-axis time-series chart** – rainfall bars (red = shock) overlaid with commodity price line.
- **Min–max normalised trends** – visual alignment of rainfall and price on a common [0, 1] scale.
- **Downloadable summary CSV** – one-click export of the full sensitivity report.

---

## Project Structure

```
YieldSense/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── data/
│   └── agri_data_master.csv  # Monthly agri data (2000–2024)
├── src/
│   ├── __init__.py
│   └── engine.py           # Statistical engine (loading, lag correlation, anomaly detection)
├── tests/
│   ├── __init__.py
│   └── test_engine.py      # pytest suite (29 tests)
└── .streamlit/
    └── config.toml         # Streamlit theme & server settings
```

---

## Data Format

Place a CSV at `data/agri_data_master.csv` with the following columns:

| Column         | Type    | Description                                      |
|----------------|---------|--------------------------------------------------|
| `yyyy_mm`      | string  | Month in `YYYY-MM` format (e.g. `2000-01`)       |
| `country`      | string  | Country name (e.g. `India`, `USA`)               |
| `crop`         | string  | Crop name (e.g. `Wheat`, `Rice`, `Maize`)        |
| `rainfall_mm`  | float   | Monthly cumulative rainfall in millimetres       |
| `temp_c`       | float   | Mean monthly temperature in °C (loaded; reserved for future modelling) |
| `price_usd`    | float   | Global commodity price in USD per metric tonne   |

Missing-value policy (applied per country/crop group):
- Gaps of **≤ 2 consecutive months** are filled by linear interpolation.
- Gaps of **> 2 consecutive months** are dropped entirely.

---

## Installation

**Python 3.10+ required.**

```bash
# 1. Clone the repository
git clone https://github.com/ak4752-a/YieldSense.git
cd YieldSense

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running Locally

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).  
Use the sidebar to select a **country** and **crop**; all charts update instantly.

---

## Running Tests

```bash
python -m pytest tests/test_engine.py -v
```

All 29 tests should pass.

---

## Deployment

### Streamlit Community Cloud (recommended – free)

1. Ensure `data/agri_data_master.csv` is committed (it is tracked by default in this repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and fill in:
   - **Repository:** `ak4752-a/YieldSense`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. The app will be live in under a minute.

> The `.streamlit/config.toml` file is committed and will be picked up automatically.
> No secrets or environment variables are required.

### Docker

```dockerfile
# Build
docker build -t yieldsense .
# Run
docker run -p 8501:8501 yieldsense
```

A minimal `Dockerfile`:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

### Other PaaS (Heroku, Render, Railway)

Add a `Procfile`:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## Algorithm: Market Response Latency Model

For each *(country, crop)* pair:

1. **Lag correlations** – compute Pearson *r* between `rainfall(t)` and `price(t + lag)` for `lag ∈ {1, 2, 3, 4, 5, 6}` months.
2. **Climate Sensitivity Index** – `max(|r|)` across all lags.
3. **Best lag** – the lag that achieves the sensitivity index.
4. **Shock detection** – months where `|z-score(rainfall)| > 2` are flagged as *climate shocks*.

A higher Sensitivity Index indicates a stronger statistical link between local rainfall patterns and the global commodity price for that country/crop combination.

---

## License

MIT

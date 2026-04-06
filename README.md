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
├── render.yaml             # Render Blueprint (deployment config)
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

### Render (recommended)

The repository includes a `render.yaml` Blueprint, so deployment is one click:

1. Fork or push the repository to your GitHub account (including the `data/` folder with `agri_data_master.csv`).
2. Go to [render.com](https://render.com) and sign in / create an account.
3. Click **New → Blueprint** and connect your GitHub repository.
4. Render will detect `render.yaml` automatically and create the **yieldsense** web service.
5. Click **Apply** — Render installs dependencies and starts the app.

**Manual setup (without Blueprint):**

1. Click **New → Web Service** and connect the GitHub repo.
2. Branch: `main`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
5. Click **Create Web Service**.

**Environment variables:** no environment variables are required. The app reads `data/agri_data_master.csv` from the repository using a relative path. If you need to supply the data file from an external location (e.g. a Render persistent disk), set the `DATA_PATH` environment variable to the absolute path of the CSV.

**Data file:** `data/agri_data_master.csv` is committed to the repository and will be available on Render automatically. If the file is not committed, attach a [Render persistent disk](https://render.com/docs/disks) at a mount path (e.g. `/data`) and set `DATA_PATH=/data/agri_data_master.csv` in the service's environment variables.

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

### Streamlit Community Cloud

1. Push your repository to GitHub (including the `data/` folder).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select the repo → set **Main file path** to `app.py`.
4. Click **Deploy**. The app will be live in under a minute.

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

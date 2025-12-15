# Env Sensors API & Dashboard

Small FastAPI service that samples two I2C sensors (SHT30 for temp/humidity, QMP6988 for temp/pressure), caches the latest reading, optionally stores it in PostgreSQL, and serves a Plotly/Tailwind dashboard at the root path.

## What it does
- Reads the sensors every aligned tick (defaults to 30s at :00/:30 UTC) and caches the last good sample.
- Serves live metrics at `/v1/metrics` and a health snapshot at `/v1/healthz`.
- Streams historical buckets and time-weighted averages from PostgreSQL (if enabled).
- Hosts the dashboard SPA from `static/index.html` at `/`, using the same origin for API calls.

## Quick start (local)
1) Install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) (Optional) Start PostgreSQL
```bash
cp docker-compose.yml.example docker-compose.yml  # adjust credentials if you want
docker compose up -d postgres
```
Set `DATABASE_URL` to the DSN (e.g. `postgresql://envuser:change_me_strong@127.0.0.1:5432/envdb`).

3) Run the service
```bash
python env_api.py  # binds 0.0.0.0:8080
```
Then open http://localhost:8080/ to see the dashboard.

## Configuration
Environment variables (sensible defaults are baked in):
- `POLL_SEC` — sampling cadence in seconds (default `30`).
- `BASE_STEP_SEC` — tick alignment (default `30`). Start/end times for range/average must align to this.
- `STALE_FACTOR` — consider cache stale after `STALE_FACTOR * POLL_SEC` seconds without a fresh sample (default `3`).
- `DEFAULT_MAX_POINTS` — safety cap for `/metrics_range` results (default `100000`).
- `DATABASE_URL` — PostgreSQL DSN. If unset, DB features are disabled unless `DB_ENABLE=1` with a DSN.
- `DB_ENABLE` — force-enable/disable DB layer (`1` or `0`).
- `DB_AUTO_SCHEMA` — create/upgrade schema on boot (default `1`).
- `ALLOW_ORIGINS` — comma-separated CORS allowlist. Default `*` keeps things open for the bundled frontend.
- `I2C_BUS`, `SHT30_ADDR`, `QMP_ADDR` — hardware knobs for the driver (defaults: bus `2`, addresses `0x44` and `0x70`).

## API
- `GET /v1/healthz` — cache freshness + DB status. Includes `lag_seconds` and `last_sample_ts` (second precision, UTC).
- `GET /v1/metrics` — latest cached sample: `temperature_c`, `humidity_pct`, `pressure_hpa`, `qmp_temp_c`, `ts`, `status` (`OK` or `OUT_DATED`). Returns 503 if no sample yet.
- `GET /v1/metrics_range?start=...&end=...&step=...&max_points=...` — bucketed history from Postgres. `start/end` must be aligned to `BASE_STEP_SEC`; `step` must be a multiple of `BASE_STEP_SEC`. Respects `max_points` guard.
- `GET /v1/average?val=...&start=...&end=...` — time-weighted average for one column (`temp_sht_c`, `temp_qmp_c`, `humidity_pct`, `pressure_hpa`) over `[start, end)`, with the same alignment rules.

## Data model (PostgreSQL)
Table `env_samples` (second-precision `TIMESTAMPTZ` primary key):
- `ts`, `temp_sht_c`, `temp_qmp_c`, `humidity_pct`, `pressure_hpa`, `status` (0=OK). Index on `ts`.
Schema is ensured automatically when `DB_AUTO_SCHEMA=1`.

## Frontend notes
- Dashboard lives in `static/index.html` and is served at `/` by FastAPI. It uses `window.location.origin` so there are no hardcoded hosts.
- Polling: gauges every 30s, averages every 60s, history interval selectable (aligned to the API step rules).
- Time ranges: presets (e.g., 1h, 6h, 12h, 24h) with auto step sizing and UTC alignment to the 30s tick.

## Operational behavior
- Sampling and DB writes run in background threads; samples are timestamped at the aligned tick, not the wall-clock when the function returns.
- Cache is never overwritten with bad reads; failed samples only flip `status`/`error`.
- All timestamps are UTC, second precision. Client charts align to the same grid so points line up cleanly.

## Troubleshooting
- Dashboard is empty: check browser devtools → Network for `/v1/metrics` and `/v1/metrics_range` responses; alignment errors return 400 with details.
- CORS issues: set `ALLOW_ORIGINS` to the exact scheme+host of your frontend (or `*` for development).
- DB missing: leave `DATABASE_URL` unset or set `DB_ENABLE=0`; `/metrics_range` and `/average` will return 503 until a DB is available.

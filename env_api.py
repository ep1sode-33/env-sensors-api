# env_api.py
# HTTP layer + sampler/cache + DB writer.
# Hardware access is isolated in driver.py (env_api only calls driver.metrics()).
# All timestamps are second-precision (no fractional seconds).
# Sampling is aligned to wall-clock ticks (e.g. :00 / :30 when BASE_STEP_SEC=30).

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, APIRouter, Query

import driver  # <-- your hardware-only module (metrics/healthz)

# ---- Config from env vars ----
# Sampling cadence (seconds). You want 30s.
POLL_SEC = float(os.getenv("POLL_SEC", "30.0"))

# Tick alignment base. You want 30 (align to :00/:30 in UTC).
BASE_STEP_SEC = int(os.getenv("BASE_STEP_SEC", "30"))

# Consider cache stale if no fresh sample for > STALE_FACTOR * POLL_SEC
STALE_FACTOR = float(os.getenv("STALE_FACTOR", "3.0"))

# Range endpoint protection
DEFAULT_MAX_POINTS = int(os.getenv("DEFAULT_MAX_POINTS", "100000"))

# PostgreSQL
DB_DSN = os.getenv("DATABASE_URL")  # e.g. postgresql://user:pass@host:5432/dbname
DB_ENABLE = os.getenv("DB_ENABLE", "1" if DB_DSN else "0") == "1"
DB_AUTO_SCHEMA = os.getenv("DB_AUTO_SCHEMA", "1") == "1"


# ---- Time helpers (SECOND precision only) ----
def _dt_now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso_now() -> str:
    return _dt_now_utc().isoformat()


def parse_dt(s: str) -> datetime:
    """
    Parse ISO-8601 timestamp string into timezone-aware datetime (UTC), second precision.
    Accepts:
      - '2025-01-01T00:00:00Z'
      - '2025-01-01T00:00:00+00:00'
      - '2025-01-01T00:00:00'  (treated as UTC)
      - UNIX seconds (e.g. '1735689600')
    """
    s = s.strip()
    if s.isdigit():
        return datetime.fromtimestamp(int(s), tz=timezone.utc).replace(microsecond=0)

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0)


def _is_aligned(dt: datetime, step_sec: int) -> bool:
    return (int(dt.timestamp()) % step_sec) == 0


def _next_tick(step_sec: int) -> datetime:
    """
    Return the next wall-clock tick aligned to step_sec (UTC).
    If now is exactly aligned, returns now (so we can sample immediately).
    """
    now = _dt_now_utc()
    epoch = int(now.timestamp())
    rem = epoch % step_sec
    if rem == 0:
        return now
    epoch += (step_sec - rem)
    return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(microsecond=0)


def _sleep_until(t: datetime) -> None:
    while True:
        now = _dt_now_utc()
        diff = (t - now).total_seconds()
        if diff <= 0:
            return
        # sleep in one shot; it is OK if slightly late
        time.sleep(diff)


# ---- Cache maintained by sampler thread (NOT driver) ----
_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {
    "temperature_c": None,
    "humidity_pct": None,
    "pressure_hpa": None,
    "qmp_temp_c": None,
    "last_sample_ts": None,   # datetime (UTC, second precision)
    "last_error": None,       # str|None
    "last_ok": False,         # bool
}

_sampler_stop = threading.Event()
_sampler_thread: Optional[threading.Thread] = None

# ---- DB store (optional) ----
_store = None
_store_err: Optional[str] = None
_db_writer_stop = threading.Event()
_db_writer_thread: Optional[threading.Thread] = None


def _init_db_or_disable() -> None:
    global _store, _store_err
    if not DB_ENABLE:
        _store = None
        _store_err = "DB disabled (DB_ENABLE=0)"
        return
    if not DB_DSN:
        _store = None
        _store_err = "DB disabled (DATABASE_URL not set)"
        return

    try:
        from pg_store import DbConfig, PostgresStore
        _store = PostgresStore(DbConfig(dsn=DB_DSN))
        _store.start()
        if DB_AUTO_SCHEMA:
            _store.ensure_schema()
        _store_err = None
    except Exception as e:
        _store = None
        _store_err = f"DB init failed: {e!r}"


def _compute_health(now: datetime) -> Dict[str, Any]:
    """
    Compute health from cache:
      - ok: last sample is recent and last_ok is True
      - last_sample_ts: ISO string (second precision) or None
      - lag_seconds: seconds since last_sample_ts or None
      - error: last_error or stale(...) message
    """
    with _cache_lock:
        last_ts: Optional[datetime] = _cache["last_sample_ts"]
        last_ok: bool = bool(_cache["last_ok"])
        last_err: Optional[str] = _cache["last_error"]

    if last_ts is None:
        return {
            "ok": False,
            "error": last_err or "no sample yet",
            "last_sample_ts": None,
            "lag_seconds": None,
        }

    lag = int((now - last_ts).total_seconds())
    stale_limit = int(POLL_SEC * STALE_FACTOR)

    if lag > stale_limit:
        return {
            "ok": False,
            "error": f"stale ({lag}s old)",
            "last_sample_ts": last_ts.isoformat(),
            "lag_seconds": lag,
        }

    if not last_ok:
        return {
            "ok": False,
            "error": last_err or "sensor error",
            "last_sample_ts": last_ts.isoformat(),
            "lag_seconds": lag,
        }

    return {
        "ok": True,
        "error": None,
        "last_sample_ts": last_ts.isoformat(),
        "lag_seconds": lag,
    }


def _sampler_loop() -> None:
    """
    Sampler loop:
      - Wait until next aligned tick (00/30 when step=30)
      - Call driver.metrics() exactly once per tick
      - Publish sample timestamp = tick (not 'now')
      - Cache is updated in env_api only
    """
    step = int(BASE_STEP_SEC)

    # Cold start: wait up to step seconds to align to tick boundary
    tick = _next_tick(step)
    _sleep_until(tick)

    while not _sampler_stop.is_set():
        # Read hardware at tick
        m = driver.metrics()  # hardware-only call; returns ok/error + values

        with _cache_lock:
            if m.get("ok"):
                _cache["temperature_c"] = m["temperature_c"]
                _cache["humidity_pct"] = m["humidity_pct"]
                _cache["pressure_hpa"] = m["pressure_hpa"]
                _cache["qmp_temp_c"] = m["qmp_temp_c"]
                _cache["last_sample_ts"] = tick  # IMPORTANT: use tick
                _cache["last_ok"] = True
                _cache["last_error"] = None
            else:
                # Do NOT overwrite last good values; only mark error status
                _cache["last_ok"] = False
                _cache["last_error"] = m.get("error") or "sensor error"

        # Next tick
        tick = datetime.fromtimestamp(int(tick.timestamp()) + step, tz=timezone.utc).replace(microsecond=0)
        _sleep_until(tick)


def _db_writer_loop() -> None:
    """
    Background DB writer:
      - writes ONLY when cache is healthy (Scheme A: store real samples only)
      - ts written = last_sample_ts (aligned tick, second precision)
      - avoids duplicates by remembering last inserted ts
    """
    global _store_err
    last_inserted_ts: Optional[str] = None

    while not _db_writer_stop.is_set():
        try:
            if _store is None:
                time.sleep(1.0)
                continue

            now = _dt_now_utc()
            h = _compute_health(now)
            if not h["ok"]:
                time.sleep(0.5)
                continue

            with _cache_lock:
                ts_dt: Optional[datetime] = _cache["last_sample_ts"]
                if ts_dt is None:
                    time.sleep(0.5)
                    continue
                ts_iso = ts_dt.isoformat()
                if ts_iso == last_inserted_ts:
                    time.sleep(0.5)
                    continue

                # Copy latest values
                t_sht = _cache["temperature_c"]
                t_qmp = _cache["qmp_temp_c"]
                rh = _cache["humidity_pct"]
                p = _cache["pressure_hpa"]

            # Insert as OK (status=0)
            _store.insert_sample(
                ts=ts_dt,  # already second precision
                temp_sht_c=float(t_sht),
                temp_qmp_c=float(t_qmp),
                humidity_pct=float(rh),
                pressure_hpa=float(p),
                status=0,
            )
            last_inserted_ts = ts_iso
            _store_err = None

        except Exception as e:
            _store_err = f"DB write error: {e!r}"
            time.sleep(2.0)

        time.sleep(0.2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start sampler
    global _sampler_thread
    _sampler_stop.clear()
    _sampler_thread = threading.Thread(target=_sampler_loop, daemon=True, name="sampler")
    _sampler_thread.start()

    # Init DB + start writer if possible
    _init_db_or_disable()
    global _db_writer_thread
    if _store is not None:
        _db_writer_stop.clear()
        _db_writer_thread = threading.Thread(target=_db_writer_loop, daemon=True, name="db-writer")
        _db_writer_thread.start()

    yield

    # Shutdown
    _sampler_stop.set()
    _db_writer_stop.set()
    if _store is not None:
        try:
            _store.stop()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/v1")


@router.get("/healthz")
def healthz():
    """
    {
      "ok": true/false,
      "error": null or string,
      "last_sample_ts": "ISO8601 (seconds)" or null,
      "lag_seconds": integer or null,
      "db_ok": bool,
      "db_error": string|null,
      "ts": "current time (seconds)"
    }
    """
    now = _dt_now_utc()
    h = _compute_health(now)

    db_ok = (_store is not None)
    return {
        "ok": h["ok"],
        "error": h["error"],
        "last_sample_ts": h["last_sample_ts"],
        "lag_seconds": h["lag_seconds"],
        "db_ok": db_ok,
        "db_error": None if db_ok else _store_err,
        "ts": now.isoformat(),
    }


@router.get("/metrics")
def metrics():
    """
    Returns cached metrics. If cache is stale/error, status becomes "OUT_DATED".
    If we have never sampled successfully, returns 503.
    """
    now = _dt_now_utc()
    h = _compute_health(now)

    with _cache_lock:
        last_ts: Optional[datetime] = _cache["last_sample_ts"]
        if last_ts is None:
            raise HTTPException(status_code=503, detail=h["error"] or "no sample yet")

        payload = {
            "temperature_c": _cache["temperature_c"],
            "humidity_pct": _cache["humidity_pct"],
            "pressure_hpa": _cache["pressure_hpa"],
            "qmp_temp_c": _cache["qmp_temp_c"],
            "ts": last_ts.isoformat(),  # sample time (tick), seconds only
        }

    payload["status"] = "OK" if h["ok"] else "OUT_DATED"
    return payload


@router.get("/metrics_range")
def metrics_range(
    start: str = Query(..., description="Start time (ISO-8601 or unix seconds)"),
    end: str = Query(..., description="End time (ISO-8601 or unix seconds)"),
    step: int = Query(..., description="Bucket step in seconds (must be multiple of BASE_STEP_SEC)"),
    max_points: int = Query(DEFAULT_MAX_POINTS, description="Safety cap for returned points"),
):
    """
    Historical range query backed by PostgreSQL (Scheme A).
    Strong alignment requirement:
      - start and end MUST be aligned to BASE_STEP_SEC, else 400
    """
    if _store is None:
        raise HTTPException(status_code=503, detail=_store_err or "DB unavailable")

    if step <= 0:
        raise HTTPException(status_code=400, detail="step must be > 0")
    if step < BASE_STEP_SEC or (step % BASE_STEP_SEC) != 0:
        raise HTTPException(status_code=400, detail=f"step must be a multiple of {BASE_STEP_SEC} seconds (>= {BASE_STEP_SEC})")
    if max_points <= 0:
        raise HTTPException(status_code=400, detail="max_points must be > 0")

    try:
        dt_start = parse_dt(start)
        dt_end = parse_dt(end)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid start/end: {e!r}")

    if dt_end < dt_start:
        raise HTTPException(status_code=400, detail="end must be >= start")

    # Strong alignment: start/end must land on BASE_STEP_SEC ticks
    if not _is_aligned(dt_start, BASE_STEP_SEC) or not _is_aligned(dt_end, BASE_STEP_SEC):
        raise HTTPException(status_code=400, detail=f"start/end must be aligned to {BASE_STEP_SEC}s tick boundaries")

    total_sec = int((dt_end - dt_start).total_seconds())
    est_points = (total_sec // step) + 1
    if est_points > max_points:
        raise HTTPException(
            status_code=400,
            detail=f"too many points: {est_points} > max_points={max_points}. Increase step or max_points.",
        )

    try:
        points = _store.query_range_buckets(
            start=dt_start,
            end=dt_end,
            step_seconds=step,
            max_points=max_points,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db query failed: {e!r}")

    return {
        "start": dt_start.isoformat(),
        "end": dt_end.isoformat(),
        "step_seconds": step,
        "count": len(points),
        "points": points,
    }


@router.get("/average")
def average(
    val: str = Query(..., description="One of: temp_sht_c, temp_qmp_c, humidity_pct, pressure_hpa"),
    start: str = Query(..., description="Start time (ISO-8601 or unix seconds)"),
    end: str = Query(..., description="End time (ISO-8601 or unix seconds)"),
):
    if _store is None:
        raise HTTPException(status_code=503, detail=_store_err or "DB unavailable")

    allowed = {"temp_sht_c", "temp_qmp_c", "humidity_pct", "pressure_hpa"}
    if val not in allowed:
        raise HTTPException(status_code=400, detail=f"val must be one of: {', '.join(sorted(allowed))}")

    try:
        dt_start = parse_dt(start)
        dt_end = parse_dt(end)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid start/end: {e!r}")

    if dt_end < dt_start:
        raise HTTPException(status_code=400, detail="end must be >= start")

    # Strong alignment (same policy as metrics_range)
    if not _is_aligned(dt_start, BASE_STEP_SEC) or not _is_aligned(dt_end, BASE_STEP_SEC):
        raise HTTPException(status_code=400, detail=f"start/end must be aligned to {BASE_STEP_SEC}s tick boundaries")

    try:
        avg = _store.average_time(
            start=dt_start,
            end=dt_end,
            column=val,
            base_step_seconds=BASE_STEP_SEC,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db query failed: {e!r}")

    return {"avg": avg}


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("env_api:app", host="0.0.0.0", port=8080)
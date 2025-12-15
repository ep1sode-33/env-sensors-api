# pg_store.py
# PostgreSQL storage layer for time-series samples (Scheme A: store only real samples).
# Requires: psycopg[binary] >= 3.x  (and psycopg_pool which ships with psycopg)
#
# Timestamp policy:
#   - API and DB both use SECOND precision only (no fractional seconds).
#   - Table uses TIMESTAMPTZ(0) and we truncate existing data to seconds on schema ensure.
#
# Value formatting policy:
#   - API outputs should be clean (e.g. 2 decimals). REAL/float can show tails after AVG().
#   - We ROUND() in SQL and also round again in Python before returning JSON.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

try:
    import psycopg
    from psycopg import sql
    from psycopg_pool import ConnectionPool
except Exception as e:  # pragma: no cover
    psycopg = None
    sql = None
    ConnectionPool = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class DbConfig:
    dsn: str
    min_size: int = 1
    max_size: int = 5
    timeout: float = 5.0


class PostgresStore:
    """
    env_samples schema (Scheme A):
      ts (timestamptz(0) primary key)  -- second precision only
      temp_sht_c (real)
      temp_qmp_c (real)
      humidity_pct (real)
      pressure_hpa (real)
      status (smallint)  -- 0=OK, 1=MISSING, 2=ERROR (Scheme A typically stores only OK rows)
    """

    # Output precision (for JSON)
    OUT_DECIMALS = 2

    def __init__(self, cfg: DbConfig):
        if psycopg is None or ConnectionPool is None or sql is None:
            raise RuntimeError(f"psycopg not available: {_IMPORT_ERROR!r}")

        self.cfg = cfg
        self.pool = ConnectionPool(
            conninfo=cfg.dsn,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            timeout=cfg.timeout,
            open=False,
        )

    def start(self) -> None:
        self.pool.open()

    def stop(self) -> None:
        self.pool.close()

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        """
        Force UTC + second precision (microsecond=0).
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.replace(microsecond=0)

    @classmethod
    def _round_out(cls, v: Any) -> Optional[float]:
        """
        Normalize numeric results for JSON output.
        Handles float/Decimal/etc. Returns float rounded to OUT_DECIMALS.
        """
        if v is None:
            return None
        try:
            return round(float(v), cls.OUT_DECIMALS)
        except Exception:
            return None

    def ensure_schema(self) -> None:
        """
        Ensure the schema exists and timestamps are second-precision.
        Safe to call multiple times.
        """
        stmts = [
            """
            CREATE TABLE IF NOT EXISTS env_samples (
                ts            TIMESTAMPTZ(0) PRIMARY KEY,
                temp_sht_c    REAL,
                temp_qmp_c    REAL,
                humidity_pct  REAL,
                pressure_hpa  REAL,
                status        SMALLINT NOT NULL DEFAULT 0
            );
            """,
            # Force ts to second precision even if table existed previously
            """
            ALTER TABLE env_samples
              ALTER COLUMN ts TYPE TIMESTAMPTZ(0)
              USING date_trunc('second', ts);
            """,
            "CREATE INDEX IF NOT EXISTS env_samples_ts_idx ON env_samples (ts);",
        ]

        with self.pool.connection() as conn:
            for s in stmts:
                conn.execute(s)
            conn.commit()

    def insert_sample(
        self,
        ts: datetime,
        temp_sht_c: float,
        temp_qmp_c: float,
        humidity_pct: float,
        pressure_hpa: float,
        status: int = 0,  # 0=OK
    ) -> None:
        ts = self._to_utc(ts)

        # Optional: round before storing (keeps raw rows tidy; AVG() still needs ROUND()).
        t_sht = round(float(temp_sht_c), self.OUT_DECIMALS)
        t_qmp = round(float(temp_qmp_c), self.OUT_DECIMALS)
        rh = round(float(humidity_pct), self.OUT_DECIMALS)
        p = round(float(pressure_hpa), self.OUT_DECIMALS)

        q = """
        INSERT INTO env_samples (ts, temp_sht_c, temp_qmp_c, humidity_pct, pressure_hpa, status)
        VALUES (%(ts)s, %(t_sht)s, %(t_qmp)s, %(rh)s, %(p)s, %(status)s)
        ON CONFLICT (ts) DO UPDATE SET
            temp_sht_c   = EXCLUDED.temp_sht_c,
            temp_qmp_c   = EXCLUDED.temp_qmp_c,
            humidity_pct = EXCLUDED.humidity_pct,
            pressure_hpa = EXCLUDED.pressure_hpa,
            status       = EXCLUDED.status
        ;
        """

        params = {
            "ts": ts,
            "t_sht": t_sht,
            "t_qmp": t_qmp,
            "rh": rh,
            "p": p,
            "status": int(status),
        }

        with self.pool.connection() as conn:
            conn.execute(q, params)
            conn.commit()

    def get_last_ts(self) -> Optional[datetime]:
        q = "SELECT ts FROM env_samples ORDER BY ts DESC LIMIT 1;"
        with self.pool.connection() as conn:
            row = conn.execute(q).fetchone()
        if not row:
            return None
        ts: datetime = row[0]
        return self._to_utc(ts)

    def query_range_buckets(
        self,
        start: datetime,
        end: datetime,
        step_seconds: int,
        max_points: int,
    ) -> List[Dict[str, Any]]:
        """
        Returns regular buckets using generate_series.
        For each bucket [t, t+step), aggregate AVG of available samples; if none -> NULL fields + status='MISSING'.

        NOTE: We ROUND(AVG(...), 2) in SQL to avoid float tails in JSON.
        """
        start = self._to_utc(start)
        end = self._to_utc(end)

        if step_seconds <= 0:
            raise ValueError("step_seconds must be > 0")
        if max_points <= 0:
            raise ValueError("max_points must be > 0")

        total_sec = (end - start).total_seconds()
        if total_sec < 0:
            raise ValueError("end must be >= start")

        est_points = int(total_sec // step_seconds) + 1
        if est_points > max_points:
            raise ValueError(f"too many points: {est_points} > max_points={max_points}")

        q = f"""
        WITH params AS (
            SELECT
                %(start)s::timestamptz AS start_ts,
                %(end)s::timestamptz   AS end_ts,
                (%(step)s::int || ' seconds')::interval AS step_iv
        ),
        buckets AS (
            SELECT
                generate_series(p.start_ts, p.end_ts, p.step_iv) AS bucket_ts,
                p.step_iv AS step_iv
            FROM params p
        ),
        windowed AS (
            SELECT
                b.bucket_ts,
                ROUND(AVG(s.temp_sht_c)::numeric, {self.OUT_DECIMALS})   AS temp_sht_c,
                ROUND(AVG(s.temp_qmp_c)::numeric, {self.OUT_DECIMALS})   AS temp_qmp_c,
                ROUND(AVG(s.humidity_pct)::numeric, {self.OUT_DECIMALS}) AS humidity_pct,
                ROUND(AVG(s.pressure_hpa)::numeric, {self.OUT_DECIMALS}) AS pressure_hpa,
                COUNT(s.ts)         AS n
            FROM buckets b
            LEFT JOIN env_samples s
              ON s.ts >= b.bucket_ts
             AND s.ts <  (b.bucket_ts + b.step_iv)
            GROUP BY b.bucket_ts
        )
        SELECT
            bucket_ts,
            temp_sht_c,
            temp_qmp_c,
            humidity_pct,
            pressure_hpa,
            CASE WHEN n = 0 THEN 'MISSING' ELSE 'OK' END AS status
        FROM windowed
        ORDER BY bucket_ts ASC;
        """

        params = {"start": start, "end": end, "step": int(step_seconds)}

        out: List[Dict[str, Any]] = []
        with self.pool.connection() as conn:
            rows = conn.execute(q, params).fetchall()

        for (ts, t_sht, t_qmp, rh, p, status) in rows:
            ts_utc = self._to_utc(ts).isoformat()  # second precision guaranteed
            out.append(
                {
                    "ts": ts_utc,
                    "temperature_c": self._round_out(t_sht),
                    "qmp_temp_c": self._round_out(t_qmp),
                    "humidity_pct": self._round_out(rh),
                    "pressure_hpa": self._round_out(p),
                    "status": str(status),
                }
            )

        return out

    def average_time(
        self,
        start: datetime,
        end: datetime,
        column: str,
        base_step_seconds: int,
    ) -> Optional[float]:
        """
        Time-weighted average over [start, end).

        Handles outages/missing gaps by capping each sample's weight to base_step_seconds.
        avg = SUM(v * dur) / SUM(dur)
        dur = min(next_ts - ts, base_step_seconds), clamped >= 0

        NOTE: result is rounded to OUT_DECIMALS in SQL + Python.
        """
        start = self._to_utc(start)
        end = self._to_utc(end)

        if end < start:
            raise ValueError("end must be >= start")
        if base_step_seconds <= 0:
            raise ValueError("base_step_seconds must be > 0")

        allowed = {
            "temp_sht_c": "temp_sht_c",
            "temp_qmp_c": "temp_qmp_c",
            "humidity_pct": "humidity_pct",
            "pressure_hpa": "pressure_hpa",
        }
        if column not in allowed:
            raise ValueError(f"invalid column: {column}")

        col_ident = sql.Identifier(allowed[column])

        q = sql.SQL(f"""
            WITH data AS (
                SELECT
                    ts,
                    {col_ident.as_string(None)} AS v,
                    LEAD(ts) OVER (ORDER BY ts) AS next_ts
                FROM env_samples
                WHERE ts >= %(start)s
                  AND ts <  %(end)s
                  AND status = 0
                  AND {col_ident.as_string(None)} IS NOT NULL
            ),
            weighted AS (
                SELECT
                    v,
                    GREATEST(
                        0,
                        LEAST(
                            %(step)s::double precision,
                            EXTRACT(EPOCH FROM (COALESCE(next_ts, %(end)s) - ts))
                        )
                    ) AS dur
                FROM data
            )
            SELECT
                CASE
                    WHEN SUM(dur) = 0 THEN NULL
                    ELSE ROUND((SUM(v * dur) / SUM(dur))::numeric, {self.OUT_DECIMALS})
                END AS avg
            FROM weighted
            WHERE dur > 0;
        """)

        # Use psycopg SQL compositing properly (Identifier can't be interpolated with as_string(None) reliably)
        # Build again the query with proper formatting:
        q = sql.SQL(f"""
            WITH data AS (
                SELECT
                    ts,
                    {{col}} AS v,
                    LEAD(ts) OVER (ORDER BY ts) AS next_ts
                FROM env_samples
                WHERE ts >= %(start)s
                  AND ts <  %(end)s
                  AND status = 0
                  AND {{col}} IS NOT NULL
            ),
            weighted AS (
                SELECT
                    v,
                    GREATEST(
                        0,
                        LEAST(
                            %(step)s::double precision,
                            EXTRACT(EPOCH FROM (COALESCE(next_ts, %(end)s) - ts))
                        )
                    ) AS dur
                FROM data
            )
            SELECT
                CASE
                    WHEN SUM(dur) = 0 THEN NULL
                    ELSE ROUND((SUM(v * dur) / SUM(dur))::numeric, {self.OUT_DECIMALS})
                END AS avg
            FROM weighted
            WHERE dur > 0;
        """).format(col=col_ident)

        params = {"start": start, "end": end, "step": int(base_step_seconds)}
        with self.pool.connection() as conn:
            row = conn.execute(q, params).fetchone()

        if not row:
            return None
        return self._round_out(row[0])

# env_api.py
import os
from fastapi import FastAPI, HTTPException, APIRouter
from sensors import EnvSensors
from contextlib import asynccontextmanager

# ---- Load config from env vars; HTTP layer only routes ----
def _parse_float(envkey: str):
    v = os.getenv(envkey)
    if v is None: return None
    try:
        x = float(v)
        return x
    except ValueError:
        return None

BUS_NUM      = int(os.getenv("I2C_BUS", "2"))
SHT30_ADDR   = int(os.getenv("SHT30_ADDR", "0x44"), 16)
QMP_ADDR     = int(os.getenv("QMP_ADDR", "0x70"), 16)
POLL_SEC     = float(os.getenv("POLL_SEC", "1.0"))

svc = EnvSensors(
    bus_num=BUS_NUM,
    sht_addr=SHT30_ADDR,
    qmp_addr=QMP_ADDR,
    poll_sec=POLL_SEC,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    svc.start()
    yield

app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/v1")

@router.get("/metrics")
def metrics():
    m = svc.metrics()
    if not m["ok"]:
        raise HTTPException(status_code=503, detail=m["error"] or "sensor unavailable")
    # Remove ok/error fields so the response is metrics-only
    return {
        "temperature_c": m["temperature_c"],
        "humidity_pct": m["humidity_pct"],
        "pressure_hpa": m["pressure_hpa"],
        "qmp_temp_c": m["qmp_temp_c"],
        "ts": m["ts"],
    }

@router.get("/healthz")
def healthz():
    return svc.health()

app.include_router(router)

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("env_api:app", host="0.0.0.0", port=8080)
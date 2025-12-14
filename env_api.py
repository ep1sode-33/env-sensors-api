# env_api.py
import os, time, threading
from datetime import datetime, timezone
from smbus2 import SMBus
from fastapi import FastAPI, HTTPException

# ======== Configuration (can be overridden by environment variables) ========
BUS_NUM     = int(os.getenv("I2C_BUS", "2"))         # device on /dev/i2c-2
SHT30_ADDR  = int(os.getenv("SHT30_ADDR", "0x44"), 16)
QMP_PRI_ADDR= int(os.getenv("QMP_ADDR", "0x70"), 16) # fallback to 0x56 if 0x70 not found
POLL_SEC    = float(os.getenv("POLL_SEC", "1.0"))    # polling interval in seconds

# ======== Sensor Driver: SHT30 (Temperature & Humidity) ========
# Reference: https://sensirion.com/media/documents/213E6A3B/63A5A569/Datasheet_SHT3x_DIS.pdf
class SHT30:
    def __init__(self, bus: SMBus, addr=0x44):
        self.bus = bus
        self.addr = addr

    def read(self):
        # High repeatability, no clock stretching: command 0x2C06
        self.bus.write_i2c_block_data(self.addr, 0x2C, [0x06])
        time.sleep(0.015)
        data = self.bus.read_i2c_block_data(self.addr, 0x00, 6)
        t_raw = (data[0] << 8) | data[1]
        h_raw = (data[3] << 8) | data[4]
        # Datasheet formulas
        t_c = -45 + 175.0 * (t_raw / 65535.0)
        rh  = 100.0 * (h_raw / 65535.0)
        return round(t_c, 2), round(rh, 2)

# ======== Sensor Driver: QMP6988 (Pressure & Internal Temperature) ========
# Reference: https://datasheet4u.com/pdf-down/Q/M/P/QMP6988-QST.pdf
class QMP6988:
    REG_CHIP_ID   = 0xD1
    CHIP_ID       = 0x5C
    REG_RESET     = 0xE0
    REG_CONFIG    = 0xF1
    REG_CTRL      = 0xF4
    REG_PRESS_MSB = 0xF7
    REG_TEMP_MSB  = 0xFA
    CALIB_START   = 0xA0
    CALIB_LEN     = 25
    SUBTRACTOR    = 8388608  # 2^23

    def __init__(self, bus: SMBus, addr=0x70):
        self.bus = bus
        self.addr = None
        # Two common addresses: 0x70 (LOW) and 0x56 (HIGH)
        for a in (addr, 0x56):
            try:
                if self.bus.read_byte_data(a, self.REG_CHIP_ID) == self.CHIP_ID:
                    self.addr = a
                    break
            except Exception:
                pass
        if self.addr is None:
            raise RuntimeError("QMP6988 not found on 0x70/0x56")

        self._read_calib()

        # IIR=16x, temperature/pressure sampling=8x, Normal mode
        self._wr(self.REG_CONFIG, 0x04)
        ctrl = ((0x04 & 7) << 5) | ((0x04 & 7) << 2) | 0x03
        self._wr(self.REG_CTRL, ctrl)
        time.sleep(0.01)

    # ------ low level ------
    def _rd(self, reg, n):
        return self.bus.read_i2c_block_data(self.addr, reg, n)
    def _wr(self, reg, val):
        self.bus.write_byte_data(self.addr, reg, val)

    @staticmethod
    def _se_u20_to_s20(v):
        # 20-bit two's complement explicit sign extension (for a0/b00)
        return v - (1 << 20) if (v & (1 << 19)) else v
    @staticmethod
    def _s16(msb, lsb):
        v = (msb << 8) | lsb
        return v - 0x10000 if v & 0x8000 else v

    def _read_calib(self):
        c = self._rd(self.CALIB_START, self.CALIB_LEN)

        # a0, b00 are 20-bit signed, spanning bytes + nibble
        a0_u20  = (c[18] << 12) | (c[19] << 4) |  (c[24] & 0x0F)
        b00_u20 = (c[0]  << 12) | (c[1]  << 4) | ((c[24] & 0xF0) >> 4)
        self.COE_a0  = self._se_u20_to_s20(a0_u20)
        self.COE_b00 = self._se_u20_to_s20(b00_u20)

        self.COE_a1  = self._s16(c[20], c[21])
        self.COE_a2  = self._s16(c[22], c[23])
        self.COE_bt1 = self._s16(c[2],  c[3])
        self.COE_bt2 = self._s16(c[4],  c[5])
        self.COE_bp1 = self._s16(c[6],  c[7])
        self.COE_b11 = self._s16(c[8],  c[9])
        self.COE_bp2 = self._s16(c[10], c[11])
        self.COE_b12 = self._s16(c[12], c[13])
        self.COE_b21 = self._s16(c[14], c[15])
        self.COE_bp3 = self._s16(c[16], c[17])

        # Fixed-point parameters (consistent with common implementation)
        self.ik_a0  = self.COE_a0
        self.ik_b00 = self.COE_b00
        self.ik_a1  =  3608 * self.COE_a1 - 1731677965
        self.ik_a2  = 16889 * self.COE_a2 -   87619360
        self.ik_bt1 =   2982 * self.COE_bt1 +  107370906
        self.ik_bt2 = 329854 * self.COE_bt2 +  108083093
        self.ik_bp1 =  19923 * self.COE_bp1 + 1133836764
        self.ik_b11 =   2406 * self.COE_b11 +  118215883
        self.ik_bp2 =   3079 * self.COE_bp2 -  181579595
        self.ik_b12 =   6846 * self.COE_b12 +   85590281
        self.ik_b21 =  13836 * self.COE_b21 +   79333336
        self.ik_bp3 =   2915 * self.COE_bp3 +  157155561

    def _comp_temp_1o256C(self, dt):
        # dt: TEMP_TXD 24bit - 2^23
        wk1 = self.ik_a1 * dt
        wk2 = (self.ik_a2 * dt) >> 14
        wk2 = (wk2 * dt) >> 10
        wk2 = ((wk1 + wk2) // 32767) >> 19
        return (self.ik_a0 + wk2) >> 4  # Unit: 1/256 °C

    def _comp_press_1o16Pa(self, dp, tx_1o256C):
        tx = tx_1o256C
        wk1 = self.ik_bt1 * tx
        wk2 = (self.ik_bp1 * dp) >> 5
        wk1 += wk2
        wk2 = (self.ik_bt2 * tx) >> 1
        wk2 = (wk2 * tx) >> 8
        wk3 = wk2
        wk2 = (self.ik_b11 * tx) >> 4
        wk2 = (wk2 * dp) >> 1
        wk3 += wk2
        wk2 = (self.ik_bp2 * dp) >> 13
        wk2 = (wk2 * dp) >> 1
        wk3 += wk2
        wk1 += wk3 >> 14
        wk2 = self.ik_b12 * tx
        wk2 = (wk2 * tx) >> 22
        wk2 = (wk2 * dp) >> 1
        wk3 = wk2
        wk2 = (self.ik_b21 * tx) >> 6
        wk2 = (wk2 * dp) >> 23
        wk2 = (wk2 * dp) >> 1
        wk3 += wk2
        wk2 = (self.ik_bp3 * dp) >> 12
        wk2 = (wk2 * dp) >> 23
        wk2 = (wk2 * dp)
        wk3 += wk2
        wk1 += wk3 >> 15
        wk1 //= 32767
        wk1 >>= 11
        wk1 += self.ik_b00
        return wk1  # Unit: 1/16 Pa

    def read(self):
        t3 = self._rd(self.REG_TEMP_MSB, 3)
        p3 = self._rd(self.REG_PRESS_MSB, 3)
        t_adc = (t3[0] << 16) | (t3[1] << 8) | t3[2]
        p_adc = (p3[0] << 16) | (p3[1] << 8) | p3[2]
        dt = t_adc - self.SUBTRACTOR
        dp = p_adc - self.SUBTRACTOR
        tx_1o256C = self._comp_temp_1o256C(dt)
        p_1o16Pa  = self._comp_press_1o16Pa(dp, tx_1o256C)
        temp_c = tx_1o256C / 256.0
        press_pa = p_1o16Pa / 16.0
        return temp_c, press_pa

# ======== Service and Cache ========
app = FastAPI()
_bus = SMBus(BUS_NUM)
_sht = SHT30(_bus, SHT30_ADDR)
_qmp = None

_cache = {
    "temperature_c": None,
    "humidity_pct": None,
    "pressure_hpa": None,
    "qmp_temp_c": None,
    "ts": None,
    "ok": False,
    "error": None,
}
_lock = threading.Lock()

def _init_qmp():
    global _qmp
    _qmp = QMP6988(_bus, QMP_PRI_ADDR)

def _sampler():
    global _qmp
    # QMP may fail to initialize, retry in loop
    while _qmp is None:
        try:
            _init_qmp()
        except Exception as e:
            with _lock:
                _cache.update(ok=False, error=f"QMP init: {e}")
            time.sleep(1.0)

    while True:
        try:
            t_sht, rh = _sht.read()
            t_qmp, p_pa = _qmp.read()
            now = datetime.now(timezone.utc).isoformat()
            with _lock:
                _cache.update(
                    temperature_c=t_sht,
                    humidity_pct=rh,
                    pressure_hpa=round(p_pa / 100.0, 2),
                    qmp_temp_c=round(t_qmp, 2),
                    ts=now,
                    ok=True,
                    error=None,
                )
        except Exception as e:
            # Read failed: mark error and attempt to reinitialize QMP
            with _lock:
                _cache.update(ok=False, error=str(e))
            try:
                _init_qmp()
            except Exception:
                pass
        time.sleep(POLL_SEC)

@app.on_event("startup")
def _start():
    threading.Thread(target=_sampler, daemon=True).start()

@app.get("/metrics")
def metrics():
    with _lock:
        if not _cache["ok"]:
            raise HTTPException(status_code=503, detail=_cache["error"] or "sensor unavailable")
        # Return a copy to prevent external modification
        return {
            "temperature_c": _cache["temperature_c"],  # Default to SHT30 temperature
            "humidity_pct": _cache["humidity_pct"],
            "pressure_hpa": _cache["pressure_hpa"],
            "qmp_temp_c": _cache["qmp_temp_c"],        # Internal temperature, for reference
            "ts": _cache["ts"],
        }

@app.get("/healthz")
def healthz():
    with _lock:
        return {"ok": _cache["ok"], "ts": _cache["ts"], "error": _cache["error"]}

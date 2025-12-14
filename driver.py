# driver.py
# Hardware-only driver for SHT30 + QMP6988 on I2C.
# Responsibilities:
#   - Initialize I2C bus and sensors
#   - Read a single sample on demand (no caching, no scheduling, no timestamps)
#   - Provide minimal health status (ok/error)
#
# References:
#   - QMP6988: https://datasheet4u.com/pdf-down/Q/M/P/QMP6988-QST.pdf
#   - SHT3x : https://sensirion.com/media/documents/213E6A3B/63A5A569/Datasheet_SHT3x_DIS.pdf

from __future__ import annotations

import os
import time
import threading
from typing import Optional, Dict, Any

from smbus2 import SMBus


# -------------------- SHT30 --------------------
class SHT30:
    def __init__(self, bus: SMBus, addr: int = 0x44):
        self.bus = bus
        self.addr = addr

    def read(self) -> tuple[float, float]:
        """
        Returns: (temperature_c, humidity_pct)
        """
        # High repeatability, no clock stretching: command 0x2C06
        self.bus.write_i2c_block_data(self.addr, 0x2C, [0x06])
        time.sleep(0.015)
        data = self.bus.read_i2c_block_data(self.addr, 0x00, 6)

        t_raw = (data[0] << 8) | data[1]
        h_raw = (data[3] << 8) | data[4]

        t_c = -45.0 + 175.0 * (t_raw / 65535.0)
        rh = 100.0 * (h_raw / 65535.0)

        return (t_c, rh)


# -------------------- QMP6988 --------------------
class QMP6988:
    REG_CHIP_ID = 0xD1
    CHIP_ID = 0x5C

    REG_CONFIG = 0xF1
    REG_CTRL = 0xF4
    REG_PRESS_MSB = 0xF7
    REG_TEMP_MSB = 0xFA

    CALIB_START = 0xA0
    CALIB_LEN = 25

    SUBTRACTOR = 8388608  # 2^23

    def __init__(self, bus: SMBus, addr: int = 0x70):
        self.bus = bus
        self.addr: Optional[int] = None

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

        # IIR=16x, temperature/pressure oversampling=8x, normal mode
        self._wr(self.REG_CONFIG, 0x04)
        ctrl = ((0x04 & 7) << 5) | ((0x04 & 7) << 2) | 0x03
        self._wr(self.REG_CTRL, ctrl)
        time.sleep(0.01)

    def _rd(self, reg: int, n: int) -> list[int]:
        assert self.addr is not None
        return self.bus.read_i2c_block_data(self.addr, reg, n)

    def _wr(self, reg: int, val: int) -> None:
        assert self.addr is not None
        self.bus.write_byte_data(self.addr, reg, val)

    @staticmethod
    def _se_u20_to_s20(v: int) -> int:
        # 20-bit two's complement sign extension
        return v - (1 << 20) if (v & (1 << 19)) else v

    @staticmethod
    def _s16(msb: int, lsb: int) -> int:
        v = (msb << 8) | lsb
        return v - 0x10000 if (v & 0x8000) else v

    def _read_calib(self) -> None:
        c = self._rd(self.CALIB_START, self.CALIB_LEN)

        # a0, b00 are 20-bit signed spanning bytes + nibble
        a0_u20 = (c[18] << 12) | (c[19] << 4) | (c[24] & 0x0F)
        b00_u20 = (c[0] << 12) | (c[1] << 4) | ((c[24] & 0xF0) >> 4)

        self.COE_a0 = self._se_u20_to_s20(a0_u20)
        self.COE_b00 = self._se_u20_to_s20(b00_u20)

        self.COE_a1 = self._s16(c[20], c[21])
        self.COE_a2 = self._s16(c[22], c[23])
        self.COE_bt1 = self._s16(c[2], c[3])
        self.COE_bt2 = self._s16(c[4], c[5])
        self.COE_bp1 = self._s16(c[6], c[7])
        self.COE_b11 = self._s16(c[8], c[9])
        self.COE_bp2 = self._s16(c[10], c[11])
        self.COE_b12 = self._s16(c[12], c[13])
        self.COE_b21 = self._s16(c[14], c[15])
        self.COE_bp3 = self._s16(c[16], c[17])

        # Fixed-point parameters (common implementation style)
        self.ik_a0 = self.COE_a0
        self.ik_b00 = self.COE_b00
        self.ik_a1 = 3608 * self.COE_a1 - 1731677965
        self.ik_a2 = 16889 * self.COE_a2 - 87619360
        self.ik_bt1 = 2982 * self.COE_bt1 + 107370906
        self.ik_bt2 = 329854 * self.COE_bt2 + 108083093
        self.ik_bp1 = 19923 * self.COE_bp1 + 1133836764
        self.ik_b11 = 2406 * self.COE_b11 + 118215883
        self.ik_bp2 = 3079 * self.COE_bp2 - 181579595
        self.ik_b12 = 6846 * self.COE_b12 + 85590281
        self.ik_b21 = 13836 * self.COE_b21 + 79333336
        self.ik_bp3 = 2915 * self.COE_bp3 + 157155561

    def _comp_temp_1o256C(self, dt: int) -> int:
        # dt: TEMP_TXD 24bit - 2^23
        wk1 = self.ik_a1 * dt
        wk2 = (self.ik_a2 * dt) >> 14
        wk2 = (wk2 * dt) >> 10
        wk2 = ((wk1 + wk2) // 32767) >> 19
        return (self.ik_a0 + wk2) >> 4  # unit: 1/256 °C

    def _comp_press_1o16Pa(self, dp: int, tx_1o256C: int) -> int:
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
        return wk1  # unit: 1/16 Pa

    def read(self) -> tuple[float, float]:
        """
        Returns: (qmp_temp_c, pressure_pa)
        """
        t3 = self._rd(self.REG_TEMP_MSB, 3)
        p3 = self._rd(self.REG_PRESS_MSB, 3)

        t_adc = (t3[0] << 16) | (t3[1] << 8) | t3[2]
        p_adc = (p3[0] << 16) | (p3[1] << 8) | p3[2]

        dt = t_adc - self.SUBTRACTOR
        dp = p_adc - self.SUBTRACTOR

        tx_1o256C = self._comp_temp_1o256C(dt)
        p_1o16Pa = self._comp_press_1o16Pa(dp, tx_1o256C)

        temp_c = tx_1o256C / 256.0
        press_pa = p_1o16Pa / 16.0

        return (temp_c, press_pa)


# -------------------- Driver Wrapper (no time, no caching) --------------------
class EnvDriver:
    """
    Hardware wrapper:
      - Owns the I2C bus handle
      - Owns sensor instances
      - Reads once on demand
    """

    def __init__(
        self,
        bus_num: int = 2,
        sht_addr: int = 0x44,
        qmp_addr: int = 0x70,
    ):
        self._lock = threading.Lock()  # serialize I2C transactions (safe if called concurrently)

        self.bus_num = bus_num
        self.sht_addr = sht_addr
        self.qmp_addr_primary = qmp_addr

        self.bus: Optional[SMBus] = None
        self.sht: Optional[SHT30] = None
        self.qmp: Optional[QMP6988] = None

        self._ok: bool = False
        self._error: Optional[str] = None

        self._open()

    def _open(self) -> None:
        try:
            self.bus = SMBus(self.bus_num)
            self.sht = SHT30(self.bus, self.sht_addr)
            self._init_qmp()
            self._ok = True
            self._error = None
        except Exception as e:
            self._ok = False
            self._error = f"init failed: {e}"
            # best-effort close
            try:
                if self.bus is not None:
                    self.bus.close()
            except Exception:
                pass
            self.bus = None
            self.sht = None
            self.qmp = None

    def _init_qmp(self) -> None:
        assert self.bus is not None
        self.qmp = QMP6988(self.bus, self.qmp_addr_primary)

    def metrics(self) -> Dict[str, Any]:
        """
        Read once immediately (no caching, no timestamps).
        Returns:
          {
            ok: bool,
            error: str|None,
            temperature_c: float|None,
            humidity_pct: float|None,
            pressure_hpa: float|None,
            qmp_temp_c: float|None
          }
        """
        with self._lock:
            if not self._ok or self.bus is None or self.sht is None:
                return {
                    "ok": False,
                    "error": self._error or "driver not initialized",
                    "temperature_c": None,
                    "humidity_pct": None,
                    "pressure_hpa": None,
                    "qmp_temp_c": None,
                }

            try:
                # SHT30
                t_sht, rh = self.sht.read()

                # QMP6988 (re-init on failure)
                if self.qmp is None:
                    self._init_qmp()
                t_qmp, p_pa = self.qmp.read()

                p_hpa = p_pa / 100.0

                self._ok = True
                self._error = None
                return {
                    "ok": True,
                    "error": None,
                    "temperature_c": round(float(t_sht), 2),
                    "humidity_pct": round(float(rh), 2),
                    "pressure_hpa": round(float(p_hpa), 2),
                    "qmp_temp_c": round(float(t_qmp), 2),
                }

            except Exception as e:
                self._ok = False
                self._error = str(e)

                # Try to recover QMP on read errors (common failure mode)
                try:
                    if self.bus is not None:
                        self._init_qmp()
                        self._ok = True
                        self._error = None
                except Exception:
                    pass

                return {
                    "ok": False,
                    "error": self._error,
                    "temperature_c": None,
                    "humidity_pct": None,
                    "pressure_hpa": None,
                    "qmp_temp_c": None,
                }

    def healthz(self) -> Dict[str, Any]:
        """
        No sampling, no timestamps. Just current driver init/read status.
        """
        return {"ok": bool(self._ok), "error": self._error}

    def close(self) -> None:
        with self._lock:
            try:
                if self.bus is not None:
                    self.bus.close()
            finally:
                self.bus = None
                self.sht = None
                self.qmp = None
                self._ok = False
                self._error = "closed"


# -------------------- Module-level two functions (as requested) --------------------
_DEFAULT: Optional[EnvDriver] = None


def _parse_int_env(key: str, default: str) -> int:
    # base=0 allows "0x44" or "68"
    return int(os.getenv(key, default), 0)


def _get_default() -> EnvDriver:
    global _DEFAULT
    if _DEFAULT is None:
        bus = _parse_int_env("I2C_BUS", "2")
        sht = _parse_int_env("SHT30_ADDR", "0x44")
        qmp = _parse_int_env("QMP_ADDR", "0x70")
        _DEFAULT = EnvDriver(bus_num=bus, sht_addr=sht, qmp_addr=qmp)
    return _DEFAULT


def metrics() -> Dict[str, Any]:
    """
    Public function: read once immediately.
    """
    return _get_default().metrics()


def healthz() -> Dict[str, Any]:
    """
    Public function: driver health only (no sampling).
    """
    return _get_default().healthz()

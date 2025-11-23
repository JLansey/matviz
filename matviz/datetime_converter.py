# filename: datetime_converter.py
from __future__ import annotations

import datetime as _dt
from typing import Optional

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

_UTC = _dt.timezone.utc
_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_UTC)

class DateCodec:
    """
    Single-value, reversible timestamp <-> integer nanoseconds codec.

    - to_number(ts) -> int (ns since UNIX epoch, UTC)
    - from_number(ns) -> same type as input (datetime | np.datetime64 | pd.Timestamp)
      and same tz-awareness:
        * Python datetime: tz-aware restored to original zone if available; else naive
        * pandas Timestamp: tz-aware restored to original zone if available; else naive
        * numpy.datetime64: tz-naive (as original)
    """

    def __init__(self):
        # minimal state for round-trip
        self._kind: Optional[str] = None       # 'py' | 'np' | 'pd'
        self._aware: bool = False              # original tz-aware?
        self._tz_key: Optional[str] = None     # zone name (e.g. 'America/New_York'), if any
        self._tzinfo: Optional[_dt.tzinfo] = None  # fallback tzinfo object (for Python datetime)

    def reset(self) -> None:
        self._kind = None
        self._aware = False
        self._tz_key = None
        self._tzinfo = None

    # ---------- encode ----------
    def to_number(self, ts) -> int:
        """Accepts: datetime, np.datetime64, or pd.Timestamp. Returns int nanoseconds since epoch (UTC)."""
        if isinstance(ts, _dt.datetime):
            self._kind = "py"
            self._aware = ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None
            if self._aware:
                self._tzinfo = ts.tzinfo
                self._tz_key = getattr(ts.tzinfo, "key", None) or getattr(ts.tzinfo, "zone", None)
                dt_utc = ts.astimezone(_UTC)
            else:
                self._tzinfo = None
                self._tz_key = None
                # treat naive as UTC wall time
                dt_utc = ts.replace(tzinfo=_UTC)
            return self._dt_to_ns(dt_utc)

        if isinstance(ts, pd.Timestamp):
            self._kind = "pd"
            t = pd.Timestamp(ts)
            self._aware = t.tz is not None
            if self._aware:
                self._tz_key = getattr(t.tz, "key", None) or getattr(t.tz, "zone", None)
                ns = int(t.tz_convert("UTC").value)  # UTC ns
            else:
                self._tz_key = None
                ns = int(t.tz_localize("UTC").value)  # treat naive as UTC wall time
            return ns

        if isinstance(ts, np.datetime64):
            self._kind = "np"
            self._aware = False
            self._tz_key = None
            self._tzinfo = None
            return int(np.datetime64(ts, "ns").astype("int64"))

        raise TypeError(f"Unsupported type: {type(ts)}")

    # ---------- decode ----------
    def from_number(self, ns: int):
        """Return the same type (datetime | np.datetime64 | pd.Timestamp) as the original input."""
        if self._kind is None:
            raise RuntimeError("No state saved. Call to_number(...) first.")

        if self._kind == "py":
            dt_utc = self._ns_to_dt(ns)  # aware UTC
            if not self._aware:
                return dt_utc.replace(tzinfo=None)
            # prefer named zone if available
            if self._tz_key and ZoneInfo is not None:
                try:
                    return dt_utc.astimezone(ZoneInfo(self._tz_key))
                except Exception:
                    pass
            # otherwise use original tzinfo if we have it
            if self._tzinfo is not None:
                try:
                    return dt_utc.astimezone(self._tzinfo)
                except Exception:
                    return dt_utc
            return dt_utc

        if self._kind == "pd":
            ts = pd.to_datetime(ns, unit="ns", utc=True)
            if self._aware:
                if self._tz_key:
                    try:
                        return ts.tz_convert(self._tz_key)
                    except Exception:
                        return ts
                return ts  # keep UTC-aware if we can't resolve the zone
            else:
                return ts.tz_convert(None)  # return naive

        if self._kind == "np":
            return np.datetime64(int(ns), "ns")

        raise AssertionError("unreachable")

    # ---------- helpers ----------
    @staticmethod
    def _dt_to_ns(dt_utc: _dt.datetime) -> int:
        """Python datetime (tz-aware UTC) -> integer nanoseconds since epoch."""
        # Avoid float timestamps for reversibility. Python datetime has microsecond resolution.
        delta = dt_utc - _EPOCH
        # delta only has microseconds; represent as ns with trailing zeros
        return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000

    @staticmethod
    def _ns_to_dt(ns: int) -> _dt.datetime:
        """Integer nanoseconds since epoch -> Python datetime in UTC (aware)."""
        seconds, rem_ns = divmod(int(ns), 1_000_000_000)
        micro, _ = divmod(rem_ns, 1_000)  # Python datetime supports microseconds (not full ns)
        return _EPOCH + _dt.timedelta(seconds=seconds, microseconds=micro)

"""usage:
codec = DateCodec()
num_series = date_series.apply(codec.to_number)
date_series_again = num_series.apply(codec.from_number)
"""
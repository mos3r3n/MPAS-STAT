"""Date, cycle and forecast-lead helpers shared by every pipeline.
"""
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Date / cycle helpers
# ---------------------------------------------------------------------------
def yyyymmddhh_to_datetime(date_str):
    """Parse a ``YYYYMMDDHH`` cycle string."""
    return datetime.strptime(str(date_str), "%Y%m%d%H")


def datetime_to_cycle(dt):
    """Format a datetime as a ``YYYYMMDDHH`` cycle directory name."""
    return dt.strftime("%Y%m%d%H")


def datetime_to_datefile(dt):
    """Format a datetime as the MPAS file-name date, ``YYYY-MM-DD_HH.00.00``."""
    return dt.strftime("%Y-%m-%d_%H.00.00")


def generate_cycles(start_date, end_date, interval_hours):
    """Cycles from ``start_date`` to ``end_date`` (``YYYYMMDDHH``, inclusive) every ``interval_hours``."""
    start_dt = yyyymmddhh_to_datetime(start_date)
    end_dt   = yyyymmddhh_to_datetime(end_date)
    cycles, dt = [], start_dt
    while dt <= end_dt:
        cycles.append(dt)
        dt += timedelta(hours=int(interval_hours))
    return cycles


def generate_forecast_leads(cfg):
    """Return the forecast lead hours (int) to verify.

    Read from ``cfg["forecast"]``:

    ``length``
        Maximum lead hour (required).
    ``interval``
        Spacing between leads in hours (required).
    ``start``
        First lead hour (default: ``interval``; 0 includes the analysis time).
    """
    fcfg     = cfg.get("forecast", {})
    length   = int(fcfg["length"])
    interval = int(fcfg["interval"])
    start    = int(fcfg.get("start", interval))

    leads, lead = [], start
    while lead <= length:
        leads.append(lead)
        lead += interval
    return leads

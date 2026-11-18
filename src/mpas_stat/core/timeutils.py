"""Date, cycle and forecast-lead helpers shared by every pipeline.
"""
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Date / cycle helpers
# ---------------------------------------------------------------------------
def yyyymmddhh_to_datetime(date_str):
    return datetime.strptime(str(date_str), "%Y%m%d%H")


def datetime_to_cycle(dt):
    return dt.strftime("%Y%m%d%H")


def datetime_to_datefile(dt):
    return dt.strftime("%Y-%m-%d_%H.00.00")


def generate_cycles(start_date, end_date, interval_hours):
    start_dt = yyyymmddhh_to_datetime(start_date)
    end_dt   = yyyymmddhh_to_datetime(end_date)
    cycles, dt = [], start_dt
    while dt <= end_dt:
        cycles.append(dt)
        dt += timedelta(hours=int(interval_hours))
    return cycles


def generate_forecast_leads(cfg):
    """Return forecast lead hours (int) for extended forecast verification.

    Controlled by cfg["forecast"]:
        length   : maximum forecast lead hour (required)
        interval : lead-time spacing in hours (required)
        start    : first lead hour (default = interval; use 0 to include
                   the initialization/analysis time itself)
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

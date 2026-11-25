"""Variable and statistic labels, in the style of NCAR mpas-jedi-graphics.

* Panel titles carry the variable and its units in plain text:
  ``T (K)``, ``U (m/s)``, ``Q (g/kg)``, ``Ps (Pa)``, ``RH (%)``.
* Axis / colour-bar labels carry the statistic, lower case as in NCAR's
  publication conventions: ``rms``, ``mean``, ``STD``, ``Count``,
  ``spread``; percent differences read ``rms: % diff. from Control``.

Units come from the variable tables (model: mpas_stat.model.variables,
obs: mpas_stat.obs.sensor_config). Older LaTeX strings there are
converted, so every figure uses the same plain notation.
"""
import re

__all__ = ["plain_units", "var_label", "stat_label", "pct_diff_label", "cfg_var_label"]

_EXACT = {
    r"$\mathrm{m \cdot s^{-1}}$": "m/s",
    r"$\mathrm{m\,s^{-1}}$": "m/s",
    r"$\mathrm{g \cdot kg^{-1}}$": "g/kg",
    r"$\mathrm{g\,kg^{-1}}$": "g/kg",
    r"$\mathrm{kg\,m^{-3}}$": "kg/m³",
    r"$^\circ\mathrm{C}$": "°C",
    r"$^\circ$": "°",
    r"$\%$": "%",
}

_STATS = {
    "rms": "rms", "rmse": "rms",
    "bias": "mean", "mean": "mean",
    "std": "STD",
    "count": "Count",
    "spread": "spread",
    "totalspread": "total spread", "totspread": "total spread",
}


def plain_units(units):
    """Plain-text units ("m/s", "K", "g/kg", "°", "%"); "" if none."""
    if units is None:
        return ""
    u = str(units).strip()
    if u in _EXACT:
        return _EXACT[u]
    if not u.startswith("$"):
        return u
    # generic LaTeX clean-up: \mathrm{...}, ^\circ, x^{-1} -> /x
    t = u.strip("$")
    t = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", t)
    t = t.replace(r"^\circ", "°").replace(r"\%", "%").replace(r"\cdot", " ").replace(r"\,", " ")
    parts = t.split()
    num = [p for p in parts if not p.endswith("^{-1}")]
    den = [p[:-len("^{-1}")] for p in parts if p.endswith("^{-1}")]
    out = " ".join(num) + ("/" + "/".join(den) if den else "")
    return out.replace("{", "").replace("}", "").strip()


def var_label(name, units=None):
    """Return the panel title for a variable, such as ``T (K)``.

    Just the name when there are no units.
    """
    u = plain_units(units)
    return f"{name} ({u})" if u else str(name)


def cfg_var_label(name, cfg):
    """Return :func:`var_label` with the units from ``cfg["variable_units"]``."""
    return var_label(name, (cfg or {}).get("variable_units", {}).get(name))


def stat_label(stat):
    """Return the axis label for a statistic: rms, mean, STD, Count, spread."""
    return _STATS.get(str(stat).lower().replace(" ", ""), str(stat))


def pct_diff_label(stat, ref):
    """Return the percent-difference axis label.

    For example ``rms: % diff. from Control``.
    """
    return f"{stat_label(stat)}: % diff. from {ref}"

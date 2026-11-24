"""Experiment lists, colours and graphic cfg dicts for obs-space figures.
"""


# Line colours assigned to experiments by their position in
# config["experiments"]["names"] (shared by every obs-space figure type, so
# one experiment keeps the same colour across products).
EXPERIMENT_PALETTE = [
    'black', 'blue', 'red', 'green', 'orange', 'brown', 'cyan', 'purple',
    'magenta', 'olive', 'teal', 'darkred', 'gold', 'navy', 'darkgreen'
]


def _build_pseudo_experiments(config):
    """
    MPAS-STAT's graphic functions plot one line/column per "experiment".
    Obs-space verification additionally distinguishes OMB (background) vs.
    OMA (analysis) *within* one experiment/cycle.  To reuse the graphic
    functions unmodified, each (experiment, OMB|OMA) combination is exposed
    to them as its own pseudo-experiment -- "<exp> - OMB" (solid) and
    "<exp> - OMA" (dashed) -- sharing the experiment's colour, matching the
    solid/dashed OMB/OMA convention of the original plots.

    Only "da"/"hofx" experiments contribute -- "hofx_forecast" experiments
    have no cycling diagnosis files to read (see _build_forecast_experiments()
    / generate_forecast_lead_graphics() for their own OMF product instead).

    Returns (names, labels, colors, linestyles, index_map) where
    index_map[j] = (exp_name, "omb"|"oma") for pseudo-experiment j.
    """
    plot_oma = config["plot_settings"].get("plot_oma", True)
    exp_names = config["experiments"]["names"]
    exp_labels = config["experiments"].get("labels", exp_names)
    exp_types = config["experiments"].get("types", ["da"] * len(exp_names))
    palette = EXPERIMENT_PALETTE

    names, labels, colors, linestyles, index_map = [], [], [], [], []
    for iexp, (name, label, etype) in enumerate(zip(exp_names, exp_labels, exp_types)):
        if etype == "hofx_forecast":
            continue

        color = palette[iexp % len(palette)]

        names.append(f"{name}__omb")
        labels.append(f"{label} - OMB")
        colors.append(color)
        linestyles.append("-")
        index_map.append((name, "omb"))

        if etype == "da" and plot_oma:
            names.append(f"{name}__oma")
            labels.append(f"{label} - OMA")
            colors.append(color)
            linestyles.append("--")
            index_map.append((name, "oma"))

    return names, labels, colors, linestyles, index_map


def _build_forecast_experiments(config):
    """The subset of configured experiments with type == "hofx_forecast",
    in cfg["experiments"] shape for plot_forecast_lead_0d. Unlike
    _build_pseudo_experiments(), no OMB/OMA doubling is needed here -- OMF
    is the only diagtype an extended-forecast HofX run produces, so each
    "hofx_forecast" experiment maps to exactly one line.

    Returns (names, labels, colors, linestyles) where `names` are the real
    experiment names (no pseudo-experiment renaming)."""
    exp_names = config["experiments"]["names"]
    exp_labels = config["experiments"].get("labels", exp_names)
    exp_types = config["experiments"].get("types", ["da"] * len(exp_names))
    palette = EXPERIMENT_PALETTE

    names, labels, colors = [], [], []
    for iexp, (name, label, etype) in enumerate(zip(exp_names, exp_labels, exp_types)):
        if etype != "hofx_forecast":
            continue
        names.append(name)
        labels.append(label)
        colors.append(palette[iexp % len(palette)])

    linestyles = ["-"] * len(names)
    return names, labels, colors, linestyles


def _experiments_for_diagtype(config, diagtype):
    """The real (non-pseudo) "da"/"hofx" experiments that contribute to one
    diagtype ("omb" or "oma") -- every da/hofx experiment has OMB, but only
    "da"-typed experiments (with plot_oma) have OMA. Used by the 2-D
    comparison products (lat-lon map, lev-time, lev-lat), which put the
    REAL experiment on the panel axis instead of MPAS-STAT's
    OMB/OMA-pseudo-experiment trick -- see _build_pseudo_experiments()'s
    docstring for why 0D/1D products still need that trick and these don't.

    Returns (names, labels, colors) -- experiments.names[0] among these is
    the comparison reference (absolute/full-state) for whichever of these
    figures enable a relative-reduction statistic."""
    exp_names = config["experiments"]["names"]
    exp_labels = config["experiments"].get("labels", exp_names)
    exp_types = config["experiments"].get("types", ["da"] * len(exp_names))
    plot_oma = config["plot_settings"].get("plot_oma", True)
    palette = EXPERIMENT_PALETTE

    names, labels, colors = [], [], []
    for iexp, (name, label, etype) in enumerate(zip(exp_names, exp_labels, exp_types)):
        if etype == "hofx_forecast":
            continue
        if diagtype == "oma" and not (etype == "da" and plot_oma):
            continue
        names.append(name)
        labels.append(label)
        colors.append(palette[iexp % len(palette)])
    return names, labels, colors


def _comparison_graphic_cfg(config, out_dir, names, labels, colors):
    """cfg dict for the 2-D REAL-experiment comparison products (lat-lon
    map, lev-time, lev-lat) -- unlike _base_graphic_cfg(), enables
    relative_reduction_statistics so experiments.names[0] draws as the
    absolute reference and every other experiment draws as a bootstrap %
    reduction vs. it, exactly like the model-space package's own 2-D
    figures. A no-op visually when `names` has fewer than 2 entries (there
    is nothing to reduce against)."""
    comparison_stats = config["plot_settings"].get(
        "comparison_relative_reduction_statistics", ["RMS", "STD"])
    return {
        "variables": [],  # filled in per call site
        "experiments": {
            "names": names,
            "labels": labels,
            "colors": colors,
            "linestyles": ["-"] * len(names),
        },
        "paths": {"out_dir": out_dir},
        "figure_format": config["plot_settings"].get("figure_format", "png"),
        "max_time_ticks": config["plot_settings"].get("max_time_ticks", 12),
        "relative_reduction_statistics": comparison_stats,
        "bootstrap": config.get("bootstrap", {}),
        **_ci_display_settings(config),
    }


def _profile_axis_meta(bin_method):
    """(coord_label, invert_y) for a sensor's vertical/latitude coordinate --
    shared by generate_profile_graphics, generate_time_level_graphics, and
    generate_lat_vert_graphics so all three agree on axis labelling."""
    if bin_method == 1:
        return "Pressure (hPa)", True
    if bin_method == 2:
        return "Height (km)", False
    return "Latitude", False


def _ci_display_settings(config):
    """plot_settings.ci_style / ci_alpha, passed to the shared plot functions
    (confidence intervals as "shade", "errorbar" or "none")."""
    ps = config.get("plot_settings", {})
    return {k: ps[k] for k in ("ci_style", "ci_alpha") if k in ps}


def _base_graphic_cfg(config, out_dir):
    """cfg dict consumed by mpas_stat.graphics.functions.plot_*."""
    names, labels, colors, linestyles, index_map = _build_pseudo_experiments(config)
    cfg = {
        "variables": [],  # filled in per call site
        "experiments": {
            "names": names,
            "labels": labels,
            "colors": colors,
            "linestyles": linestyles,
        },
        "paths": {"out_dir": out_dir},
        "figure_format": config["plot_settings"].get("figure_format", "png"),
        "max_time_ticks": config["plot_settings"].get("max_time_ticks", 12),
        # OMB vs. OMA are plotted as absolute values on the same panel (like
        # the original script), not as a percent-difference product -- so
        # disable MPAS-STAT's relative/percent-reduction plotting mode.
        "relative_reduction_statistics": [],
    }
    cfg.update(_ci_display_settings(config))
    return cfg, index_map

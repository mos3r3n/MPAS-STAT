"""Correctness tests for the weighted/pooled statistics (2026-11 revision).

Each corrected statistic is checked against a brute-force computation on the
raw samples:

* model space: area- and mass-weighted pooling, RMS over the column/bin for
  2-D products, sqrt(mean variance) spread, unbiased + finite-ensemble spread;
* cycle aggregation: pooled mean/RMS/STD/spread equal the statistic of all
  cycles' samples together;
* obs space: one common sample for OMB/OMA and across experiments, QC flag 19
  passing everywhere, spread definition, dateline-crossing domains, GNSS-RO
  bending angle, and the legacy switch.

Run from the repository root with::

    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import netCDF4 as nc4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpas_stat.core.grouping import grouped_weighted_stats  # noqa: E402
from mpas_stat.core.pooling import (  # noqa: E402
    bootstrap_pool_reduction_ci,
    pool_from_rows,
    pooled_value,
    read_components,
)
from mpas_stat.model.driver import _finalize_accumulator, layer_mass_weights  # noqa: E402
from mpas_stat.model.kernels import calculate_statistics_one_task_weighted  # noqa: E402
from mpas_stat.obs.binning import _apply_domain_mask, compute_stats_2d_latlon  # noqa: E402
from mpas_stat.obs.matching import restrict_to_common_sample  # noqa: E402
from mpas_stat.obs.reader import _ensemble_spread, load_observation_data  # noqa: E402

RNG = np.random.default_rng(42)


def wstats(x, w):
    x, w = np.asarray(x, float).ravel(), np.broadcast_to(w, np.shape(x)).ravel()
    ok = np.isfinite(x)
    x, w = x[ok], w[ok]
    mean = np.sum(w * x) / np.sum(w)
    return mean, np.sqrt(np.sum(w * x * x) / np.sum(w)), np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w))


class TestGroupedWeightedStats(unittest.TestCase):
    def test_matches_brute_force(self):
        x = RNG.standard_normal((50, 6)) + 0.3
        x[3, 2] = np.nan
        w = RNG.uniform(0.5, 2.0, (50, 1))
        g = RNG.integers(0, 4, (50, 1))
        out = grouped_weighted_stats(x, w, g, 4, spread=np.abs(x))
        for k in range(4):
            sel = np.broadcast_to(g == k, x.shape)
            ww = np.broadcast_to(w, x.shape)[sel]
            mean, rms, std = wstats(x[sel], ww)
            self.assertAlmostEqual(out["mean"][k], mean)
            self.assertAlmostEqual(out["rms"][k], rms)
            self.assertAlmostEqual(out["std"][k], std)
            ok = np.isfinite(x[sel])
            self.assertAlmostEqual(out["weight"][k], ww[ok].sum())
            self.assertAlmostEqual(out["spread"][k], wstats(np.abs(x[sel]), ww)[1])

    def test_empty_group_is_nan(self):
        out = grouped_weighted_stats(np.array([1.0]), 1.0, np.array([0]), 2)
        self.assertTrue(np.isnan(out["rms"][1]))
        self.assertEqual(out["weight"][1], 0.0)


class TestModelWeightedKernels(unittest.TestCase):
    def setUp(self):
        nc, nl = 300, 8
        self.diff = RNG.standard_normal((nc, nl)) * np.linspace(1, 3, nl)
        self.spread = np.abs(RNG.standard_normal((nc, nl)))
        self.area = RNG.uniform(0.2, 3.0, nc)
        p = np.linspace(1000.0, 100.0, nl)[None, :] * RNG.uniform(0.95, 1.0, (nc, 1))
        self.lw = layer_mass_weights(p, SimpleNamespace(mass_weighting=True))
        lat = RNG.uniform(-60, 60, nc)
        lat_bins = np.arange(-60.0, 61.0, 30.0)
        lon_bins = np.arange(0.0, 361.0, 120.0)
        lat_idx = np.clip(np.digitize(lat, lat_bins) - 1, 0, 3)
        lon_idx = RNG.integers(0, 3, nc)
        self.meta = {
            "lat": lat, "lat_bins": lat_bins, "lon_bins": lon_bins, "lat_idx": lat_idx,
            "group_ids_latlon": lat_idx * 3 + lon_idx, "level_bins": np.array([0, 4, 8]),
            "cell_weight": self.area, "level_weight": self.lw,
        }
        self.cfg = SimpleNamespace(regions={"Global": (-90, 90), "NH": (0, 90)},
                                   hgt_levels=[1.0, 2.0], pres_levels=[850.0, 500.0])
        self.prepared = {"diff": self.diff, "fcst_sprd": self.spread, "is_3d": True,
                         "diff_hgt": self.diff[:, :2], "sprd_hgt": None,
                         "diff_pres": self.diff[:, :2], "sprd_pres": None}

    def task(self, name):
        return calculate_statistics_one_task_weighted("T", name, self.prepared, self.meta, self.cfg)[2]

    def test_mass_weights_sum_to_column_pressure(self):
        p = np.array([[1000.0, 900.0, 800.0]])
        dp = layer_mass_weights(p, SimpleNamespace(mass_weighting=True))
        np.testing.assert_allclose(dp * np.mean([100, 100, 100]), [[100, 100, 100]])

    def test_lat_lon_rms_is_column_rms_not_rms_of_column_mean(self):
        box = self.task("2D_lat_lon")["lat_lon"]
        gid = self.meta["group_ids_latlon"]
        k = int(np.flatnonzero(np.bincount(gid, minlength=12))[0])
        cells = gid == k
        w = self.area[cells, None] * self.lw[cells]
        mean, rms, std = wstats(self.diff[cells], w)
        r, c = divmod(k, 3)
        self.assertAlmostEqual(box["rms"][r, c], rms)
        self.assertAlmostEqual(box["std"][r, c], std)
        self.assertAlmostEqual(box["spread"][r, c], wstats(self.spread[cells], w)[1])
        # The old definition (RMS of the column-mean error) is much smaller.
        col_mean = np.mean(self.diff[cells], axis=1)
        self.assertGreater(box["rms"][r, c], 2 * np.sqrt(np.mean(col_mean ** 2)))

    def test_regional_0d_and_profiles(self):
        nh = self.meta["lat"] >= 0
        zero_d = self.task("0D_regions")["NH"]
        mean, rms, _ = wstats(self.diff[nh], self.area[nh, None] * self.lw[nh])
        self.assertAlmostEqual(zero_d["rms"], rms)
        self.assertAlmostEqual(zero_d["mean"], mean)
        prof = self.task("1D_vert_profile")["vert_profile_NH"]
        self.assertAlmostEqual(prof["rms"][5], wstats(self.diff[nh, 5], self.area[nh])[1])
        bins = self.task("1D_vert_bin_profile")["vert_bin_profile_Global"]
        self.assertAlmostEqual(bins["rms"][1], wstats(self.diff[:, 4:8], self.area[:, None] * self.lw[:, 4:8])[1])


class TestEnsembleSpread(unittest.TestCase):
    def test_finalize_accumulator_unbiased_and_corrected(self):
        members = RNG.standard_normal((5, 10))
        accum = {"sum": members.sum(0), "sumsq": (members ** 2).sum(0),
                 "count": np.full(10, 5, dtype=np.int32)}
        _, sprd0 = _finalize_accumulator(accum)
        _, sprd = _finalize_accumulator(accum, ddof=1, finite_ensemble_correction=True)
        np.testing.assert_allclose(sprd0, members.std(0, ddof=0))
        np.testing.assert_allclose(sprd, members.std(0, ddof=1) * np.sqrt(6 / 5))

    def test_obs_member_spread(self):
        members = RNG.standard_normal((4, 7))
        members[1, 2] = np.nan
        out = _ensemble_spread(members, ddof=1, finite_correction=True)
        n = 3
        expected = np.nanstd(members[:, 2], ddof=1) * np.sqrt((n + 1) / n)
        self.assertAlmostEqual(out[2], expected)
        np.testing.assert_allclose(_ensemble_spread(members, 0, False), np.nanstd(members, 0))


class TestPooling(unittest.TestCase):
    """Pooled per-cycle statistics equal the statistic of all samples."""

    def test_pooled_equals_all_samples(self):
        cycles = [RNG.standard_normal(n) + 0.5 for n in (5, 500, 50)]
        stats = [{"weight": np.float64(len(c)), "mean": c.mean(), "rms": np.sqrt(np.mean(c * c)),
                  "std": c.std(), "spread": np.sqrt(np.mean(c * c))} for c in cycles]

        class Group:
            def __init__(self, d):
                self.variables = d
        allx = np.concatenate(cycles)
        expected = {"mean": allx.mean(), "rms": np.sqrt(np.mean(allx ** 2)), "std": allx.std(),
                    "spread": np.sqrt(np.mean(allx ** 2))}
        for stat, value in expected.items():
            pool = pool_from_rows(stat, [[read_components(Group(s), stat)] for s in stats])
            self.assertAlmostEqual(float(pooled_value(pool)[0]), value)
        # The plain mean of per-cycle RMS (old behaviour) differs.
        self.assertNotAlmostEqual(np.mean([s["rms"] for s in stats]), expected["rms"], places=3)

    def test_time_aggregation_config(self):
        from mpas_stat.core.pooling import resolve_time_aggregation
        # Model graphic.yaml: "statistics" is the list of stats to plot.
        self.assertEqual(resolve_time_aggregation({"statistics": ["rms", "mean"]}), "pooled")
        self.assertEqual(resolve_time_aggregation({"statistics": ["rms"], "time_aggregation": "mean"}), "mean")
        # Obs config: "statistics" is the settings block.
        self.assertEqual(resolve_time_aggregation({"statistics": {"method": "legacy"}}), "mean")
        self.assertEqual(resolve_time_aggregation({"statistics": {"time_aggregation": "mean"}}), "mean")

    def test_obs_names_and_reduction(self):
        class Group:
            def __init__(self, d):
                self.variables = d
        ref = pool_from_rows("RMS", [[read_components(Group({"RMS": np.float64(2.0), "Count": np.float64(10)}), "RMS")]] * 4)
        exp = pool_from_rows("RMS", [[read_components(Group({"RMS": np.float64(1.0), "Count": np.float64(10)}), "RMS")]] * 4)
        red, lo, hi = bootstrap_pool_reduction_ci(ref, exp, n_boot=50)
        self.assertAlmostEqual(float(red[0]), 50.0)
        self.assertAlmostEqual(float(lo[0]), 50.0)


def _write_ioda(path, lat, lon, pres, groups, qc=None, order=None):
    order = np.arange(len(lat)) if order is None else order
    with h5py.File(path, "w") as f:
        md = f.create_group("MetaData")
        md["latitude"] = lat[order]
        md["longitude"] = lon[order]
        md["pressure"] = pres[order]
        md["dateTime"] = np.arange(len(lat), dtype=np.int64)[order]
        f.require_group("ObsValue")["airTemperature"] = np.zeros(len(lat))
        for g, val in groups.items():
            f.require_group(g)["airTemperature"] = np.asarray(val)[order]
        for g, val in (qc or {}).items():
            f.require_group(g)["airTemperature"] = np.asarray(val)[order]


class TestObsSampleAndQC(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        n = 40
        self.lat = RNG.uniform(-50, 50, n)
        self.lon = RNG.uniform(0, 360, n)
        self.pres = RNG.uniform(2e4, 1e5, n)
        self.omb = RNG.standard_normal(n)
        self.oma = 0.5 * self.omb
        self.qc0 = np.zeros(n, dtype="i4")
        self.qc0[:4] = 19          # background check
        self.qc2 = self.qc0.copy()
        self.qc2[4:10] = 10        # rejected later in the outer loop
        self.settings = {"qc_pass_flags": [0, 19], "spread_ddof": 1,
                         "spread_finite_ensemble_correction": True}

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_flag_19_passes_everywhere_only_in_weighted(self):
        p = self.path("a.h5")
        _write_ioda(p, self.lat, self.lon, self.pres, {"ombg": self.omb, "oman": self.oma},
                    {"EffectiveQC0": self.qc0, "EffectiveQC2": self.qc2})
        new = load_observation_data(p, "aircraft", "airTemperature", 1, 0, 1, settings=self.settings)
        old = load_observation_data(p, "aircraft", "airTemperature", 1, 0, 1)
        self.assertEqual(int(np.sum(new["qc"] == 0)), 40)
        self.assertEqual(int(np.sum(old["qc"] == 0)), 36)

    def test_common_sample_for_omb_oma_and_experiments(self):
        a, b = self.path("a.h5"), self.path("b.h5")
        _write_ioda(a, self.lat, self.lon, self.pres, {"ombg": self.omb, "oman": self.oma},
                    {"EffectiveQC0": self.qc0, "EffectiveQC2": self.qc2})
        qc_b = self.qc0.copy()
        qc_b[30:33] = 12
        # Experiment B stores the same observations in a different order.
        _write_ioda(b, self.lat, self.lon, self.pres, {"ombg": self.omb + 1, "oman": self.oma + 1},
                    {"EffectiveQC0": qc_b, "EffectiveQC2": qc_b}, order=RNG.permutation(40))
        vecs = [load_observation_data(f, "aircraft", "airTemperature", 1, ol, 1, settings=self.settings)
                for f in (a, b) for ol in (0, 2)]
        matched = restrict_to_common_sample(vecs)
        counts = [int(np.sum(v["qc"] == 0)) for v in matched]
        self.assertEqual(counts, [31] * 4)       # 40 - 6 (outer loop) - 3 (exp B)
        # Same observations: the sorted kept innovations of A and B differ by exactly 1.
        kept = [np.sort(v["innov"][v["qc"] == 0]) for v in matched]
        np.testing.assert_allclose(kept[2], kept[0] + 1)

    def test_bending_angle_uses_hofx_as_innovation(self):
        p = self.path("g.h5")
        with h5py.File(p, "w") as f:
            md = f.create_group("MetaData")
            md["latitude"] = np.zeros(3)
            md["longitude"] = np.zeros(3)
            md["height"] = np.array([1000.0, 2000.0, 3000.0])
            f.require_group("ObsValue")["bendingAngle"] = np.full(3, 100.0)
            f.require_group("hofx")["bendingAngle"] = np.array([1.0, -2.0, 0.5])
        vec = load_observation_data(p, "gnssrobndropp1d", "bendingAngle", 2, 0, 1, settings=self.settings)
        np.testing.assert_allclose(vec["innov"], [1.0, -2.0, 0.5])


class TestObsCollectorsPooled(unittest.TestCase):
    """Pooled and mean aggregation give arrays of the same shape for every
    obs-space collector (the lat-vert product transposes its samples)."""

    def test_lat_vert_and_lat_lon_shapes(self):
        from datetime import datetime
        from mpas_stat.obs.collect import collect_obs_lat_lon_2d, collect_obs_lat_vert_2d
        from mpas_stat.obs.common import load_config
        from mpas_stat.obs.process import process_experiment
        from mpas_stat.obs.writer import write_obs_diag_files

        with tempfile.TemporaryDirectory() as tmp:
            n = 400
            cycles = [datetime(2024, 9, 13, 6), datetime(2024, 9, 13, 12)]
            for c in cycles:
                db = os.path.join(tmp, "exp", "CyclingDA", c.strftime("%Y%m%d%H"), "dbOut")
                os.makedirs(db)
                lat, lon, pres = RNG.uniform(-60, 60, n), RNG.uniform(0, 360, n), RNG.uniform(2e4, 1e5, n)
                _write_ioda(os.path.join(db, "obsout_da_aircraft.h5"), lat, lon, pres,
                            {"ombg": RNG.standard_normal(n), "oman": RNG.standard_normal(n)},
                            {"EffectiveQC0": np.zeros(n, "i4"), "EffectiveQC2": np.zeros(n, "i4")})
            cfg_path = os.path.join(tmp, "cfg.yaml")
            Path(cfg_path).write_text(
                f"paths: {{da_dir: {tmp}, hofx_dir: {tmp}, diag_dir: {tmp}/diag, graphic_dir: {tmp}/fig}}\n"
                "experiments: {names: [exp], types: [da], outerloop: [2], n_member: [1]}\n"
                "sensors: [aircraft]\n"
                "domains: {global: [null, null, null, null]}\n"
                "plot_settings: {apply_domains: [global], latlon_lat_bin_width: 30, latlon_lon_bin_width: 60}\n"
                "time_settings: {start_time: '2024-09-13 06:00:00', end_time: '2024-09-13 12:00:00', interval: 6}\n"
            )
            config = load_config(cfg_path)
            write_obs_diag_files(process_experiment(config, "aircraft"), config, "aircraft", f"{tmp}/diag")
            boot = {"n_boot": 20, "ci": 95, "seed": 0}
            args = (f"{tmp}/diag", "exp", "omb", "aircraft", cycles, ["T"], ["airTemperature"])
            for mode in ("mean", "pooled"):
                lv = collect_obs_lat_vert_2d(*args, "Pressure (hPa)", True, boot, time_aggregation=mode)
                ll = collect_obs_lat_lon_2d(*args, time_aggregation=mode)
                for out in (lv["RMS"]["T"], ll["RMS"]["T"]):
                    self.assertEqual(out["data"].shape, out["samples"].shape[1:])
                    if mode == "pooled":
                        self.assertEqual(out["pool"]["w"].shape, out["samples"].shape)


class TestComparisonMarkers(unittest.TestCase):
    """Comparison figures plot 100*(exp-ref)/ref; lower RMS than the
    reference (negative difference) gets the upward triangle."""

    def _markers(self, plot_fn, *args):
        import matplotlib
        matplotlib.use("Agg")
        from unittest import mock
        import matplotlib.axes
        calls = []
        real = matplotlib.axes.Axes.scatter

        def spy(ax, *a, **k):
            calls.append((k.get("marker"), np.asarray(a[1] if len(a) > 1 else []).ravel()))
            return real(ax, *a, **k)
        with mock.patch.object(matplotlib.axes.Axes, "scatter", spy), \
                mock.patch("mpas_stat.graphics.scorecard._save_fig"), \
                mock.patch("mpas_stat.graphics.panels.plt", create=True):
            plot_fn(*args)
        return [m for m, _ in calls if m in ("^", "v")]

    def test_scorecard_triangle_follows_improvement(self):
        from mpas_stat.graphics.scorecard import plot_forecast_scorecard
        better = {"leads": np.array([6.0]), "samples": np.full((4, 1), 1.0)}
        worse = {"leads": np.array([6.0]), "samples": np.full((4, 1), 2.0)}
        cfg = {"experiments": {"names": ["ref", "exp"], "labels": ["Ref", "Exp"]},
               "bootstrap": {"n_boot": 20}}
        # exp has LOWER RMS than ref -> improvement -> upward triangle
        self.assertEqual(self._markers(plot_forecast_scorecard,
                                       {"ref": [worse], "exp": [better]}, ["U 500"], "rms", cfg), ["^"])
        self.assertEqual(self._markers(plot_forecast_scorecard,
                                       {"ref": [better], "exp": [worse]}, ["U 500"], "rms", cfg), ["v"])

    def test_difference_helper(self):
        from mpas_stat.core.pooling import entry_difference_ci
        ref = {"samples": np.full((3, 2), 2.0)}
        exp = {"samples": np.array([[1.0, 3.0]] * 3)}
        diff, lo, hi = entry_difference_ci(ref, exp, {"n_boot": 20, "ci": 95, "seed": 0})
        np.testing.assert_allclose(diff, [-50.0, 50.0])
        self.assertTrue(np.all(lo <= diff) and np.all(diff <= hi))


class TestConfidenceIntervalStyle(unittest.TestCase):
    def test_get_ci_style(self):
        from mpas_stat.graphics.style import get_ci_style
        self.assertEqual(get_ci_style({}), "shade")
        self.assertEqual(get_ci_style({"ci_style": "errorbar"}), "errorbar")
        self.assertEqual(get_ci_style({"ci_style": "bars"}), "errorbar")
        self.assertEqual(get_ci_style({"plot_style": {"ci_style": "none"}}), "none")
        self.assertEqual(get_ci_style({"profile_ci_style": "errorbar"}), "errorbar")  # older key
        self.assertEqual(get_ci_style({"profile_errorbar": False}), "none")
        with self.assertRaises(ValueError):
            get_ci_style({"ci_style": "dots"})

    def test_draw_ci(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpas_stat.graphics.style import draw_ci
        x, y = np.arange(3.0), np.array([1.0, 2.0, 3.0])
        for style, horizontal in (("shade", False), ("shade", True), ("errorbar", False),
                                  ("errorbar", True), ("none", False)):
            fig, ax = plt.subplots()
            draw_ci(ax, x, y, y - 0.5, y + 0.5, "blue", {"ci_style": style}, horizontal=horizontal)
            self.assertEqual(len(ax.collections) > 0, style != "none", style)
            if style == "errorbar":
                self.assertEqual(len(ax.containers), 1)
            plt.close(fig)

    def test_obs_cfg_passes_ci_style(self):
        from mpas_stat.obs.experiments import _ci_display_settings
        self.assertEqual(_ci_display_settings({"plot_settings": {"ci_style": "errorbar", "ci_alpha": 0.3}}),
                         {"ci_style": "errorbar", "ci_alpha": 0.3})
        self.assertEqual(_ci_display_settings({"plot_settings": {}}), {})


class TestColourScales(unittest.TestCase):
    def test_nice_symmetric_levels(self):
        from mpas_stat.graphics.colour import nice_levels
        lv = nice_levels(-23.7, 23.7, 20, diverging=True)
        self.assertAlmostEqual(lv[0], -lv[-1])
        self.assertTrue(np.any(np.isclose(lv, 0.0)))          # 0 on a boundary
        self.assertGreaterEqual(lv[-1], 23.7)                 # covers the data
        self.assertLessEqual(lv[-1], 30.0)                    # ...with a clean limit
        steps = np.diff(lv)
        np.testing.assert_allclose(steps, steps[0])
        seq = nice_levels(0.37, 1.42, 16, diverging=False)
        self.assertLessEqual(seq[0], 0.37)
        self.assertGreaterEqual(seq[-1], 1.42)

    def test_colour_scale_discrete_shared_and_continuous(self):
        import matplotlib.colors as mcolors
        from mpas_stat.graphics.colour import colour_scale
        a, b = np.array([[-3.0, 1.0]]), np.array([[12.0, -4.0]])
        cmap, norm, levels = colour_scale("RdBu_r", None, None, {}, True, extend="both",
                                          data_arrays=[a, b])
        self.assertIsInstance(norm, mcolors.BoundaryNorm)
        self.assertGreaterEqual(levels[-1], 12.0)   # shared: covers both panels
        _, norm2, levels2 = colour_scale("RdBu_r", None, None, {"colorbar": {"discrete": False}},
                                         True, data_arrays=[a])
        self.assertIsNone(levels2)
        self.assertNotIsInstance(norm2, mcolors.BoundaryNorm)
        _, _, fixed = colour_scale("RdBu_r", -10, 10, {}, True, data_arrays=[b])
        self.assertEqual((fixed[0], fixed[-1]), (-10.0, 10.0))  # explicit limits win

    def test_colorbar_ticks_keep_zero(self):
        from mpas_stat.graphics.colour import colorbar_ticks
        ticks = colorbar_ticks(np.linspace(-20, 20, 21), max_ticks=9)
        self.assertIn(0.0, ticks)
        self.assertLessEqual(len(ticks), 11)

    def test_cvd_safe_palette_default(self):
        from mpas_stat.graphics.colour import CVD_SAFE_PALETTE
        from mpas_stat.graphics.style import get_exp_plot_settings
        from mpas_stat.obs.experiments import _palette
        _, _, colors, _ = get_exp_plot_settings({"experiments": {"names": ["a", "b"]}})
        self.assertEqual(colors[:2], CVD_SAFE_PALETTE[:2])
        self.assertEqual(_palette({"plot_settings": {"palette": ["k", "r"]}}), ["k", "r"])
        self.assertEqual(_palette({}), CVD_SAFE_PALETTE)


class TestErrorConsistencyDiagnostics(unittest.TestCase):
    """Desroziers estimates recover KNOWN observation/background errors for an
    optimal analysis, also when OMA comes from a separate, reordered file,
    and the ensemble consistency ratio is ~1 for a consistent ensemble."""

    SO, SB, N = 2.0, 1.5, 6000

    def _write(self, tmp, nmember=1, assumed=1.5):
        rng = np.random.default_rng(3)
        n, so, sb = self.N, self.SO, self.SB
        lat, lon, p = rng.uniform(-60, 60, n), rng.uniform(0, 360, n), rng.uniform(2e4, 1e5, n)
        xb = sb * rng.standard_normal(n)
        y = so * rng.standard_normal(n)
        xa = xb + sb**2 / (sb**2 + so**2) * (y - xb)
        members = xb[None, :] + sb * rng.standard_normal((nmember, n))
        members -= members.mean(0) - xb
        db = os.path.join(tmp, "exp", "CyclingDA", "2024091306", "dbOut")
        os.makedirs(db)
        files = [("obsout_da_aircraft.h5", np.arange(n), ("ombg", y - xb), "EffectiveQC0")]
        if nmember > 1:   # ensemble: OMA in a separate, REORDERED file
            files.append(("obsout_da_aircraft_ana.h5", rng.permutation(n), ("oman", y - xa), "EffectiveQC2"))
        for fname, order, (grp, val), qc in files:
            with h5py.File(os.path.join(db, fname), "w") as f:
                md = f.create_group("MetaData")
                md["latitude"], md["longitude"], md["pressure"] = lat[order], lon[order], p[order]
                for v in ("windEastward", "windNorthward", "airTemperature"):
                    f.require_group("ObsValue")[v] = y[order]
                    f.require_group(grp)[v] = val[order]
                    f.require_group(qc)[v] = np.zeros(n, "i4")
                    f.require_group("ObsError")[v] = np.full(n, so)
                    f.require_group("EffectiveError2")[v] = np.full(n, assumed * so)
                    f.require_group("EffectiveError0")[v] = np.full(n, assumed * so)
                    if nmember == 1:
                        f.require_group("oman")[v] = (y - xa)[order]
                        f.require_group("EffectiveQC2")[v] = np.zeros(n, "i4")
                    for k in range(nmember if nmember > 1 else 0):
                        f.require_group(f"hofx0_{k+1}")[v] = members[k][order]
        return db

    def _run(self, nmember, method="weighted"):
        from mpas_stat.obs.common import load_config
        from mpas_stat.obs.process import process_experiment
        from mpas_stat.obs.writer import write_obs_diag_files
        tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, tmp)
        self._write(tmp, nmember)
        cfg_path = os.path.join(tmp, "cfg.yaml")
        Path(cfg_path).write_text(
            f"paths: {{da_dir: {tmp}, hofx_dir: {tmp}, diag_dir: {tmp}/diag, graphic_dir: {tmp}/fig}}\n"
            f"experiments: {{names: [exp], types: [da], outerloop: [2], n_member: [{nmember}]}}\n"
            "sensors: [aircraft]\ndomains: {global: [null, null, null, null]}\n"
            "plot_settings: {apply_domains: [global], latlon_domain: global}\n"
            "time_settings: {start_time: '2024-09-13 06:00:00', end_time: '2024-09-13 06:00:00', interval: 6}\n"
            f"statistics: {{method: {method}}}\n")
        config = load_config(cfg_path)
        write_obs_diag_files(process_experiment(config, "aircraft"), config, "aircraft", f"{tmp}/diag")
        self.config, self.diag_dir = config, f"{tmp}/diag"
        return nc4.Dataset(f"{tmp}/diag/exp/VerifyOBS/2024091306/obsstat_aircraft.2024-09-13_06.00.00.nc")

    def test_deterministic_recovers_known_errors(self):
        with self._run(1) as ds:
            g = ds["omb/airTemperature/0D/global"]
            self.assertAlmostEqual(float(g["ObsErrorDesroziers"][...]), self.SO, delta=0.06)
            self.assertAlmostEqual(float(g["BkgErrorDesroziers"][...]), self.SB, delta=0.06)
            self.assertAlmostEqual(float(g["ObsErrorAssumed"][...]), 1.5 * self.SO, places=4)
            self.assertNotIn("ObsErrorDesroziers", ds["oma/airTemperature/0D/global"].variables)

    def test_ensemble_reordered_analysis_file_and_consistency_ratio(self):
        with self._run(10) as ds:
            g = ds["omb/windEastward/0D/global"]
            self.assertAlmostEqual(float(g["ObsErrorDesroziers"][...]), self.SO, delta=0.06)
            self.assertAlmostEqual(float(g["BkgErrorDesroziers"][...]), self.SB, delta=0.06)
            cr = float(g["TotalSpread"][...]) / float(g["RMS"][...])
            self.assertAlmostEqual(cr, 1.0, delta=0.05)

    def test_export_tables_and_jedi_filter(self):
        import csv
        import yaml
        from mpas_stat.obs.error_export import export_error_estimates
        self._run(1).close()
        files = export_error_estimates(self.config, "aircraft", self.diag_dir)
        self.assertEqual(sorted(os.path.splitext(f)[1] for f in files), [".csv", ".yaml"])
        csv_path = next(f for f in files if f.endswith(".csv"))
        self.assertIn(os.path.join("exp", "VerifyOBS", "ObsErrorEstimates"), csv_path)
        rows = list(csv.DictReader(open(csv_path)))
        tot = next(r for r in rows if r["variable"] == "T" and r["level"] == "all")
        self.assertAlmostEqual(float(tot["desroziers_obs_error"]), self.SO, delta=0.06)
        self.assertAlmostEqual(float(tot["desroziers_to_assumed_ratio"]), 1 / 1.5, delta=0.03)
        doc = yaml.safe_load(open(next(f for f in files if f.endswith(".yaml"))))
        flt = doc["variables"]["T"]["jedi_filter"]
        self.assertEqual(flt["action"]["name"], "assign error")
        opts = flt["action"]["error function"]["options"]
        self.assertEqual(opts["xvar"]["name"], "MetaData/pressure")
        self.assertEqual(len(opts["xvals"]), len(opts["errors"]))
        self.assertGreater(min(opts["xvals"]), 1000.0)          # Pa, not hPa
        self.assertTrue(all(abs(e - self.SO) < 0.3 for e in opts["errors"]))

    def test_ensemble_rank_histogram_and_spread_skill_written(self):
        from mpas_stat.graphics.plots_ensemble import rank_histogram_summary
        from mpas_stat.obs.collect import collect_obs_rank_histograms, collect_obs_spread_skill
        from mpas_stat.obs.common import generate_cycles, resolve_time_settings
        with self._run(10) as ds:
            g = ds["omb/airTemperature/0D/global"]
            hist = np.asarray(g["RankHistogram"][:])
            self.assertEqual(hist.size, 11)
            self.assertEqual(hist.sum(), float(g["Count"][...]))
            self.assertNotIn("RankHistogram", ds["oma/airTemperature/0D/global"].variables)
            prof = np.asarray(ds["omb/airTemperature/1D/vert_pres_profile_global/RankHistogram"][:])
            self.assertEqual(prof.sum(), hist.sum())        # every obs falls in one level bin
        _, outlier, delta = rank_histogram_summary(hist)
        self.assertAlmostEqual(outlier, 1.0, delta=0.15)    # consistent ensemble: flat
        self.assertLess(delta, 0.08)
        start, end, step = resolve_time_settings(self.config, "aircraft")
        cycles = generate_cycles(start, end, step)
        args = (self.diag_dir, "exp", "aircraft", cycles, ["T"], ["airTemperature"], "global")
        rh = collect_obs_rank_histograms(*args, profile_key="vert_pres_profile_global")
        self.assertEqual(rh["T"]["hist"].sum(), hist.sum())
        ss = collect_obs_spread_skill(*args, n_groups=5, min_count=10)["T"]
        self.assertEqual(int(ss["count"].sum()), int(hist.sum()))
        # consistent ensemble: pooled over all groups, error and spread agree
        # (per group they need not: here the true spread is constant, so
        # binning on the sampled spread only shows regression to the mean)
        c = ss["count"]
        self.assertAlmostEqual(np.sqrt(np.sum(c * ss["rms"] ** 2) / c.sum())
                               / np.sqrt(np.sum(c * ss["total_spread"] ** 2) / c.sum()),
                               1.0, delta=0.05)

    def test_normalized_departures_known_values(self):
        # d = O-B ~ N(0, so^2 + sb^2); the DA assumed 1.5 * so; optimal OMA
        # has variance so^4 / (so^2 + sb^2).
        so, sb = self.SO, self.SB
        with self._run(1) as ds:
            g = ds["omb/airTemperature/0D/global"]
            self.assertAlmostEqual(float(g["NormDepRMS"][...]),
                                   np.sqrt(so**2 + sb**2) / (1.5 * so), delta=0.02)
            self.assertAlmostEqual(float(g["NormDepMean"][...]), 0.0, delta=0.03)
            hist = np.asarray(g["NormDepHist"][:])
            self.assertEqual(hist.sum(), float(g["NormDepCount"][...]))
            self.assertNotIn("NormDepTotalRMS", g.variables)      # deterministic
            a = ds["oma/airTemperature/0D/global"]
            self.assertAlmostEqual(float(a["NormDepRMS"][...]),
                                   np.sqrt(so**4 / (so**2 + sb**2)) / (1.5 * so), delta=0.02)
            prof = ds["omb/airTemperature/1D/vert_pres_profile_global"]
            self.assertEqual(float(np.sum(prof["NormDepCount"][:])), hist.sum())
        with self._run(10) as ds:
            g = ds["omb/windEastward/0D/global"]
            self.assertAlmostEqual(float(g["NormDepTotalRMS"][...]), 1.0, delta=0.04)
            self.assertIn("NormDepTotalHist", g.variables)

    def test_normalized_departures_collected_and_pooled(self):
        from mpas_stat.graphics.plots_normdep import gaussian_tail_fraction
        from mpas_stat.obs.collect import collect_obs_normalized_departures
        from mpas_stat.obs.common import generate_cycles, resolve_time_settings
        self._run(1).close()
        start, end, step = resolve_time_settings(self.config, "aircraft")
        out = collect_obs_normalized_departures(
            self.diag_dir, "exp", "aircraft", generate_cycles(start, end, step), ["T"],
            ["airTemperature"], "global", "vert_pres_profile_global")["T"]
        self.assertEqual(set(out) >= {"omb", "oma", "coord", "edges"}, True)
        self.assertEqual(out["omb"]["hist"].size, out["edges"].size - 1)
        self.assertAlmostEqual(out["omb"]["rms"], np.sqrt(self.SO**2 + self.SB**2) / (1.5 * self.SO),
                               delta=0.02)
        self.assertAlmostEqual(gaussian_tail_fraction(3.0), 0.0026998, places=6)

    def test_legacy_writes_no_error_diagnostics(self):
        with self._run(1, method="legacy") as ds:
            self.assertNotIn("ObsErrorDesroziers", ds["omb/airTemperature/0D/global"].variables)
            self.assertNotIn("NormDepRMS", ds["omb/airTemperature/0D/global"].variables)


class TestEnsembleRank(unittest.TestCase):
    """Rank of the observation among (perturbed) members, and the level
    index helper used for rank histograms by level."""

    def _ranks(self, spread_scale, perturb=True):
        from mpas_stat.obs.reader import _observation_rank
        rng = np.random.default_rng(0)
        n, m, so, sb = 40000, 9, 1.0, 1.5
        truth = np.zeros(n)
        mean = truth + sb * rng.standard_normal(n)
        y = truth + so * rng.standard_normal(n)
        members = mean[None] + spread_scale * sb * rng.standard_normal((m, n))
        return _observation_rank(members, y, None, np.full(n, so), perturb=perturb, seed=1)

    def test_reliable_flat_underdispersive_u_shaped(self):
        from mpas_stat.graphics.plots_ensemble import rank_histogram_summary
        flat = np.bincount(self._ranks(1.0).astype(int), minlength=10)
        _, out_flat, d_flat = rank_histogram_summary(flat)
        self.assertAlmostEqual(out_flat, 1.0, delta=0.08)
        self.assertLess(d_flat, 0.05)
        narrow = np.bincount(self._ranks(0.5).astype(int), minlength=10)
        _, out_narrow, _ = rank_histogram_summary(narrow)
        self.assertGreater(out_narrow, 1.5)
        # without the obs-error perturbation even a reliable ensemble looks U-shaped
        raw = np.bincount(self._ranks(1.0, perturb=False).astype(int), minlength=10)
        self.assertGreater(rank_histogram_summary(raw)[1], 1.2)

    def test_missing_member_gives_nan(self):
        from mpas_stat.obs.reader import _observation_rank
        members = np.array([[1.0, np.nan], [2.0, 2.0]])
        r = _observation_rank(members, np.array([1.5, 1.5]), None, None, perturb=False)
        self.assertEqual(r[0], 1.0)
        self.assertTrue(np.isnan(r[1]))

    def test_level_bin_index_matches_calculate_stats(self):
        from mpas_stat.obs.binning import calculate_stats, level_bin_index
        rng = np.random.default_rng(2)
        for method, levels, vals in (
                (1, np.array([1000., 850., 500., 250., 100.]), rng.uniform(50, 1050, 3000)),
                (3, np.arange(-80., 81., 10.), rng.uniform(-90, 90, 3000))):
            idx = level_bin_index(vals, levels, method)
            counts = calculate_stats(np.ones_like(vals), np.zeros_like(vals), vals,
                                     levels, method)["count"][:-1]
            np.testing.assert_array_equal(np.bincount(idx[idx >= 0], minlength=levels.size),
                                          counts)

    def test_merge_spread_bins_conserves_counts(self):
        from mpas_stat.obs.collect import merge_spread_bins
        count = np.array([0, 5, 100, 300, 50, 0, 10, 535])
        m = {"x": np.arange(8.0)}
        c, mm = merge_spread_bins(count, m, n_groups=4, min_count=1)
        self.assertEqual(c.sum(), count.sum())
        self.assertAlmostEqual(np.sum(c * mm["x"]), np.sum(count * m["x"]))


class TestObsDisplaySettings(unittest.TestCase):
    def test_plot_style_and_colorbar_reach_obs_figures(self):
        from mpas_stat.obs.experiments import _base_graphic_cfg, _comparison_graphic_cfg
        config = {"plot_settings": {"ci_style": "errorbar", "figure_format": "pdf",
                                    "colorbar": {"discrete": False}},
                  "plot_style": {"font_size": 14, "diverging_cmap": "RdYlBu_r"},
                  "experiments": {"names": ["a", "b"], "labels": ["A", "B"],
                                  "types": ["da", "da"], "outerloop": [2, 2],
                                  "n_member": [1, 1]}}
        cfg = _comparison_graphic_cfg(config, "/tmp/x", ["a", "b"], ["A", "B"], ["k", "r"])
        self.assertEqual(cfg["plot_style"]["diverging_cmap"], "RdYlBu_r")
        self.assertFalse(cfg["colorbar"]["discrete"])
        self.assertEqual(cfg["ci_style"], "errorbar")
        base, _ = _base_graphic_cfg(config, "/tmp/x")
        self.assertEqual(base["plot_style"]["font_size"], 14)


class TestChannelCovariance(unittest.TestCase):
    """Desroziers channel covariances recover a KNOWN correlated R for an
    optimal analysis (K = B (B + R)^-1), also with a separate, reordered
    analysis file (ensemble layout)."""

    NCH, RHO_R, RHO_B = 9, 0.5, 0.8
    SO = np.array([3.0, 2.5, 2.0, 0.6, 0.30, 0.25, 0.25, 0.30, 0.35])
    SB = np.array([2.0, 1.5, 1.0, 0.4, 0.25, 0.22, 0.22, 0.28, 0.35])

    def _cov(self, sig, rho):
        i = np.arange(self.NCH)
        return np.outer(sig, sig) * rho ** np.abs(i[:, None] - i[None, :])

    def _run(self, ensemble):
        from mpas_stat.obs.common import load_config
        from mpas_stat.obs.process import process_experiment
        from mpas_stat.obs.writer import write_obs_diag_files
        tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, tmp)
        R, B = self._cov(self.SO, self.RHO_R), self._cov(self.SB, self.RHO_B)
        K = B @ np.linalg.inv(B + R)
        rng = np.random.default_rng(7)
        n, cycles = 8000, ["2024091300", "2024091306", "2024091312", "2024091318"]
        for c in cycles:
            db = os.path.join(tmp, "exp", "CyclingDA", c, "dbOut")
            os.makedirs(db)
            lat, lon = rng.uniform(-60, 60, n), rng.uniform(0, 360, n)
            eo = rng.standard_normal((n, self.NCH)) @ np.linalg.cholesky(R).T
            eb = rng.standard_normal((n, self.NCH)) @ np.linalg.cholesky(B).T
            d = eo - eb
            oa = d - d @ K.T
            files = [("obsout_da_amsua_n18.h5", np.arange(n), {"ombg": d, "oman": oa})]
            if ensemble:
                files = [("obsout_da_amsua_n18.h5", np.arange(n), {"ombg": d}),
                         ("obsout_da_amsua_n18_ana.h5", rng.permutation(n), {"oman": oa})]
            for fname, order, groups in files:
                with h5py.File(os.path.join(db, fname), "w") as f:
                    md = f.create_group("MetaData")
                    md["latitude"], md["longitude"] = lat[order], lon[order]
                    for g, val in groups.items():
                        f.require_group(g)["brightnessTemperature"] = val[order]
                    for q in ("EffectiveQC0", "EffectiveQC2"):
                        f.require_group(q)["brightnessTemperature"] = np.zeros((n, self.NCH), "i4")
                    f.require_group("ObsError")["brightnessTemperature"] = np.tile(self.SO, (n, 1))[order]
                    if ensemble:
                        for k in range(3):
                            f.require_group(f"hofx0_{k+1}")["brightnessTemperature"] = \
                                250 + rng.standard_normal((n, self.NCH))
        cfg_path = os.path.join(tmp, "cfg.yaml")
        Path(cfg_path).write_text(
            f"paths: {{da_dir: {tmp}, hofx_dir: {tmp}, diag_dir: {tmp}/diag, graphic_dir: {tmp}/fig}}\n"
            f"experiments: {{names: [exp], types: [da], outerloop: [2], n_member: [{3 if ensemble else 1}]}}\n"
            "sensors: [amsua_n18]\ndomains: {global: [null, null, null, null], "
            "NH: [0.0, 90.0, -180.0, 180.0]}\n"
            "plot_settings: {apply_domains: [global, NH], latlon_domain: global}\n"
            "time_settings: {start_time: '2024-09-13 00:00:00', end_time: '2024-09-13 18:00:00', interval: 6}\n"
            "statistics: {correlation_channels: {amsua_n18: all}}\n")
        config = load_config(cfg_path)
        write_obs_diag_files(process_experiment(config, "amsua_n18"), config, "amsua_n18",
                             f"{tmp}/diag")
        return config, f"{tmp}/diag", R

    def _check(self, ensemble):
        from mpas_stat.obs.channel_cov import correlation_from_cov
        from mpas_stat.obs.collect import collect_channel_covariance
        from mpas_stat.obs.common import generate_cycles, resolve_time_settings
        config, diag, R = self._run(ensemble)
        cycles = generate_cycles(*resolve_time_settings(config, "amsua_n18"))
        cov = collect_channel_covariance(diag, "exp", "amsua_n18", cycles, "global")
        self.assertEqual(list(cov["channels"]), list(range(1, self.NCH + 1)))   # "all"
        self.assertEqual(cov["n_cycles"], 4)
        sigma, corr, min_eig = correlation_from_cov(cov["R"])
        np.testing.assert_allclose(sigma, self.SO, rtol=0.05)
        true_corr = R / np.outer(self.SO, self.SO)
        np.testing.assert_allclose(corr, true_corr, atol=0.05)
        self.assertGreater(min_eig, 0.0)
        nh = collect_channel_covariance(diag, "exp", "amsua_n18", cycles, "NH")
        self.assertLess(nh["count"][0, 0], cov["count"][0, 0])            # domain mask
        return config, diag

    def test_deterministic_recovers_correlated_R_and_exports(self):
        config, diag = self._check(ensemble=False)
        from mpas_stat.obs.error_export import export_error_estimates
        files = export_error_estimates(config, "amsua_n18", diag)
        rfile = next(f for f in files if "desroziers_R_" in f and f.endswith(".yaml"))
        import yaml
        doc = yaml.safe_load(open(rfile))
        self.assertEqual(len(doc["correlation"]), self.NCH)
        self.assertAlmostEqual(doc["correlation"][3][4], self.RHO_R, delta=0.05)

    def test_ensemble_reordered_analysis_file(self):
        self._check(ensemble=True)


class TestDateTickLabels(unittest.TestCase):
    def test_month_and_year_on_their_own_line(self):
        """Day numbers alone on the first line; month (+ year) on the second,
        only at the first tick and when the month changes."""
        from mpas_stat.graphics.axes import _compact_cycle_label as label
        seq, prev, out = ["2024091300", "2024091306", "2024091400", "2024100100",
                          "2024100200", "2025010100"], None, []
        for c in seq:
            lab = label(c, prev)
            out.append(lab)
            prev = c if lab else prev
        self.assertEqual(out, ["13\nSep 2024", "", "14", "1\nOct", "2", "1\nJan 2025"])
        self.assertTrue(all("\n" not in lab.split("\n")[0] for lab in out))


class TestPlotExtent(unittest.TestCase):
    """Maps and latitude axes zoom to where the data are."""

    def test_finite_extent_and_dateline(self):
        from mpas_stat.graphics.extent import finite_extent, lon_extent_center, store_extent
        lat = np.arange(-85.0, 90.0, 10.0)
        lon = np.arange(-175.0, 180.0, 10.0)
        d = np.full((lat.size, lon.size), np.nan)
        d[(lat > 0) & (lat < 30), :][:, :] = np.nan
        d[np.ix_((lat > 0) & (lat < 30), (lon > 85) & (lon < 155))] = 1.0
        (x0, x1), (y0, y1) = finite_extent(lon, lat, d, lon_axis="x")
        self.assertEqual((x0, x1, y0, y1), (90.0, 150.0, 0.0, 30.0))
        # a box across the dateline: 160E..160W
        d2 = np.full_like(d, np.nan)
        d2[np.ix_((lat > 0) & (lat < 30), (lon > 155) | (lon < -155))] = 1.0
        (x0, x1), _ = finite_extent(lon, lat, d2, lon_axis="x")
        self.assertEqual((x0, x1), (160.0, 200.0))
        self.assertEqual(lon_extent_center((x0, x1)), 180.0)
        store = {"a": {"T": {"data": d, "lon": lon, "lat": lat}},
                 "b": {"T": {"data": d2, "lon": lon, "lat": lat}}}
        (sx0, sx1), _ = store_extent(store, "lon", "lat")
        self.assertEqual((sx0, sx1), (-180.0, 180.0))     # plain union, no dateline wrap
        self.assertIsNone(finite_extent(lon, lat, np.full_like(d, np.nan)))

    def test_extent_modes(self):
        from mpas_stat.graphics.extent import extent_mode
        self.assertEqual(extent_mode({}), "auto")
        self.assertEqual(extent_mode({"map_extent": "full"}), "full")
        self.assertEqual(extent_mode({"map_extent": [0, 30, 90, 150]}), [0.0, 30.0, 90.0, 150.0])

    def test_regional_figures_zoom(self):
        """Obs only in 0-30N, 90-150E: lat-lon map and lat-vert cross-section
        are drawn over that box, not the globe."""
        import matplotlib.pyplot as plt
        from mpas_stat.graphics import plots_2d
        from mpas_stat.obs.common import load_config
        from mpas_stat.obs.figures import generate_lat_lon_graphics, generate_lat_vert_graphics
        from mpas_stat.obs.process import process_experiment
        from mpas_stat.obs.writer import write_obs_diag_files
        tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, tmp)
        rng = np.random.default_rng(0)
        n = 3000
        for exp in ("ctl", "tst"):
            for c in ("2024091300", "2024091306"):
                db = os.path.join(tmp, exp, "CyclingDA", c, "dbOut")
                os.makedirs(db)
                crng = np.random.default_rng(int(c))
                with h5py.File(os.path.join(db, "obsout_da_aircraft.h5"), "w") as f:
                    md = f.create_group("MetaData")
                    md["latitude"], md["longitude"] = crng.uniform(0, 30, n), crng.uniform(90, 150, n)
                    md["pressure"] = crng.uniform(2e4, 1e5, n)
                    for v in ("windEastward", "windNorthward", "airTemperature"):
                        f.require_group("ombg")[v] = rng.standard_normal(n)
                        f.require_group("oman")[v] = 0.5 * rng.standard_normal(n)
                        f.require_group("ObsValue")[v] = np.zeros(n)
                        for q in ("EffectiveQC0", "EffectiveQC2"):
                            f.require_group(q)[v] = np.zeros(n, "i4")
        cfg_path = os.path.join(tmp, "cfg.yaml")
        Path(cfg_path).write_text(
            f"paths: {{da_dir: {tmp}, hofx_dir: {tmp}, diag_dir: {tmp}, graphic_dir: {tmp}/fig}}\n"
            "experiments: {names: [ctl, tst], types: [da, da], outerloop: [2, 2], n_member: [1, 1]}\n"
            "sensors: [aircraft]\ndomains: {global: [null, null, null, null]}\n"
            "plot_settings: {apply_domains: [global], latlon_domain: global}\n"
            "time_settings: {start_time: '2024-09-13 00:00:00', end_time: '2024-09-13 06:00:00', interval: 6}\n"
            "bootstrap: {n_boot: 50}\n")
        config = load_config(cfg_path)
        write_obs_diag_files(process_experiment(config, "aircraft"), config, "aircraft", tmp)
        captured = []
        orig = plots_2d._save_fig
        plots_2d._save_fig = lambda fig, path, cfg=None: (captured.append(fig), plt.close(fig))
        self.addCleanup(setattr, plots_2d, "_save_fig", orig)
        for gen in (generate_lat_lon_graphics, generate_lat_vert_graphics):
            jobs = gen(config, "aircraft", os.path.join(tmp, "fig"), tmp)
            func, args, kw = jobs[0]
            func(*args, **kw)
        map_ax, vert_ax = captured[0].axes[0], captured[1].axes[0]
        if hasattr(map_ax, "get_extent"):                      # Cartopy installed
            import cartopy.crs as ccrs
            x0, x1, y0, y1 = map_ax.get_extent(ccrs.PlateCarree())
        else:
            (x0, x1), (y0, y1) = map_ax.get_xlim(), map_ax.get_ylim()
        self.assertGreater(x0, 80.0)
        self.assertLess(x1, 160.0)
        self.assertGreater(min(y0, y1), -10.0)
        self.assertLess(max(y0, y1), 40.0)
        lo, hi = sorted(vert_ax.get_xlim())
        self.assertGreater(lo, -10.0)
        self.assertLess(hi, 40.0)
        # map_extent: full restores the whole grid
        config["plot_settings"]["map_extent"] = "full"
        captured.clear()
        func, args, kw = generate_lat_vert_graphics(config, "aircraft", os.path.join(tmp, "fig"), tmp)[0]
        func(*args, **kw)
        lo, hi = sorted(captured[0].axes[0].get_xlim())
        self.assertLess(lo, -60.0)


class TestScorecard(unittest.TestCase):
    """The scorecard reports a significant improvement where the test
    experiment has a smaller background error, and nothing significant
    where the two experiments are statistically identical."""

    @classmethod
    def setUpClass(cls):
        from mpas_stat.obs.common import load_config
        from mpas_stat.obs.process import process_experiment
        from mpas_stat.obs.writer import write_obs_diag_files
        tmp = tempfile.mkdtemp()
        cls.tmp = tmp
        rng = np.random.default_rng(1)
        n = 3000
        cycles = [f"2024091{3 + i // 4}{(i % 4) * 6:02d}" for i in range(8)]
        for c in cycles:
            crng = np.random.default_rng(int(c))
            lat, lon = crng.uniform(-60, 60, n), crng.uniform(0, 360, n)
            p = crng.uniform(2e4, 1e5, n)
            for exp, scale in (("ctl", 1.0), ("tst", 0.7)):
                db = os.path.join(tmp, exp, "CyclingDA", c, "dbOut")
                os.makedirs(db)
                with h5py.File(os.path.join(db, "obsout_da_aircraft.h5"), "w") as f:
                    md = f.create_group("MetaData")
                    md["latitude"], md["longitude"], md["pressure"] = lat, lon, p
                    for v in ("windEastward", "windNorthward", "airTemperature"):
                        s = scale if v == "airTemperature" else 1.0
                        f.require_group("ombg")[v] = 2.0 * s * rng.standard_normal(n)
                        f.require_group("oman")[v] = rng.standard_normal(n)
                        f.require_group("ObsValue")[v] = np.zeros(n)
                        for g in ("EffectiveQC0", "EffectiveQC2"):
                            f.require_group(g)[v] = np.zeros(n, "i4")
        cfg_path = os.path.join(tmp, "cfg.yaml")
        Path(cfg_path).write_text(
            f"paths: {{da_dir: {tmp}, hofx_dir: {tmp}, diag_dir: {tmp}, graphic_dir: {tmp}/fig}}\n"
            "experiments: {names: [ctl, tst], labels: [Control, Test], types: [da, da], "
            "outerloop: [2, 2], n_member: [1, 1]}\n"
            "sensors: [aircraft]\ndomains: {global: [null, null, null, null]}\n"
            "plot_settings: {apply_domains: [global], latlon_domain: global, "
            "scorecard: {statistics: [RMS], min_cycles: 5}}\n"
            "time_settings: {start_time: '2024-09-13 00:00:00', "
            "end_time: '2024-09-14 18:00:00', interval: 6}\n"
            "bootstrap: {n_boot: 200}\n")
        cls.config = load_config(cfg_path)
        write_obs_diag_files(process_experiment(cls.config, "aircraft"), cls.config,
                             "aircraft", tmp)

    @classmethod
    def tearDownClass(cls):
        __import__("shutil").rmtree(cls.tmp)

    def test_scorecard_cells(self):
        from mpas_stat.obs.scorecard import generate_scorecards, scorecard_data, scorecard_settings
        config, tmp = self.config, self.tmp
        data = scorecard_data(config, tmp, "ctl", "tst", "omb", scorecard_settings(config))
        rows = data["rows"]
        t_all = rows.index(("aircraft", "T", "all"))
        u_all = rows.index(("aircraft", "U", "all"))
        self.assertAlmostEqual(data["diff"][t_all, 0], -30.0, delta=2.0)
        self.assertLess(data["hi"][t_all, 0], 0.0)                     # significant
        self.assertLess(abs(data["diff"][u_all, 0]), 3.0)
        few = dict(scorecard_settings(config), min_cycles=20)        # 8 cycles only
        data_few = scorecard_data(config, tmp, "ctl", "tst", "omb", few)
        self.assertTrue(np.isnan(data_few["hi"]).all())                 # no markers
        self.assertAlmostEqual(data_few["diff"][t_all, 0], data["diff"][t_all, 0])
        self.assertGreater(sum(r[1] == "T" for r in rows), 2)          # level rows too
        files = generate_scorecards(config, tmp)
        self.assertTrue(any(f.endswith(".png") for f in files))
        self.assertTrue(any(f.endswith(".csv") for f in files))

    def test_profile_diff_omb_and_oma_separately(self):
        """Test vs control profiles compare O-B with O-B and O-A with O-A
        (never O-B with O-A), in separate omb/ and oma/ folders."""
        from mpas_stat.core.bootstrap import get_bootstrap_settings
        from mpas_stat.core.pooling import entry_difference_ci
        from mpas_stat.obs.common import generate_cycles, resolve_time_settings
        from mpas_stat.obs.collect import collect_obs_profile_pools
        from mpas_stat.obs.figures import _profile_diff_jobs
        from mpas_stat.core.jobs import run_job
        cycles = generate_cycles(*resolve_time_settings(self.config, "aircraft"))
        args = ("aircraft", cycles, ["T", "U"], ["airTemperature", "windEastward"],
                "vert_pres_profile_global", "RMS", "Pressure (hPa)", True)
        boot = get_bootstrap_settings(self.config)
        omb = {e: collect_obs_profile_pools(self.tmp, e, "omb", *args) for e in ("ctl", "tst")}
        oma = {e: collect_obs_profile_pools(self.tmp, e, "oma", *args) for e in ("ctl", "tst")}
        self.assertEqual(omb["ctl"]["T"]["pool"]["w"].shape[0], len(cycles))   # aligned
        d_omb, lo, hi = entry_difference_ci(omb["ctl"]["T"], omb["tst"]["T"], boot)
        lev = np.nanargmax(omb["ctl"]["T"]["pool"]["w"].sum(0))
        self.assertAlmostEqual(d_omb[lev], -30.0, delta=4.0)    # test O-B 30 % smaller
        self.assertLess(hi[lev], 0.0)
        d_oma, _, _ = entry_difference_ci(oma["ctl"]["T"], oma["tst"]["T"], boot)
        self.assertLess(abs(d_oma[lev]), 5.0)                   # O-A identical in law
        jobs = _profile_diff_jobs(self.config, "aircraft", os.path.join(self.tmp, "fig", "aircraft"),
                                  self.tmp, cycles, ["T", "U"], ["airTemperature", "windEastward"],
                                  "vert_pres_profile_global", "Pressure (hPa)", True)
        for job in jobs:
            self.assertTrue(run_job(job)[0])
        for diagtype in ("omb", "oma"):
            folder = os.path.join(self.tmp, "fig", "aircraft", diagtype, "profile_diff_1d")
            self.assertTrue(any(f.endswith("_RMS.png") for f in os.listdir(folder)))


class TestDatelineDomains(unittest.TestCase):
    def test_mask_crosses_dateline(self):
        lon = np.array([175.0, -175.0, 0.0, -170.0])
        lat = np.zeros(4)
        for box in ([-10, 10, 170, -170], [-10, 10, 170, 190]):
            qc = _apply_domain_mask(None, lat, lon, "pacific", box)
            np.testing.assert_array_equal(qc == 0, [True, True, False, True])

    def test_2d_grid_crosses_dateline(self):
        out = compute_stats_2d_latlon(np.array([1.0, 2.0, 3.0]), None, np.zeros(3),
                                      np.array([175.0, -175.0, 0.0]),
                                      np.array([-10.0, 10.0]), np.array([170.0, 180.0, 190.0]))
        np.testing.assert_array_equal(out["count"], [[1.0, 1.0]])


class TestVirtualTemperature(unittest.TestCase):
    def test_tv_and_theta(self):
        from mpas_stat.model.variables import extract_mpas_variables_list
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.nc")
            with nc4.Dataset(path, "w") as ds:
                ds.createDimension("Time", None)
                ds.createDimension("nCells", 2)
                ds.createDimension("nVertLevels", 1)
                for name, val in (("theta", 300.0), ("qv", 0.01), ("pressure", 100000.0)):
                    v = ds.createVariable(name, "f8", ("Time", "nCells", "nVertLevels"))
                    v[0] = np.full((2, 1), val)
            out = extract_mpas_variables_list(path, ["Tv", "TH"])
            np.testing.assert_allclose(out["TH"], 300.0)
            np.testing.assert_allclose(out["Tv"], 300.0 * (1 + 0.01 / 0.622) / 1.01)


if __name__ == "__main__":
    unittest.main()

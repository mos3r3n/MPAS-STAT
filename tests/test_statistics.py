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

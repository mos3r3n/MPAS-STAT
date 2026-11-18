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

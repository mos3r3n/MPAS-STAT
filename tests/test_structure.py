"""Tests for the modular package layout and the shared helpers it introduced.

Run from the repository root with::

    python -m unittest discover -s tests -v

* every historical import path (graphics.functions, graphics.io, obs.diag,
  obs.plot, model.kernels, ...) must keep resolving to the same objects as
  the new focused modules;
* the shared helpers in mpas_stat.core and the de-duplicated obs/model
  helpers must behave like the code they replaced.
"""
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpas_stat.core import jobs  # noqa: E402
from mpas_stat.core.grouping import grouped_stats_1d  # noqa: E402
from mpas_stat.model import kernels  # noqa: E402
from mpas_stat.model.driver import _result_dim_map  # noqa: E402


class TestBackwardsCompatibleImports(unittest.TestCase):
    """Old module paths are facades over the new modules."""

    def test_graphics_functions_facade(self):
        from mpas_stat.graphics import functions, panels, plots_0d, scorecard, style
        self.assertIs(functions.plot_time_series_0d, plots_0d.plot_time_series_0d)
        self.assertIs(functions._compute_panel_data, panels._compute_panel_data)
        self.assertIs(functions._save_fig, style._save_fig)
        self.assertIs(functions.plot_forecast_scorecard, scorecard.plot_forecast_scorecard)
        for name in functions.__all__:
            self.assertTrue(hasattr(functions, name), name)

    def test_graphics_io_facade(self):
        from mpas_stat.core import bootstrap, timeutils
        from mpas_stat.graphics import collect, collect_forecast, io
        self.assertIs(io.bootstrap_reduction_ci, bootstrap.bootstrap_reduction_ci)
        self.assertIs(io.generate_cycles, timeutils.generate_cycles)
        self.assertIs(io.collect_stats_0d, collect.collect_stats_0d)
        self.assertIs(io.collect_forecast_scorecard, collect_forecast.collect_forecast_scorecard)
        for name in io.__all__:
            self.assertTrue(hasattr(io, name), name)

    def test_obs_facades(self):
        from mpas_stat.obs import (
            diag, experiments, figures, plot, process, reader, writer,
        )
        self.assertIs(diag.main, process.main)
        self.assertIs(diag.load_observation_data, reader.load_observation_data)
        self.assertIs(diag.write_obs_diag_files, writer.write_obs_diag_files)
        self.assertIs(plot.main, figures.main)
        self.assertIs(plot._build_forecast_experiments, experiments._build_forecast_experiments)

    def test_kernels_reexport_grouped_stats(self):
        self.assertIs(kernels.grouped_stats_1d, grouped_stats_1d)


def _ok_job(path):
    Path(path).write_text("done")


def _bad_job():
    raise RuntimeError("boom")


class TestJobs(unittest.TestCase):
    def test_lead_cfg_adds_suffix_without_mutating(self):
        cfg = {"a": 1}
        out = jobs.lead_cfg(cfg, 6)
        self.assertEqual(out["_lead_suffix"], "_f006")
        self.assertNotIn("_lead_suffix", cfg)

    def test_resolve_n_workers(self):
        self.assertEqual(jobs.resolve_n_workers({}), 1)
        self.assertEqual(jobs.resolve_n_workers({"n_workers": 4}), 4)
        self.assertEqual(jobs.resolve_n_workers({"parallel": {"n_workers": 3}}), 3)
        self.assertEqual(jobs.resolve_n_workers({"n_workers": 0}), 1)

    def test_run_job_captures_errors(self):
        ok, err = jobs.run_job((_bad_job, (), {}))
        self.assertFalse(ok)
        self.assertIn("RuntimeError: boom", err)

    def test_run_jobs_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.txt")
            jobs.run_jobs([(_bad_job, (), {}), (_ok_job, (target,), {})], n_workers=1)
            self.assertTrue(os.path.exists(target))


class TestModelKernels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n, self.nlev = 200, 4
        self.diff = rng.standard_normal((self.n, self.nlev))
        self.spread = np.abs(rng.standard_normal((self.n, self.nlev)))
        self.lat = rng.uniform(-80, 80, self.n)
        self.lat_bins = np.arange(-90.0, 91.0, 30.0)
        self.lat_idx = np.clip(np.digitize(self.lat, self.lat_bins) - 1, 0, len(self.lat_bins) - 2)

    def test_lat_pres_matches_naive_loop(self):
        stats = kernels.compute_2d_lat_pres(
            self.diff, self.spread, self.lat_idx, self.lat_bins, np.arange(self.nlev))
        nlat = len(self.lat_bins) - 1
        self.assertEqual(stats["rms"].shape, (nlat, self.nlev))
        for i in range(nlat):
            sel = self.lat_idx == i
            if not sel.any():
                continue
            np.testing.assert_allclose(stats["mean"][i], self.diff[sel].mean(axis=0))
            np.testing.assert_allclose(stats["rms"][i], np.sqrt((self.diff[sel] ** 2).mean(axis=0)))
            np.testing.assert_allclose(stats["spread"][i], self.spread[sel].mean(axis=0))

    def test_regional_profiles_keys_and_masking(self):
        regions = {"Global": (-90, 90), "NH": (0, 90)}
        out = kernels.compute_1d_vert_hgt_profile(
            self.diff, None, self.lat, np.arange(self.nlev), regions)
        self.assertEqual(list(out), ["vert_hgt_profile_Global", "vert_hgt_profile_NH"])
        self.assertNotIn("spread", out["vert_hgt_profile_NH"])
        nh = self.lat >= 0
        np.testing.assert_allclose(out["vert_hgt_profile_NH"]["mean"], self.diff[nh].mean(axis=0))

    def test_result_dim_map(self):
        self.assertEqual(_result_dim_map("1D", "vert_pres_profile_Global"),
                         {1: ("nPressureLevels",)})
        self.assertEqual(_result_dim_map("3D", "lat_lon_hgt"),
                         {3: ("nLatBins", "nLonBins", "nHeightLevels")})
        with self.assertRaises(ValueError):
            _result_dim_map("2D", "lat_unknown")


class TestObsReader(unittest.TestCase):
    """load_observation_data now shares one code path for radiances and
    conventional obs; check both innovation sources and the HofX QC."""

    def _write(self, path, radiance, groups):
        n = 5
        with h5py.File(path, "w") as f:
            md = f.create_group("MetaData")
            md["latitude"] = np.zeros(n)
            md["longitude"] = np.array([0.0, 90.0, 190.0, 270.0, 359.0])
            md["pressure"] = np.full(n, 50000.0)
            shape = (n, 9) if radiance else (n,)
            var = "brightnessTemperature" if radiance else "airTemperature"
            f.require_group("ObsValue")[var] = np.full(shape, 10.0)
            for g, val in groups.items():
                arr = np.full(shape, val)
                if g.startswith("hofx"):
                    arr[0, ...] = 9.96921e36  # missing HofX
                f.require_group(g)[var] = arr

    def test_conventional_and_radiance_paths(self):
        from mpas_stat.obs.reader import load_observation_data
        with tempfile.TemporaryDirectory() as tmp:
            for radiance in (False, True):
                diag_obs, var, bm = (("amsua_n18", 5, 3) if radiance
                                     else ("aircraft", "airTemperature", 1))
                p = os.path.join(tmp, f"r{radiance}.h5")
                self._write(p, radiance, {"ombg": 1.5, "hofx2": 7.0})
                omb = load_observation_data(p, diag_obs, var, bm, outerloop=0, nmember=1)
                np.testing.assert_array_equal(omb["innov"], np.full(5, 1.5))
                self.assertIsNone(omb["qc"])
                np.testing.assert_allclose(omb["lon"], [0, 90, -170, -90, -1])

                oma = load_observation_data(p, diag_obs, var, bm, outerloop=2, nmember=1)
                np.testing.assert_allclose(oma["innov"][1:], 3.0)  # ObsValue - hofx2
                self.assertTrue(np.isnan(oma["qc"][0]))           # missing HofX excluded
                np.testing.assert_array_equal(oma["qc"][1:], 0.0)


class TestObsStatsRoundTrip(unittest.TestCase):
    """process_experiment -> write_obs_diag_files -> collect_obs_* round trip."""

    def test_round_trip(self):
        from mpas_stat.obs.collect import collect_obs_profiles_1d, collect_obs_stats_0d
        from mpas_stat.obs.common import load_config
        from mpas_stat.obs.process import process_experiment
        from mpas_stat.obs.writer import write_obs_diag_files

        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            cycle = datetime(2024, 9, 13, 6)
            db = os.path.join(tmp, "exp", "CyclingDA", "2024091306", "dbOut")
            os.makedirs(db)
            n = 300
            innov = rng.standard_normal(n)
            with h5py.File(os.path.join(db, "obsout_da_aircraft.h5"), "w") as f:
                md = f.create_group("MetaData")
                md["latitude"] = rng.uniform(-60, 60, n)
                md["longitude"] = rng.uniform(0, 360, n)
                md["pressure"] = rng.uniform(20000, 100000, n)
                for v in ("windNorthward", "windEastward", "airTemperature"):
                    f.require_group("ombg")[v] = innov
                    f.require_group("oman")[v] = 0.5 * innov
                    f.require_group("EffectiveQC0")[v] = np.zeros(n, dtype="i4")
                    f.require_group("EffectiveQC2")[v] = np.zeros(n, dtype="i4")

            cfg_path = os.path.join(tmp, "cfg.yaml")
            Path(cfg_path).write_text(
                f"paths: {{da_dir: {tmp}, hofx_dir: {tmp}, diag_dir: {tmp}/diag, graphic_dir: {tmp}/fig}}\n"
                "experiments: {names: [exp], types: [da], outerloop: [2], n_member: [1]}\n"
                "sensors: [aircraft]\n"
                "domains: {global: [null, null, null, null]}\n"
                "plot_settings: {apply_domains: [global]}\n"
                "time_settings: {start_time: '2024-09-13 06:00:00', end_time: '2024-09-13 06:00:00', interval: 6}\n"
            )
            config = load_config(cfg_path)
            results = process_experiment(config, "aircraft")
            write_obs_diag_files(results, config, "aircraft", f"{tmp}/diag")

            stats = collect_obs_stats_0d(f"{tmp}/diag", "exp", "omb", "aircraft", [cycle],
                                         ["T"], ["airTemperature"], ["global"])
            got = stats["2024091306"]["T"]["0D"]["global"]
            self.assertEqual(got["Count"], n)
            self.assertAlmostEqual(got["RMS"], float(np.sqrt(np.mean(innov ** 2))), places=5)

            profiles = collect_obs_profiles_1d(
                f"{tmp}/diag", "exp", "oma", "aircraft", [cycle], ["T"], ["airTemperature"],
                "vert_pres_profile_global", "Pressure (hPa)", True)
            self.assertIn("T", profiles["Count"])
            self.assertEqual(np.nansum(profiles["Count"]["T"]["mean_profile"]), n)
            self.assertTrue(os.path.exists(
                f"{tmp}/diag/exp/VerifyOBS/2024091306/obsstat_aircraft.2024-09-13_06.00.00.nc"))

            # experiments.write: other experiments are read but not written
            config["experiments"]["write"] = ["nobody"]
            with self.assertRaises(ValueError):
                write_obs_diag_files(results, config, "aircraft", f"{tmp}/diag2")
            config["experiments"]["write"] = []
            write_obs_diag_files(results, config, "aircraft", f"{tmp}/diag2")
            self.assertFalse(os.path.exists(f"{tmp}/diag2/exp"))


if __name__ == "__main__":
    unittest.main()

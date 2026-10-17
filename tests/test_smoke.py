"""Focused smoke tests for the MPAS-STAT verification package.

Run from the repository root with::

    python -m unittest discover -s tests -v

These cover a few review-driven fixes that are cheap to regress and easy to
get wrong silently:

* model lat/lon bin centers must always match the bin edges, even for a
  regional span that is not an exact multiple of the bin width;
* a non-global domain must still mask observations when a sensor has no
  QC array;
* "percent reduction" must be positive when the tested experiment improves
  on the reference;
* pressure-level binning must put each innovation in the expected bin.

They are intentionally narrow unit tests, not an end-to-end pipeline run.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import netCDF4 as nc4
import numpy as np

# Make the src-layout package importable when this file is run directly
# (`python tests/test_smoke.py`) as well as via unittest discovery.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpas_stat.graphics.functions import _compute_panel_data
from mpas_stat.graphics.io import bootstrap_reduction_ci, get_bootstrap_settings
from mpas_stat.model.config import MPASConfig
from mpas_stat.model.driver import prepare_bin_metadata
from mpas_stat.model.interp import configure_numba_threads
from mpas_stat.obs.common import get_levels, load_config
from mpas_stat.obs.diag import (
    _apply_domain_mask, _read_obs_type, _resolve_h5_path, calculate_stats,
)


class TestBinMetadata(unittest.TestCase):
    def _write_mesh(self, lat_deg, lon_deg):
        fd, path = tempfile.mkstemp(suffix=".nc")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        with nc4.Dataset(path, "w") as ds:
            ds.createDimension("nCells", len(lat_deg))
            ds.createDimension("nVertLevels", 55)
            lat = ds.createVariable("latCell", "f8", ("nCells",))
            lat[:] = np.radians(np.asarray(lat_deg, dtype=np.float64))
            lon = ds.createVariable("lonCell", "f8", ("nCells",))
            lon[:] = np.radians(np.asarray(lon_deg, dtype=np.float64))
        return path

    def test_odd_regional_span_centers_match_bins(self):
        # 20..49 degrees is a 29-degree span: with 2-degree bins it yields
        # 15 bins, which used to produce only 14 independently-generated
        # centers and a netCDF shape mismatch at write time.
        path = self._write_mesh([20.0, 30.0, 40.0, 49.0],
                                [100.0, 110.0, 120.0, 129.0])
        cfg = MPASConfig(
            init_file=path,
            regional=True,
            lat_bin_width=2.0,
            lon_bin_width=2.0,
        )
        meta = prepare_bin_metadata(cfg)

        nlat = len(meta["lat_bins"]) - 1
        nlon = len(meta["lon_bins"]) - 1
        self.assertEqual(len(meta["lat_centers"]), nlat)
        self.assertEqual(len(meta["lon_centers"]), nlon)
        self.assertAlmostEqual(meta["lat_centers"][-1], 49.0)
        self.assertAlmostEqual(meta["lon_centers"][-1], 129.0)
        self.assertLess(int(meta["group_ids_latlon"].max()), nlat * nlon)

    def test_interp_module_imports(self):
        # Regression guard for PEP 604 annotations (int | None) that made
        # this module unimportable on Python 3.9.
        self.assertGreaterEqual(configure_numba_threads(1), 1)


class TestDomainMask(unittest.TestCase):
    def test_non_global_domain_masks_without_qc(self):
        lat = np.array([0.0, 40.0, 40.0])
        lon = np.array([0.0, 0.0, -170.0])
        qc = _apply_domain_mask(None, lat, lon, "NXTro",
                                [30.0, 90.0, -180.0, 180.0])
        self.assertIsNotNone(qc)
        np.testing.assert_array_equal(np.isnan(qc), [True, False, False])
        np.testing.assert_array_equal(qc[1:], [0.0, 0.0])

    def test_non_global_domain_combines_with_qc(self):
        lat = np.array([0.0, 40.0, 40.0])
        lon = np.array([0.0, 0.0, -170.0])
        qc = _apply_domain_mask(np.array([0.0, 0.0, 5.0]), lat, lon,
                                "NXTro", [30.0, 90.0, -180.0, 180.0])
        np.testing.assert_array_equal(np.isnan(qc), [True, False, False])
        np.testing.assert_array_equal(qc[1:], [0.0, 5.0])

    def test_global_domain_is_a_no_op(self):
        self.assertIsNone(
            _apply_domain_mask(None, np.array([0.0]), np.array([0.0]),
                               "global", [None, None, None, None])
        )


class TestReductionSign(unittest.TestCase):
    def test_improvement_is_positive(self):
        ref = np.full((3, 1), 2.0)
        exp = np.full((3, 1), 1.0)
        red, ci_low, ci_high = bootstrap_reduction_ci(
            ref, exp, n_boot=200, ci=95, seed=0,
        )
        self.assertAlmostEqual(float(red[0]), 50.0)
        self.assertAlmostEqual(float(ci_low[0]), 50.0)
        self.assertAlmostEqual(float(ci_high[0]), 50.0)

    def test_degradation_is_negative(self):
        ref = np.full((3, 1), 1.0)
        exp = np.full((3, 1), 2.0)
        red, _, _ = bootstrap_reduction_ci(ref, exp, n_boot=200, ci=95, seed=0)
        self.assertAlmostEqual(float(red[0]), -100.0)

    def test_2d_panel_marks_positive_reduction_as_improvement(self):
        ref = np.array([[2.0]] * 3)
        entry = {"data": np.array([1.0]), "samples": np.array([[1.0]] * 3)}
        boot = get_bootstrap_settings({})
        plot_data, significant = _compute_panel_data(
            entry, ref, iexp=1, use_rel=True, stat_name="rms",
            boot=boot, cfg={}, ref_key="samples", var_name="U",
        )
        self.assertAlmostEqual(float(plot_data[0]), 50.0)
        self.assertTrue(bool(significant[0]))


class TestPressureBins(unittest.TestCase):
    def test_each_pressure_lands_in_its_bin(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        qc = np.zeros(data.shape, dtype=float)
        pressure = np.array([980.0, 930.0, 860.0, 30.0])
        levels = np.array([1000.0, 925.0, 850.0, 50.0])

        stats = calculate_stats(data, qc, pressure, levels, bin_method=1)

        # One observation per level, plus the final total row.
        np.testing.assert_array_equal(stats["count"], [1.0, 1.0, 1.0, 1.0, 4.0])
        np.testing.assert_array_equal(stats["mean"], [1.0, 2.0, 3.0, 4.0, 2.5])
        self.assertAlmostEqual(float(stats["rmse"][-1]), np.sqrt(7.5))


class TestConfigAndFileResolution(unittest.TestCase):
    def test_missing_config_path_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_config(os.path.join(tmp, "does_not_exist.yaml"))

    def test_partial_config_is_merged_with_defaults(self):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        with open(path, "w") as f:
            f.write(
                "time_settings:\n"
                "  start_time: '2024-09-13 06:00:00'\n"
                "  end_time: '2024-09-14 00:00:00'\n"
                "  interval: 3\n"
            )
        cfg = load_config(path)
        self.assertEqual(cfg["time_settings"]["interval"], 3)
        self.assertIn("sensors", cfg)
        self.assertIn("global", cfg["domains"])

    def test_h5_fallback_never_picks_analysis_as_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            ana = os.path.join(tmp, "obsout_da_sondes_ana.h5")
            open(ana, "w").close()
            background = os.path.join(tmp, "obsout_da_sondes_2024091306.h5")
            open(background, "w").close()

            # With no background file, the fallback must not return the
            # separate ensemble-analysis file.
            with tempfile.TemporaryDirectory() as empty:
                open(os.path.join(empty, "obsout_da_sondes_ana.h5"), "w").close()
                self.assertIsNone(
                    _resolve_h5_path(os.path.join(empty, "obsout_da_sondes.h5"))
                )

            # A dated background file is a valid fallback, and the result
            # must be the background, never the _ana file.
            resolved = _resolve_h5_path(os.path.join(tmp, "obsout_da_sondes.h5"))
            self.assertEqual(resolved, background)
            self.assertNotEqual(resolved, ana)

    def test_obs_type_supports_dataset_and_group_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, "dataset.h5")
            with h5py.File(dataset_path, "w") as f:
                f.create_dataset("ObsType", data=np.array([120, 221]))

            group_path = os.path.join(tmp, "group.h5")
            with h5py.File(group_path, "w") as f:
                f.create_group("ObsType").create_dataset(
                    "windNorthward", data=np.array([1, 2])
                )

            with h5py.File(dataset_path, "r") as f:
                np.testing.assert_array_equal(
                    _read_obs_type(f, "windNorthward"), [120, 221]
                )
            with h5py.File(group_path, "r") as f:
                np.testing.assert_array_equal(
                    _read_obs_type(f, "windNorthward"), [1, 2]
                )

    def test_get_levels_accepts_zero_bin_bottom(self):
        levels = get_levels(2, bin_width=1.0, bin_bottom=0.0, bin_top=2.0)
        np.testing.assert_allclose(levels, [0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()

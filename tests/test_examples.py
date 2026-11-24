"""Every example configuration under examples/ loads and is self-consistent.

Observation-space examples go through the same resolvers the statistics and
graphics stages use; model-space inputs through MPASConfig; model-space
graphics configs are checked for known plot types and complete blocks.
"""
import glob
import os
import sys
import unittest

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mpas_stat.core.timeutils import generate_forecast_leads  # noqa: E402
from mpas_stat.model.config import MPASConfig  # noqa: E402
from mpas_stat.obs.common import (  # noqa: E402
    generate_cycles,
    load_config,
    resolve_domains,
    resolve_statistics_settings,
    resolve_time_settings,
)
from mpas_stat.obs.scorecard import scorecard_settings  # noqa: E402
from mpas_stat.obs.sensor_config import get_obs_config  # noqa: E402
from mpas_stat.radar.common import resolve_radar_settings  # noqa: E402

OBS = sorted(glob.glob(os.path.join(ROOT, "examples", "obs", "*.yaml")))
MODEL = sorted(glob.glob(os.path.join(ROOT, "examples", "model", "*.yaml")))

MODEL_PLOT_TYPES = {
    "time_series_0d", "profiles_1d", "profile_diff_1d", "profile_reduction_1d",
    "time_level_2d", "lat_vert_2d", "lat_lon_2d", "lat_lon_slice_2d",
    "forecast_lead_0d", "forecast_lead_diff_0d", "forecast_lead_reduction_0d",
    "forecast_profiles_1d", "forecast_profile_diff_1d", "forecast_profile_reduction_1d",
    "forecast_time_level_2d", "forecast_lat_vert_2d", "forecast_lat_lon_2d",
    "forecast_scorecard",
}


class TestObsExamples(unittest.TestCase):
    def test_found(self):
        self.assertGreaterEqual(len(OBS), 10)

    def test_each_obs_example(self):
        for path in OBS:
            with self.subTest(example=os.path.basename(path)):
                raw = yaml.safe_load(open(path))
                config = load_config(path)
                exps = config["experiments"]
                n = len(exps["names"])
                for key in ("types", "outerloop", "n_member"):
                    self.assertEqual(len(exps[key]), n, key)
                if "labels" in raw["experiments"]:
                    self.assertEqual(len(raw["experiments"]["labels"]), n)
                for w in exps.get("write") or []:
                    self.assertIn(w, exps["names"])
                settings = resolve_statistics_settings(config)
                applied = resolve_domains(config)
                ps = config["plot_settings"]
                for d in ps.get("apply_domains") or []:
                    self.assertIn(d, config["domains"])     # else silently dropped
                lld = ps.get("latlon_domain", "global")
                self.assertTrue(lld in ("global", "auto") or lld in config["domains"], lld)
                sc = scorecard_settings(config)
                for d in sc["domains"] or []:
                    self.assertIn(d, applied)               # scorecards use applied domains
                forecast = "hofx_forecast" in exps["types"]
                if forecast:
                    leads = generate_forecast_leads(config)
                    self.assertTrue(leads)
                    for lead in sc["leads"] or []:
                        self.assertIn(int(lead), leads)
                for sensor in raw.get("sensors") or []:
                    get_obs_config(sensor)          # raises for unknown sensors
                    start, end, step = resolve_time_settings(config, sensor,
                                                             for_forecast=forecast)
                    self.assertTrue(generate_cycles(start, end, step))
                if raw.get("radar"):
                    radar = resolve_radar_settings(config)
                    self.assertIsNotNone(radar)
                    self.assertTrue(forecast)
                    for s in radar["sensors"]:
                        get_obs_config(s)
                if settings["method"] == "legacy":
                    self.assertFalse(settings["error_diagnostics"])


class TestModelExamples(unittest.TestCase):
    def test_found(self):
        self.assertGreaterEqual(len(MODEL), 6)

    def test_each_model_example(self):
        for path in MODEL:
            name = os.path.basename(path)
            raw = yaml.safe_load(open(path))
            with self.subTest(example=name):
                if "stats_input" in name:
                    cfg = MPASConfig.from_yaml(path)
                    if cfg.n_member > 1:
                        self.assertIn("{", cfg.forecast_file)   # member template
                    continue
                for key in ("paths", "experiments", "time", "variables", "statistics",
                            "plot_types"):
                    self.assertIn(key, raw)
                self.assertTrue(set(raw["plot_types"]) <= MODEL_PLOT_TYPES,
                                set(raw["plot_types"]) - MODEL_PLOT_TYPES)
                n = len(raw["experiments"]["names"])
                for key in ("labels", "colors", "linestyles"):
                    if key in raw["experiments"]:
                        self.assertEqual(len(raw["experiments"][key]), n, key)
                if any(t.startswith("forecast_") for t in raw["plot_types"]):
                    self.assertTrue(generate_forecast_leads(raw))
                if "forecast_scorecard" in raw["plot_types"]:
                    self.assertIn("rows", raw["scorecard"])


if __name__ == "__main__":
    unittest.main()

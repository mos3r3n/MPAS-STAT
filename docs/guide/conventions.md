# 3. Shared conventions

Model and observation space use the same file layout, plotting code and
comparison rules, so their figures read the same way.

## 3.1 Labels

Labels follow the NCAR mpas-jedi-graphics style:

| Where | Shows | Examples |
|---|---|---|
| Panel title | Variable and units | `T (K)`, `U (m/s)`, `Q (g/kg)`, `Ps (Pa)`, `Channel 7 (K)` |
| Axis or colour bar | Statistic | `rms`, `mean`, `STD`, `Count`, `spread`, `total spread` |
| Comparison axis | Statistic and reference | `rms: % diff. from Control` |
| Date axis | Day, with the month and year once below | `13`, `14`, … above `Sep 2024` |

Units come from the variable tables (`mpas_stat.model.variables`,
`mpas_stat.obs.sensor_config`).

## 3.2 Comparing experiments

The first entry of `experiments.names` is the **reference**. Every
comparison figure shows

```{math}
\text{% difference} = 100\,\frac{\text{exp} - \text{ref}}{\text{ref}}
```

of the statistic pooled over cycles.

- **Negative is better**: a smaller rms or STD than the reference, drawn
  blue and marked ▲. Positive is worse: red, ▼.
- **Significance**: ▲/▼ (or a circle on line plots) mark differences whose
  95 % confidence interval excludes zero. The interval comes from a paired
  bootstrap over cycles: the same cycles are resampled for both
  experiments (§5.5).
- **Small references**: the difference is not shown where the reference
  value is below `relative_min_reference` (a larger threshold applies to
  water vapour).
- **Like with like**: O−B is compared only with O−B, O−A with O−A and O−F
  with O−F, in separate figures.

Figures where this applies: `profile_diff_1d`, `forecast_lead_diff_0d`,
the 2-D maps and sections (right-hand columns), and both scorecards.

## 3.3 Shared design

- **One file layout.** `mpasstat.<date>.nc` and `obsstat_<obs>.<date>.nc`
  both use `<variable>/<0D|1D|2D>/<key>/<statistic>`, so the same reading
  code and the same tools (`ncdump`, Panoply) work for both.
- **One plotting engine.** Observation space draws its figures with the
  model-space functions in `mpas_stat.graphics`. Styling, colour bars,
  panel layout and file names are shared.
- **O−B and O−A on line plots.** The plotting functions draw one line per
  "experiment", so observation space passes each experiment twice, as
  `<label> - OMB` (solid) and `<label> - OMA` (dashed) in the same colour.
- **Forecast leads.** The `forecast` block (`length`, `interval`, `start`)
  means the same on both sides, and leads are pooled the same way.
- **Statistic names.** Observation space uses `RMS`, `Bias`, `STD`,
  `Count`, `Spread`, `TotalSpread`; model space `rms`, `mean`, `std`,
  `spread`. Figure labels use the common style of §3.1.
- **Parallel drawing.** `n_workers` draws figures in parallel: by figure
  in model space, by sensor in observation space. The output does not
  depend on it.

## 3.4 Limitations

- Bootstrap intervals treat cycles as independent, which makes them
  somewhat too narrow (§5.5).
- Observation-space absolute profiles and cycle–level sections have no
  confidence intervals; use the comparison figures for significance.
- Observation space has no 3-D binned products.
- Radar figures use their own simpler styling (§2.6).

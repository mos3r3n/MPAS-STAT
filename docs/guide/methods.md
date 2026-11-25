# 5. Statistical methods

How the statistics are computed, pooled over cycles and compared, and how
to read the diagnostics built on them. These are the defaults since
2026-11; §5.10 lists what changed and how to reproduce older results.

## 5.1 Model space: weighted, pooled statistics

Each product pools **all samples** that fall into one output cell — grid
cells of a region, latitude band or lat-lon box and, for products spanning
several model levels, every level in the column or level bin — into one
weighted mean, RMS and STD:

- **Area weights.** Cells are weighted by `areaCell`, so the refined part
  of a variable-resolution or regional mesh does not dominate.
- **Mass weights.** When model levels are pooled (0-D regions, latitude-bin
  profile, 2-D lat-lon, level-bin products), each level is weighted by its
  layer pressure thickness, from the reference (init) pressure.
- **Squares before square roots.** The 2-D lat-lon RMS of a 3-D variable
  is the RMS over the column, and level-bin products pool every level in
  the bin, so errors of opposite sign do not cancel.
- Per-level products (vertical, height and pressure profiles; lat-hgt,
  lat-pres and their 3-D counterparts) use area weights only.
- Every product also stores its total `weight`, used for pooling over cycles.

## 5.2 Pooling over cycles (`time_aggregation: pooled`)

Figures that summarize many cycles pool the per-cycle statistics with their
weights (model: `weight`; obs: `Count`), which gives the statistic of all
samples of all cycles taken together:

- mean = Σ w·mean / Σ w
- RMS = √(Σ w·RMS² / Σ w)
- STD = √(Σ w·(STD² + mean²) / Σ w − mean²)
- spread = √(Σ w·spread² / Σ w)

`Count` and other statistics keep the plain mean over cycles. Bootstrap
confidence intervals resample whole cycles and re-pool. `mean` restores the
plain average of per-cycle statistics, in which a cycle with 5 observations
counts as much as one with 5,000.

## 5.3 Ensemble spread

Per grid point or observation, spread is the ensemble standard deviation
with `spread_ddof` (1 = unbiased variance) times √((N+1)/N) when
`spread_finite_ensemble_correction` is on, so that for a reliable N-member
ensemble it matches the RMSE of the ensemble mean (Fortin et al. 2014).
Spread is aggregated as √(mean variance), not as the mean standard
deviation (which is smaller, making ensembles look under-dispersive).

The model-space reference (the external analysis in `init_file`) has its own
error, which is not included: a reliable ensemble will look somewhat
under-dispersive against it. Observation-space `TotalSpread` does include
the observation error.

A single-member (deterministic) experiment has no spread. Nothing
spread-related is written or drawn for it:
- **Observation space:** its statistics files have no `Spread`,
  `TotalSpread` or `NormDepTotal*`, and none of the ensemble-only
  diagnostics (rank histograms, spread–skill, consistency ratio).
- **Spread figures:** time series, profiles and Hovmöllers are drawn only
  when at least one experiment in the figure is an ensemble, and they show
  only the ensemble experiments.
- **Model space:** `spread` in `statistics` is skipped, with a printed note,
  when none of the experiments' statistics files contain spread.

## 5.4 Observation space: QC and one common sample

- **QC flags.** `qc_pass_flags` (default `[0, 19]`) applies to every
  innovation source alike. Flag 19 is JEDI's background check, so that
  check does not shape the verified sample.
- **Common sample.** With `common_sample: true`, for each cycle (or init
  cycle and lead) and variable, every OMB, OMA and OMF vector being compared
  — OMB and OMA of each experiment, and all experiments — keeps only the
  observations valid (QC passed, finite innovation) in **all** of them.
  Observations are matched by latitude, longitude, pressure or height and
  `dateTime`, not by file order. Otherwise an "OMA improves on OMB" or
  "experiment B beats A" result partly reflects which observations each one
  rejected.
- **GNSS-RO.** Bending angle and refractivity use the HofX group directly
  as the innovation.

## 5.5 Interpreting results

- **Significance.** Bootstrap intervals resample cycles as independent.
  Consecutive cycles are correlated, so the intervals are somewhat too narrow
  and significance markers appear more often than their nominal rate; many
  cells are also tested at once. Treat isolated significant cells with
  caution.
- **Model-space reference.** Model-space statistics verify against an
  external analysis, which is not independent truth: it favours experiments
  that resemble it, especially at short lead times. Prefer observation-space
  results for analysis-time and early-lead conclusions.

## 5.6 Observation and background error consistency

For experiments that have both OMB and OMA (`da` runs with an analysis
outer loop) the statistics files carry three extra variables in the `omb`
group, for every domain and vertical bin:

| Variable | Definition |
|---|---|
| `ObsErrorDesroziers` | σo ≈ √mean[(O−A)(O−B)] |
| `BkgErrorDesroziers` | σb ≈ √mean[(A−B)(O−B)], with A−B = (O−B) − (O−A) |
| `ObsErrorAssumed` | √mean(σ²) of the error the DA actually used (`EffectiveError<outerloop>`, else `ObsError`) |

They are computed on observations that passed QC in both OMB and OMA
(matched by location and time, so a separate, reordered analysis file is
fine) and are pooled over cycles in mean-square form. The figure
`error_consistency_1d/` shows, per variable, the assumed error with the
two Desroziers estimates (top row) and two ratios (bottom row):

- **Desroziers σo / assumed σo.** Below 1: the assigned observation error is
  larger than the data suggest (observations under-weighted); above 1:
  too small (over-weighted).
- **Consistency ratio** TotalSpread / RMS(OMB) (ensemble runs). Below 1:
  the ensemble is under-dispersive; above 1: over-dispersive.

```{figure} ../gallery/obs_error_consistency.png
:width: 100%

Radiosondes. Top: the assumed σo with the Desroziers σo and σb
estimates. Bottom: Desroziers σo / assumed σo and total spread / RMS(OMB);
both should be near 1. Here the
assumed wind error is too large near the surface and too small aloft.
```

The Desroziers estimates are exact only when the analysis used the correct
error statistics; treat them as a consistency check, not a tuning answer
(Desroziers et al. 2005, *QJRMS* 131, 3385–3396). Values are written only by
`method: weighted` with `error_diagnostics: true`; radiances are shown for
the global domain only, and surface obs get no profile figure.

**Exported tables.** The graphics stage also writes, for every experiment
with OMA and every sensor,

```
<diag_dir>/<exp>/VerifyOBS/ObsErrorEstimates/
    desroziers_<sensor>_<first cycle>-<last cycle>.csv
    desroziers_<sensor>_<first cycle>-<last cycle>.yaml
```

To write only these (no figures), run
`mpas-stat-obs-errors -c obs_graphic.yaml` (or
`python -m mpas_stat.cli.obs_errors -c obs_graphic.yaml`).

- The **CSV** has one row per variable, domain and level, plus an `all`
  row per variable and domain: count, assumed error, Desroziers σo and σb,
  their ratio and (ensembles) the consistency ratio. Units are those of the
  statistics files (specific humidity in g/kg).
- The **YAML** holds the global-domain profiles in IODA units (specific
  humidity in kg/kg) and, per variable, a UFO filter that assigns the
  Desroziers σo:

  ```yaml
  jedi_filter:
    filter: Perform Action
    filter variables: [{name: airTemperature}]
    action:
      name: assign error
      error function:
        name: ObsFunction/ObsErrorModelStepwiseLinear
        options:
          xvar: {name: MetaData/pressure}     # Pa (MetaData/height in m for height bins)
          xvals: [100000, 92500, ...]
          errors: [1.21, 1.05, ...]
  ```

  For radiances it is one `error parameter vector` over the channels.

The "assumed" error is the effective error, i.e. after any QC inflation, so
assigning σo directly as the prior error lets later filters inflate it
again; scaling your existing error model by the ratio is the alternative.
Review the numbers, iterate (re-run the DA and diagnose again) and do not
apply them blindly.

## 5.7 Ensemble reliability: rank histograms and spread-skill

For cycling experiments with `n_member > 1`, the statistics stage also
writes, in the `omb` group:

| Variable | Where | Meaning |
|---|---|---|
| `RankHistogram` | `0D/<domain>` (`nRank` = n_member + 1) and `1D/<profile>` (level × rank) | how many observations fall at each rank among the members |
| `SpreadSkillCount`, `SpreadSkillTotalSpreadMS`, `SpreadSkillSpreadMS`, `SpreadSkillOMBMS`, `SpreadSkillOMBMean` | `0D/<domain>` (`nSpreadBins`) | per-bin count and mean squares, in fixed bins of the per-observation total spread (`SpreadBinEdges`, 40 per decade) |

Before ranking, every member value is perturbed with a random draw from the
observation error `ObsError` (Hamill 2001, *MWR* 129, 550–560). Without the
perturbation, a perfectly reliable ensemble still looks under-dispersive,
because the observations carry their own error. Set
`rank_histogram_perturb: false` to rank against the raw members.

All of these quantities add across cycles. The figures are drawn when
`plot_settings.generate_ensemble_diagnostics` is on (the default):

- **`rank_histogram/rank_histogram_<domain>`:** frequency relative to a flat
  histogram (1 = reliable).
  - A U shape means the spread is too small; a dome means it is too large; a
    slope means a bias.
  - Each panel lists the *outlier ratio*: the share of observations in the two
    extreme ranks, divided by its expected value 2/(n_member+1).
  - It also lists Δ, the mean |frequency − 1|, which is 0 when flat.
- **`rank_histogram/rank_histogram_<profile>_<exp>`:** the same as a rank ×
  level map (by latitude for radiances), global domain only.
- **`spread_skill/spread_skill_<domain>`:** observations grouped by total
  spread into up to `plot_settings.spread_skill.groups` (default 10) groups of
  about equal count. Groups with fewer than `spread_skill.min_count` (default
  50) observations are dropped. The plot shows RMS(OMB) against total spread
  per group; points above the 1:1 line mean the spread is too small for the
  error.

```{figure} ../gallery/obs_rank_histogram.png
:width: 80%

Radiosonde rank histograms, 10 members. Near flat, slightly domed for
winds: the spread is a little too large.
```

```{figure} ../gallery/obs_spread_skill.png
:width: 75%

Spread–skill for the same ensemble. Points fall below the diagonal at
large spread: where the ensemble is most spread out, it over-estimates
the error.
```

The total spread includes the assumed observation error, so a mis-specified
observation error (see §5.6) shows up here too. Read the two diagnostics
together.

## 5.8 Normalized departures

Departures are divided by the error the DA itself assigned (the same
`EffectiveError<outer loop>` / `ObsError` as `ObsErrorAssumed`;
`EffectiveError0` for O−B). For ensembles, O−B is also divided by the total
spread √(spread² + σo²). Written per domain and level, in the `omb` and
`oma` groups:

| Variable | Meaning |
|---|---|
| `NormDepRMS`, `NormDepMean`, `NormDepCount` | RMS, mean and count of d/σo, where d = O−B or O−A |
| `NormDepTotalRMS`, `NormDepTotalMean`, `NormDepTotalCount` | the same for (O−B)/total spread (ensembles, `omb` only) |
| `NormDepHist`, `NormDepTotalHist` | 0D histograms over `NormDepBinEdges` (−8 to 8 in steps of 0.25; values outside go in the end bins) |

They are pooled over cycles with their own counts. Figures go to
`<sensor>/normalized_departure/` and are switched by
`plot_settings.generate_normalized_departures`:

- **`normdep_histogram_<domain>`:** the O−B distributions on a log scale,
  next to a Gaussian with the same RMS. Each panel lists the RMS, the mean and
  the share of |z| > 3 together with its Gaussian value.
- **`normdep_<profile>`:** the RMS (top row) and mean (bottom row) of the
  normalized departures by level.

```{figure} ../gallery/obs_normdep_hist.png
:width: 85%

Radiosonde (O−B)/σo for both experiments (solid) and (O−B)/total spread
for the ensemble (dashed), against Gaussians with the same RMS (dotted).
The statistics in each panel give the RMS, mean and share beyond 3σ.
```

```{figure} ../gallery/obs_normdep_profile.png
:width: 100%

The same by pressure: RMS (top) and mean (bottom) of the normalized
departures.
```

How to read them:

- **(O−B)/σo:** its expected RMS is √(1 + σb²/σo²), which is above 1. It is
  the quantity the background check compares with its threshold.
- **(O−B)/total spread:** RMS should be 1. Above 1 means too little spread or
  too small an observation error; below 1 means the opposite.
- **(O−A)/σo:** mean square = 2 Jo / n, the analysis cost per observation.
- **Mean:** a bias in units of σo. For example, a mean of −0.2 means the
  background is warm by 0.2 σo.
- **Tails:** heavier than the Gaussian means gross errors or non-Gaussian
  errors. That is where QC thresholds or a Huber norm act, and it matters for
  the significance of RMS differences.

## 5.9 Inter-channel observation-error correlations (radiances)

For radiance sensors in "da" experiments with OMA, the Desroziers estimate
of §5.6 is extended to every pair of channels i, j:

    R_ij ≈ mean[(O−A)_i (O−B)_j]     B_ij ≈ mean[(A−B)_i (O−B)_j]     D_ij = mean[(O−B)_i (O−B)_j]

Each is taken over the locations where both channels pass QC in O−B and
O−A. OMB and OMA rows are matched by location, so the separate, reordered
`_ana.h5` file of an ensemble run works too.

**Channels.** By default the channels are the sensor's configured channels.
`statistics.correlation_channels` maps a sensor to a list of channels, or to
`all` (every channel in the file).

**Where it is stored.** Each statistics file gets a `channel_covariance/`
group with `Channels`, and one subgroup per domain holding
`ObsErrorCovDesroziers`, `BkgErrorCovDesroziers`, `OMBCov` and `PairCount`.
These are pooled over cycles, weighted by count.

**Outputs:**

- **Figure** `<sensor>/channel_correlation/channel_correlation_<domain>`:
  the correlation matrix of the symmetrized R (top) and of B (bottom), one
  column per experiment. The title gives the smallest eigenvalue of R. It is
  switched by `plot_settings.generate_channel_correlations`.

  ```{figure} ../gallery/obs_channel_corr.png
  :width: 75%

  AMSU-A channels 5–9: correlations of the estimated observation error
  (top) and background error (bottom). The synthetic data have
  uncorrelated errors, so the off-diagonal values are near zero.
  ```
- **Export** `<diag_dir>/<exp>/VerifyOBS/ObsErrorEstimates/desroziers_R_<sensor>_<cycles>.{csv,yaml}`
  (global domain): channels, σ per channel and the correlation matrix, so that
  R = diag(σ) · C · diag(σ). The files also give the smallest eigenvalue and
  the smallest pair count.

**Caveats.**
- R is exact only for an optimal analysis.
- The estimate is not symmetric in general; it is symmetrized here.
- Sampling noise can make it indefinite (smallest eigenvalue < 0). Regularize
  it, e.g. by eigenvalue clipping or reconditioning, before using it as a
  covariance.
- The exported files are not in any UFO-specific format; convert them to what
  your observation-error covariance model reads.

## 5.10 Changes in 2026-11 and older results

The methods above replaced the original ones in 2026-11. Results from
before then differ in these ways:

| Area | Before 2026-11 | Now |
|---|---|---|
| Model-space weights | Every cell and level counted equally | Area and layer-mass weights (§5.1) |
| Column and level-bin RMS | Signed error averaged over the column or bin first, so errors of opposite sign cancelled and the "RMS" mostly showed the bias | Squares pooled before the square root |
| Pooling over cycles | Plain mean of per-cycle statistics | Weighted pooling (§5.2) |
| Ensemble spread | Mean of standard deviations, ddof 0 | √(mean variance), ddof 1 with the (N+1)/N factor (§5.3) |
| QC flag 19 | Passed only for HofX-derived innovations | Passed for every source (`qc_pass_flags`) |
| Observation sample | Each O−B, O−A and experiment on its own QC'd sample | One common sample (§5.4) |
| GNSS-RO bending angle | Computed as ObsValue − HofX by mistake | HofX group used as the innovation |
| Model `Tv` | Returned potential temperature by mistake | Virtual temperature (`TH` is potential temperature) |

To reproduce older results exactly, set all three:

```yaml
statistics_method: legacy        # model-space statistics configuration
time_aggregation: mean           # model-space graphics configuration
statistics: {method: legacy}     # observation-space configuration
```

The `Tv` and bending-angle fixes are not reverted by these settings.

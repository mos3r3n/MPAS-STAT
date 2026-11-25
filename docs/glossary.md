# Glossary

```{glossary}
O−B (OMB)
  Observation minus background: the observation minus the model
  background (first guess) interpolated to it. The main measure of
  short-range forecast quality in observation space.

O−A (OMA)
  Observation minus analysis. Smaller than O−B when the analysis draws
  to the observations; that alone does not make an analysis better.

O−F (OMF)
  Observation minus forecast, at a given lead time, from
  `hofx_forecast` experiments.

H(x), HofX
  The model state mapped to an observation: the simulated observation.

Reference experiment
  The first entry of `experiments.names`. Every comparison figure shows
  the other experiments relative to it (§3.2).

Percent difference
  `100·(exp − ref)/ref` of a pooled statistic. Negative means a smaller
  error than the reference.

Pooling
  Combining cycles (or samples) by their weights so the result equals
  the statistic of all samples together, rather than averaging
  per-cycle statistics (§5.2).

Common sample
  The observations valid in every O−B, O−A and O−F being compared, so
  differences are not caused by different QC decisions (§5.4).

Paired bootstrap
  Resampling whole cycles, the same cycles for both experiments, to get
  a confidence interval for their difference (§5.5).

Spread
  Standard deviation of the ensemble members about their mean (§5.3).

Total spread
  √(spread² + σo²): ensemble spread plus the assumed observation error;
  what rms(O−B) should match for a reliable ensemble.

σo, σb
  Observation and background error standard deviations. "Assumed σo" is
  the error the DA used (`EffectiveError`).

Desroziers diagnostics
  Estimates of σo and σb from products of O−B and O−A statistics
  (Desroziers et al. 2005), used to check the assumed errors (§5.6).

Normalized departure
  A departure divided by the assumed σo (or by the total spread); its
  RMS and distribution show whether the assumed errors fit (§5.8).

Rank histogram
  How often the observation falls at each rank among the sorted
  ensemble members; flat for a reliable ensemble (§5.7).

Domain
  A latitude–longitude box over which observation-space statistics are
  averaged (`domains`). Model space uses latitude `regions`.

Cycle
  One assimilation time, written `YYYYMMDDHH`. For forecasts, the init
  cycle is when the forecast starts and the lead its length in hours.
```

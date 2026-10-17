# Example configurations

| File | Purpose |
|---|---|
| `obs_verify_example.yaml` | Cycling DA OMB/OMA verification (`types: [da, ...]`). |
| `obs_forecast_example.yaml` | Extended-forecast OMF skill verification (`types: [hofx_forecast]`). |

Both are annotated templates. Replace the `/path/to/...` placeholders and
the example experiment/sensor names before running. The observation-space
statistics and graphics stages must use the same config file.

Radar composite-reflectivity configuration is described in
[`docs/USER_GUIDE.md`](../docs/USER_GUIDE.md), section 2.6.

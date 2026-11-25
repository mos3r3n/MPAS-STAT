# Configuration reference

The annotated templates below list every option with its default. The
user guide explains them in context: {doc}`guide/model_space` (§1.3–1.5),
{doc}`guide/obs_space` (§2.3) and {doc}`guide/methods` (statistics methods).

| Template | Used by |
|---|---|
| `configs/model_input.yaml` | `mpas-stat-model` (one forecast file per run) |
| `configs/model_graphic.yaml` | `mpas-stat-graphics` |
| `configs/obs_input.yaml` | `scripts/run_verify_obs*.ksh` (per experiment and cycle) |
| `configs/obs_graphic.yaml` | `mpas-stat-obs-plot`, `mpas-stat-obs-errors` |

## `model_graphic.yaml`

```{literalinclude} ../configs/model_graphic.yaml
:language: yaml
```

## `model_input.yaml`

```{literalinclude} ../configs/model_input.yaml
:language: yaml
```

## `obs_graphic.yaml`

```{literalinclude} ../configs/obs_graphic.yaml
:language: yaml
```

## `obs_input.yaml`

```{literalinclude} ../configs/obs_input.yaml
:language: yaml
```

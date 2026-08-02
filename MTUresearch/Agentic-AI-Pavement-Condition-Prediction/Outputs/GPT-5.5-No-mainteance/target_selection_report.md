# TARGET_SELECT Report

## Selected Target

- Target column: `IRI_mean_y`
- Task type: `regression`
- Target family: `iri`
- Candidate source: `/content/pavement_agentic_workspace/target_candidates.csv`

`IRI_mean_y` is selected as the pavement condition prediction target and should be modeled as a continuous regression target.

## Rationale

IRI / International Roughness Index is a standard pavement ride-quality and condition measure. The `_y` suffix suggests an outcome-side or later-condition variable in this joined dataset, making `IRI_mean_y` appropriate for future or post-treatment pavement condition prediction.

## Candidate Comparison

Approved target candidates are listed below as CSV-formatted markdown text:

```csv
column,family,family_confidence,reasons,leakage_guard
IRI_mean_x,iri,high,"[""Matched IRI/Roughness via token 'iri' or substring 'roughness'.""]",Potential condition metric. Downstream agents must prevent leakage by excluding any post-period / outcome / future condition columns from features if applicable.
IRI_mean_y,iri,high,"[""Matched IRI/Roughness via token 'iri' or substring 'roughness'.""]",Potential condition metric. Downstream agents must prevent leakage by excluding any post-period / outcome / future condition columns from features if applicable.
Cracking_Percent_mean_x,cracking_percent,high,"[""Matched Cracking via substring 'cracking' or token 'crack'."", ""Name suggests percent measure (%, percent, or pct).""]",Potential condition metric. Downstream agents must prevent leakage by excluding any post-period / outcome / future condition columns from features if applicable.
Rutting_mean_x,rutting,medium,"[""Matched Rutting via substring 'rutting' or token 'rut'.""]",Potential condition metric. Downstream agents must prevent leakage by excluding any post-period / outcome / future condition columns from features if applicable.

```

Competing candidates were not selected because `_x` variables are more consistent with baseline/input-side condition features.

## Leakage Controls

- Remove `IRI_mean_y` from model features.
- Drop rows with missing `IRI_mean_y` for supervised modeling; do not impute target values.
- Treat `IRI_mean_x` as conditionally allowed only if it is baseline/current/pre-treatment IRI measured before `IRI_mean_y`.
- If temporal provenance of `IRI_mean_x` is unclear, run:
  1. a full baseline model with `IRI_mean_x`, and
  2. a strict no-prior-IRI model excluding `IRI_mean_x`.
- Exclude non-target `_y` columns from predictors if such columns appear later.

## Target Diagnostics

```json
{
  "dataset_id": "9bc07dd2ac12",
  "file_path": "/content/drive/MyDrive/Agentic-AI-Asset-management/Tamim/codes/Data/Maintenance-data/2017-2019-maintenance-dataset.csv",
  "state": "TARGET_SELECT",
  "target_column": "IRI_mean_y",
  "task_type": "regression",
  "target_dtype_original": "float64",
  "target_dtype_modeling": "numeric",
  "row_count": 5801,
  "non_null_count": 5801,
  "missing_count": 0,
  "missing_percent": 0.0,
  "coercion_fail_count": 0,
  "min": 30.0,
  "max": 388.0,
  "mean": 87.15614549215653,
  "median": 71.7,
  "standard_deviation": 49.58512461764218,
  "quantiles": {
    "0.01": 32.0,
    "0.05": 37.0,
    "0.25": 51.8,
    "0.5": 71.7,
    "0.75": 109.0,
    "0.95": 188.0,
    "0.99": 258.0
  }
}
```

## Artifacts

- `/content/pavement_agentic_workspace/target_selection.json`
- `/content/pavement_agentic_workspace/target_selection_report.md`
- `/content/pavement_agentic_workspace/target_diagnostics.json`
- `/content/pavement_agentic_workspace/target_diagnostics.csv`
- `/content/pavement_agentic_workspace/leakage_policy.json`
- `/content/pavement_agentic_workspace/plots/target_iri_mean_y_distribution.png`

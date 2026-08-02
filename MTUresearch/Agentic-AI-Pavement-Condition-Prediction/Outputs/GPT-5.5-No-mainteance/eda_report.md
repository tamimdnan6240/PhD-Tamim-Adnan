# EDA / Feature Engineering Design Report

## State

`EDA_FEATURE_ENGINEERING_DESIGN`

No model training, final preprocessing fitting, or target transformation fitting was performed in this state.

## Target and Task

- Target column: `IRI_mean_y`
- Task type: `regression`
- Target use: numeric continuous pavement IRI / ride-quality prediction target
- Feature rule: `IRI_mean_y` is excluded from all model feature sets.

## Dataset

- Dataset ID: `9bc07dd2ac12`
- File path: `/content/drive/MyDrive/Agentic-AI-Asset-management/Tamim/codes/Data/Maintenance-data/2017-2019-maintenance-dataset.csv`
- Shape: `5801` rows x `23` columns

## Leakage Controls

Hard-excluded columns:

```json
[
  "IRI_mean_y"
]
```

Conditionally allowed features:

```json
[
  "IRI_mean_x"
]
```

Availability-sensitive features:

```json
[
  "Future_AADT_mean_x"
]
```

`IRI_mean_x` is conditionally allowed only as a baseline/current/pre-outcome IRI feature. High correlation between `IRI_mean_x` and `IRI_mean_y` does not by itself prove leakage, but it increases the need for two-feature-set modeling.

`Future_AADT_mean_x` is not automatically banned solely because it contains `Future`; it is flagged for later availability-at-prediction-time review.

## Feature Sets

### baseline_condition_allowed

Includes valid pre-outcome features, including conditionally allowed `IRI_mean_x`.

```json
[
  "AADT_mean_x",
  "AADT_Single_Unit_mean_x",
  "AADT_Combination_mean_x",
  "Future_AADT_mean_x",
  "IRI_mean_x",
  "Thickness_Rigid_mean_x",
  "Thickness_Flexible_mean_x",
  "Base_Thickness_mean_x",
  "F_System_mode",
  "Urban_Type",
  "Surface_Type_mode",
  "Base_Type_mode_x",
  "Rutting_mean_x",
  "Cracking_Percent_mean_x",
  "Faulting_mean_x",
  "Last_Overlay_Thickness_mean_x",
  "Treatment_type",
  "RHU_AV_x",
  "FRZ_IDX_x",
  "TEMP_AVG_x",
  "PRECIPITATION_x",
  "Age_x"
]
```

### strict_no_prior_iri

Excludes both `IRI_mean_y` and `IRI_mean_x`.

```json
[
  "AADT_mean_x",
  "AADT_Single_Unit_mean_x",
  "AADT_Combination_mean_x",
  "Future_AADT_mean_x",
  "Thickness_Rigid_mean_x",
  "Thickness_Flexible_mean_x",
  "Base_Thickness_mean_x",
  "F_System_mode",
  "Urban_Type",
  "Surface_Type_mode",
  "Base_Type_mode_x",
  "Rutting_mean_x",
  "Cracking_Percent_mean_x",
  "Faulting_mean_x",
  "Last_Overlay_Thickness_mean_x",
  "Treatment_type",
  "RHU_AV_x",
  "FRZ_IDX_x",
  "TEMP_AVG_x",
  "PRECIPITATION_x",
  "Age_x"
]
```

## Feature Groups

```json
{
  "traffic_loading": [
    "AADT_mean_x",
    "AADT_Single_Unit_mean_x",
    "AADT_Combination_mean_x",
    "Future_AADT_mean_x"
  ],
  "pavement_structure": [
    "Thickness_Rigid_mean_x",
    "Thickness_Flexible_mean_x",
    "Base_Thickness_mean_x",
    "Last_Overlay_Thickness_mean_x",
    "Surface_Type_mode",
    "Base_Type_mode_x"
  ],
  "baseline_condition": [
    "IRI_mean_x",
    "Rutting_mean_x",
    "Cracking_Percent_mean_x",
    "Faulting_mean_x"
  ],
  "treatment": [
    "Treatment_type",
    "Last_Overlay_Thickness_mean_x"
  ],
  "climate_environment": [
    "RHU_AV_x",
    "FRZ_IDX_x",
    "TEMP_AVG_x",
    "PRECIPITATION_x"
  ],
  "age_system_context": [
    "Age_x",
    "F_System_mode",
    "Urban_Type"
  ],
  "availability_sensitive": [
    "Future_AADT_mean_x"
  ],
  "leakage_banned": [
    "IRI_mean_y"
  ]
}
```

## Preprocessing Plan

- Numeric imputation: median, fit inside training/CV folds only.
- Categorical imputation: `Unknown` or most frequent, fit inside training/CV folds only.
- Categorical encoding: one-hot encoding with `handle_unknown='ignore'`.
- Scaling: optional for linear/kernel/distance-sensitive models.
- Target: drop missing target rows; do not impute target.

## Top Numeric Correlations with Target

These correlations are for EDA association analysis only. `IRI_mean_y` must not be used as a feature.

- `IRI_mean_x`: 0.8511
- `F_System_mode`: 0.5698
- `AADT_Combination_mean_x`: -0.3524
- `AADT_Single_Unit_mean_x`: -0.3075
- `Base_Thickness_mean_x`: -0.2981
- `AADT_mean_x`: -0.2859
- `Future_AADT_mean_x`: -0.2842
- `Thickness_Rigid_mean_x`: -0.2206
- `Surface_Type_mode`: -0.1997
- `Cracking_Percent_mean_x`: 0.1569

## Artifacts

- EDA plan: `/content/pavement_agentic_workspace/eda_feature_engineering_plan.json`
- Feature groups: `/content/pavement_agentic_workspace/feature_groups.json`
- Feature audit: `/content/pavement_agentic_workspace/feature_audit.csv`
- Leakage screening: `/content/pavement_agentic_workspace/leakage_screening_report.json`
- Preprocessing plan: `/content/pavement_agentic_workspace/preprocessing_plan.json`
- EDA summary: `/content/pavement_agentic_workspace/eda_summary.json`
- Plot files: `/content/pavement_agentic_workspace/plots`

## Recommended Next Modeling Approach

Train and evaluate at least two model feature sets later:

1. `baseline_condition_allowed`: includes `IRI_mean_x`.
2. `strict_no_prior_iri`: excludes `IRI_mean_x`.

All preprocessing and feature engineering must be implemented inside modeling pipelines and fit only on train/CV folds.

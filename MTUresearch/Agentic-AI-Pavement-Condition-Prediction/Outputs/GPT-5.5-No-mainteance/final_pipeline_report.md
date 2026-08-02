# Final Pavement Condition Prediction Pipeline Report

## 1. Executive Summary

This final package documents an approved pavement condition prediction pipeline for:

- Target: `IRI_mean_y`
- Task: `regression`
- Best model: `HistGradientBoostingRegressor`
- Best feature set: `baseline_condition_allowed`

The approved best model achieved:

- RMSE: `21.9044`
- MAE: `14.5319`
- R²: `0.7957`
- Median absolute error: `8.9006`
- Explained variance: `0.7961`
- SMAPE: `16.0909`

The best model uses baseline/current IRI (`IRI_mean_x`) as a conditionally allowed feature. Deployment requires confirming that `IRI_mean_x` is truly measured before `IRI_mean_y`.

---

## 2. Dataset Governance

- `DATASET_ID`: `9bc07dd2ac12`
- `FILE_PATH`: `/content/drive/MyDrive/Agentic-AI-Asset-management/Tamim/codes/Data/Maintenance-data/2017-2019-maintenance-dataset.csv`
- Registry source: `mm.validate_registry(WORK_DIR)`
- File exists: `True`
- File path outside `WORK_DIR`: `True`
- `PINNED_FILE_PATH` matches registry: `True`

Dataset shape:

```json
{
  "rows": 5801,
  "columns": 23
}
```

Dataset columns as `List[str]`:

```python
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
  "IRI_mean_y",
  "RHU_AV_x",
  "FRZ_IDX_x",
  "TEMP_AVG_x",
  "PRECIPITATION_x",
  "Age_x"
]
```

---

## 3. Target Selection

Selected target: `IRI_mean_y`  
Task type: `regression`

`IRI_mean_y` was selected because International Roughness Index is a standard pavement ride-quality and condition measure. The `_y` suffix indicates this column is the outcome-side condition target in this pipeline.

---

## 4. Leakage Policy

Core leakage controls:

- `IRI_mean_y` is excluded from all feature matrices.
- `IRI_mean_x` is conditionally allowed only as baseline/current/pre-outcome IRI.
- Non-target `_y` variables, target-derived variables, delta IRI, improvement, deterioration, post-outcome condition variables, labels, and outcome columns are excluded.
- Both feature sets are retained for governance: `baseline_condition_allowed` and `strict_no_prior_iri`.

Feature-set leakage checks:

```json
{
  "IRI_mean_y_excluded_from_all_feature_sets": true,
  "IRI_mean_x_included_only_in_baseline_condition_allowed": true,
  "final_excluded_columns": [
    "IRI_mean_y"
  ]
}
```

---

## 5. EDA and Feature Groups

Approved feature groups include traffic/loading, pavement structure, baseline condition, treatment, climate/environment, and age/system/context variables. They are documented in `/content/pavement_agentic_workspace/feature_groups.json`.

---

## 6. Modeling Methodology

Train/test and CV strategy:

```json
{
  "dataset_id": "9bc07dd2ac12",
  "file_path": "/content/drive/MyDrive/Agentic-AI-Asset-management/Tamim/codes/Data/Maintenance-data/2017-2019-maintenance-dataset.csv",
  "state": "MODELING_PIPELINE_DESIGN",
  "random_state": 42,
  "test_size": 0.2,
  "train_row_count": 4640,
  "test_row_count": 1161,
  "target_summary_train": {
    "count": 4640,
    "missing_count": 0,
    "mean": 87.26375,
    "median": 71.7,
    "std": 49.860571760086394,
    "min": 30.0,
    "max": 388.0,
    "q25": 51.8,
    "q75": 108.825
  },
  "target_summary_test": {
    "count": 1161,
    "missing_count": 0,
    "mean": 86.72609819121448,
    "median": 71.7,
    "std": 48.48740255188228,
    "min": 30.0,
    "max": 355.0,
    "q25": 51.6,
    "q75": 109.0
  },
  "stratification_used": true,
  "stratification_notes": "Target quantile-bin stratification used for train/test split."
}
```

Models optimized:

- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `HistGradientBoostingRegressor`

All preprocessing was implemented inside sklearn pipelines. Imputers and encoders were fit only inside training/CV folds and not globally during EDA.

---

## 7. Model Results

Model comparison summary:

feature_set,model_name,cv_rmse_mean,test_rmse,test_mae,test_r2,test_smape
baseline_condition_allowed,HistGradientBoostingRegressor,23.05043521424192,21.904398318987035,14.531929524204388,0.7957420130339768,16.090872763505008
baseline_condition_allowed,ExtraTreesRegressor,22.77770286960612,22.46627562415801,14.479722848787992,0.7851286283153007,15.596839287136632
baseline_condition_allowed,RandomForestRegressor,22.80006053858534,22.504808915540497,14.763631090128683,0.7843909179955373,15.975567155475463
strict_no_prior_iri,HistGradientBoostingRegressor,34.85040718735278,34.1714731679602,23.599168816190765,0.5028999633779379,25.66062298821986
strict_no_prior_iri,RandomForestRegressor,34.71724215091341,34.577843140794535,23.826481800905288,0.4910065535436996,25.979032471617423
strict_no_prior_iri,ExtraTreesRegressor,34.916195052869895,34.64909485765422,23.187918210799072,0.4889067104366109,24.618336179760057


Best approved model:

```json
{
  "model_name": "HistGradientBoostingRegressor",
  "feature_set": "baseline_condition_allowed",
  "metrics": {
    "test_rmse": 21.904398318987035,
    "test_mae": 14.531929524204388,
    "test_r2": 0.7957420130339768,
    "test_median_absolute_error": 8.900613844655837,
    "test_explained_variance": 0.796140995434934,
    "test_smape": 16.090872763505008
  }
}
```

Strict no-prior-IRI benchmark:

```json
{
  "model_name": "HistGradientBoostingRegressor",
  "feature_set": "strict_no_prior_iri",
  "metrics": {
    "test_rmse": 34.1714731679602,
    "test_mae": 23.599168816190765,
    "test_r2": 0.5028999633779379
  }
}
```

---

## 8. `IRI_mean_x` Sensitivity

Approved sensitivity summary:

```json
{
  "delta_rmse_strict_minus_baseline": 12.267074848973166,
  "delta_mae_strict_minus_baseline": 9.067239291986377,
  "delta_r2_strict_minus_baseline": -0.292842049656039,
  "interpretation": "Performance drops substantially when IRI_mean_x is excluded. This is engineering-plausible because pavement roughness persists over time, but deployment requires confirming that IRI_mean_x is true baseline/pre-outcome IRI."
}
```

Removing `IRI_mean_x` worsened model performance substantially:

- ΔRMSE strict-minus-baseline: `12.2671`
- ΔMAE strict-minus-baseline: `9.0672`
- ΔR² strict-minus-baseline: `-0.2928`

This does not automatically prove leakage. Pavement roughness is persistent over time, so prior IRI is expected to be highly predictive of future or post-period IRI. However, it is the primary deployment governance issue.

---

## 9. Artifact Organization

Metrics/XAI artifacts are organized under:

```text
/content/pavement_agentic_workspace/metrics
```

Model artifacts are organized under:

```text
/content/pavement_agentic_workspace/models
```

Plot artifacts are organized under:

```text
/content/pavement_agentic_workspace/plots
```

Prediction artifacts are organized under:

```text
/content/pavement_agentic_workspace/predictions
```

---

## 10. Pavement-Engineering Interpretation

The approved best model performance is strong for network-level IRI prediction:

- RMSE ≈ `21.9`
- MAE ≈ `14.5`
- R² ≈ `0.80`

In common U.S. pavement-management interpretation, IRI is often measured in inches per mile. Lower IRI indicates smoother pavement and better ride quality; higher IRI indicates rougher pavement and greater maintenance concern.

The accuracy is suitable for network-level forecasting, risk screening, treatment prioritization support, and budget/scenario planning. It should not be treated as a sole project-level design or treatment decision tool.

The dependence on `IRI_mean_x` is engineering-plausible because roughness persists over time. A road that is rough at baseline is likely to remain rough or deteriorate unless treated. But use of `IRI_mean_x` requires strict temporal provenance confirmation.

---

## 11. Asset-Management Use Cases

Potential agency uses include network-level condition forecasting, identifying sections likely to cross good/fair/poor IRI thresholds, prioritizing candidate maintenance or rehabilitation projects, budget scenario planning, performance management reporting, and risk screening for high-traffic or climate-exposed corridors.

Use predictions with uncertainty, especially near agency IRI thresholds.

---

## 12. Deployment Cautions

Primary cautions:

- Confirm `IRI_mean_x` is measured before `IRI_mean_y`.
- Confirm baseline distress variables are pre-outcome.
- Confirm treatment variables are known before prediction and not post-outcome leakage.
- Confirm `Future_AADT_mean_x` is available at planning/prediction time.
- Do not treat predictive feature importance as causal effect.
- Avoid using the model as the only basis for project-level treatment decisions.
- Report uncertainty near agency good/fair/poor IRI thresholds.

---

## 13. Future Validation Recommendations

Recommended next validation:

1. Time-based validation: train on earlier years and test on later years.
2. Route/segment grouped validation to avoid overestimating performance due to nearby or repeated segment records.
3. District/spatial validation by region, district, climate zone, and urban/rural context.
4. Threshold-based error analysis near agency IRI treatment and reporting thresholds.
5. Residual review by pavement type, treatment, traffic/loading, climate, rutting/cracking/faulting, functional system, and district/route.

---

## 14. Final Operational Recommendation

Use the `baseline_condition_allowed` model when current/pre-outcome IRI is verified available at prediction time. Use the `strict_no_prior_iri` model as the conservative benchmark when `IRI_mean_x` timing or availability is uncertain.

The pipeline is ready for governed validation and pilot deployment review, not uncontrolled production use.

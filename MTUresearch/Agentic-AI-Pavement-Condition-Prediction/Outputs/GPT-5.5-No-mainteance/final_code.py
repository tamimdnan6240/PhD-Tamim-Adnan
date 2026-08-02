from mm_runtime import mm
mm.health_check()

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

WORK_DIR = "/content/pavement_agentic_workspace"
STATE_NOW = "FINAL_REPORTING_ARCHIVE"
STATE_TASK = "FINAL_REPORTING_ARCHIVE"
TARGET_COLUMN = "IRI_mean_y"
TASK_TYPE = "regression"

work_dir = Path(WORK_DIR)
work_dir.mkdir(parents=True, exist_ok=True)

# Registry is the ONLY source of dataset path.
reg = mm.validate_registry(WORK_DIR)
FILE_PATH = reg["latest_path"]
DATASET_ID = reg["latest_dataset_id"]

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(FILE_PATH)

work_dir_resolved = work_dir.resolve()
file_path_resolved = Path(FILE_PATH).resolve()
try:
    file_path_resolved.relative_to(work_dir_resolved)
    raise RuntimeError(f"Registry FILE_PATH must not be inside WORK_DIR: {FILE_PATH}")
except ValueError:
    pass

pinned = os.environ.get("PINNED_FILE_PATH")
if pinned and str(Path(pinned).resolve()) != str(file_path_resolved):
    raise RuntimeError(
        "PINNED_FILE_PATH does not match registry latest_path. "
        f"PINNED_FILE_PATH={pinned}; registry latest_path={FILE_PATH}"
    )

base_meta = {"dataset_id": DATASET_ID, "file_path": FILE_PATH, "state": STATE_NOW}

metrics_dir = work_dir / "metrics"
models_dir = work_dir / "models"
plots_dir = work_dir / "plots"
predictions_dir = work_dir / "predictions"
for d in [metrics_dir, models_dir, plots_dir, predictions_dir]:
    d.mkdir(parents=True, exist_ok=True)

final_report_path = work_dir / "final_pipeline_report.md"
final_summary_path = work_dir / "final_pipeline_summary.json"
artifact_manifest_path = work_dir / "artifact_manifest.json"
deployment_checklist_path = work_dir / "deployment_readiness_checklist.md"
final_code_path = work_dir / "final_code.py"

required_input_artifacts = {
    "discovery_dataset_profile": work_dir / "dataset_profile.json",
    "target_candidates": work_dir / "target_candidates.csv",
    "target_selection": work_dir / "target_selection.json",
    "target_diagnostics": work_dir / "target_diagnostics.json",
    "leakage_policy": work_dir / "leakage_policy.json",
    "eda_feature_engineering_plan": work_dir / "eda_feature_engineering_plan.json",
    "feature_groups": work_dir / "feature_groups.json",
    "feature_audit": work_dir / "feature_audit.csv",
    "leakage_screening": work_dir / "leakage_screening_report.json",
    "preprocessing_plan": work_dir / "preprocessing_plan.json",
    "eda_summary": work_dir / "eda_summary.json",
    "eda_report": work_dir / "eda_report.md",
    "modeling_plan": work_dir / "modeling_plan.json",
    "modeling_feature_sets": work_dir / "modeling_feature_sets.json",
    "train_test_split_metadata": work_dir / "train_test_split_metadata.json",
    "model_comparison_report": work_dir / "model_comparison_report.md",
    "root_model_metrics": work_dir / "model_metrics.json",
    "metrics_model_performance": metrics_dir / "model_performance_metrics.csv",
    "metrics_model_metrics": metrics_dir / "model_metrics.json",
    "metrics_model_metrics_long": metrics_dir / "model_metrics_long.csv",
    "metrics_feature_importance_best": metrics_dir / "feature_importance_best_models.csv",
    "best_model": models_dir / "best_model.joblib",
    "best_model_metadata": models_dir / "best_model_metadata.json",
}
required_modeling_plots = [
    plots_dir / "model_comparison_test_rmse.png",
    plots_dir / "model_comparison_test_mae.png",
    plots_dir / "model_comparison_test_r2.png",
    plots_dir / "cv_rmse_by_model.png",
    plots_dir / "observed_vs_predicted_best_model.png",
    plots_dir / "residuals_vs_predicted_best_model.png",
    plots_dir / "residual_distribution_best_model.png",
    plots_dir / "absolute_error_by_feature_set.png",
    plots_dir / "iri_mean_x_sensitivity_comparison.png",
    plots_dir / "feature_importance_best_model.png",
]

def rag_search(kind, query, k=10):
    return mm.rag_search(query=query, kind=kind, k=k, meta_filters={"dataset_id": DATASET_ID})

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if pd.isna(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

def file_info(path, artifact_type, state_created_or_used, notes="", group=""):
    p = Path(path)
    return {
        "artifact_path": str(p),
        "artifact_type": artifact_type,
        "artifact_group": group,
        "state_created_or_used": state_created_or_used,
        "exists": bool(p.exists()),
        "file_size_bytes": int(p.stat().st_size) if p.exists() and p.is_file() else None,
        "notes": notes,
    }

# Required memory retrievals and upstream state verification.
target_candidates_memory_hits = mm.rag_search(kind="target_candidates", query="target_candidates.csv", k=3, meta_filters={"dataset_id": DATASET_ID})
dataset_artifact_memory_hits = mm.rag_search(kind="dataset_artifact", query="COLUMNS", k=3, meta_filters={"dataset_id": DATASET_ID})
model_metrics_memory_hits = mm.rag_search(kind="model_metrics", query="MODEL_METRICS", k=3, meta_filters={"dataset_id": DATASET_ID})
model_metrics_per_model_memory_hits = mm.rag_search(kind="model_metrics_per_model", query="metrics", k=10, meta_filters={"dataset_id": DATASET_ID})

required_upstream_states = ["DISCOVER_DATASET", "TARGET_SELECT", "EDA_FEATURE_ENGINEERING_DESIGN", "MODELING_PIPELINE_DESIGN"]
upstream_state_hits = {}
missing_upstream_states = []
for st in required_upstream_states:
    hits = rag_search(kind="pipeline_state", query=f"PIPELINE_STATE_COMPLETED={st}", k=10)
    upstream_state_hits[st] = len(hits)
    if not hits:
        missing_upstream_states.append(st)
if missing_upstream_states:
    raise RuntimeError("Cannot run FINAL_REPORTING_ARCHIVE because upstream states are missing: " + "; ".join(missing_upstream_states))
prior_final_state_hits = rag_search(kind="pipeline_state", query="PIPELINE_STATE_COMPLETED=FINAL_REPORTING_ARCHIVE", k=10)
state_already_completed_before_run = bool(prior_final_state_hits)

# Verify input artifacts/directories.
missing_required_artifacts = [f"{name}: {path}" for name, path in required_input_artifacts.items() if not path.exists()]
for dname, dpath in {"metrics": metrics_dir, "models": models_dir, "plots": plots_dir, "predictions": predictions_dir}.items():
    if not dpath.exists() or not dpath.is_dir():
        missing_required_artifacts.append(f"{dname}_dir: {dpath}")
missing_required_artifacts.extend([str(p) for p in required_modeling_plots if not p.exists()])
model_joblib_files = sorted(models_dir.glob("*.joblib"))
prediction_csv_files = sorted(predictions_dir.glob("*.csv"))
plot_files = sorted(plots_dir.glob("*.png"))
metrics_csv_files = sorted(metrics_dir.glob("*.csv"))
if not model_joblib_files: missing_required_artifacts.append(f"no .joblib model artifacts under {models_dir}")
if not prediction_csv_files: missing_required_artifacts.append(f"no prediction CSV artifacts under {predictions_dir}")
if not plot_files: missing_required_artifacts.append(f"no PNG plots under {plots_dir}")
if missing_required_artifacts:
    raise RuntimeError("Missing required input artifacts for FINAL_REPORTING_ARCHIVE: " + "; ".join(missing_required_artifacts))

# Load dataset (mandatory) and print/log.
df = pd.read_csv(FILE_PATH, low_memory=False)
dataset_columns = [str(c) for c in df.columns.tolist()]
df.columns = dataset_columns
dataset_shape = df.shape
print("FILE_PATH:", FILE_PATH)
print("DATASET_ID:", DATASET_ID)
print("df.shape:", df.shape)
print("columns List[str]:", dataset_columns)
print("df.head(5):")
print(df.head(5))
prior_art = mm.rag_get_latest(kind="dataset_artifact", meta_filters={"dataset_id": DATASET_ID})
if prior_art and prior_art.get("meta") and prior_art["meta"].get("columns"):
    if prior_art["meta"]["columns"] != dataset_columns:
        raise RuntimeError("Column mismatch vs prior artifact")
mm.rag_add(kind="dataset_artifact", text="Loaded dataset (see meta for path, shape, columns)", meta={"dataset_id": DATASET_ID, "file_path": FILE_PATH, "shape": list(df.shape), "ncols": int(df.shape[1]), "columns": dataset_columns, "state": STATE_NOW})

# Load key artifacts and validate approved target/model claims.
with open(required_input_artifacts["discovery_dataset_profile"], "r", encoding="utf-8") as f: dataset_profile = json.load(f)
with open(required_input_artifacts["target_selection"], "r", encoding="utf-8") as f: target_selection = json.load(f)
with open(required_input_artifacts["leakage_policy"], "r", encoding="utf-8") as f: leakage_policy = json.load(f)
with open(required_input_artifacts["leakage_screening"], "r", encoding="utf-8") as f: leakage_screening = json.load(f)
with open(required_input_artifacts["modeling_feature_sets"], "r", encoding="utf-8") as f: modeling_feature_sets = json.load(f)
with open(required_input_artifacts["train_test_split_metadata"], "r", encoding="utf-8") as f: split_metadata = json.load(f)
with open(required_input_artifacts["metrics_model_metrics"], "r", encoding="utf-8") as f: model_metrics = json.load(f)
with open(required_input_artifacts["best_model_metadata"], "r", encoding="utf-8") as f: best_model_metadata = json.load(f)
metrics_df = pd.read_csv(required_input_artifacts["metrics_model_performance"])
metrics_long_df = pd.read_csv(required_input_artifacts["metrics_model_metrics_long"])

if target_selection.get("target_column") != TARGET_COLUMN: raise RuntimeError("target_selection.json does not confirm target_column=IRI_mean_y")
if target_selection.get("task_type") != TASK_TYPE: raise RuntimeError("target_selection.json does not confirm task_type=regression")
if TARGET_COLUMN not in leakage_policy.get("banned_features", []): raise RuntimeError("leakage_policy.json does not ban IRI_mean_y from features.")
feature_sets = modeling_feature_sets.get("feature_sets", {})
if TARGET_COLUMN in feature_sets.get("baseline_condition_allowed", []): raise RuntimeError("IRI_mean_y leaked into baseline_condition_allowed.")
if TARGET_COLUMN in feature_sets.get("strict_no_prior_iri", []): raise RuntimeError("IRI_mean_y leaked into strict_no_prior_iri.")
if "IRI_mean_x" not in feature_sets.get("baseline_condition_allowed", []): raise RuntimeError("IRI_mean_x missing from baseline_condition_allowed.")
if "IRI_mean_x" in feature_sets.get("strict_no_prior_iri", []): raise RuntimeError("IRI_mean_x leaked into strict_no_prior_iri.")
required_long_schema = ["dataset_id", "file_path", "state", "feature_set", "model_name", "metric_name", "value", "timestamp"]
if list(metrics_long_df.columns) != required_long_schema: raise RuntimeError(f"model_metrics_long.csv schema invalid: {list(metrics_long_df.columns)}")

best_model_from_json = model_metrics.get("best_model", {})
best_model_name = best_model_from_json.get("model_name") or best_model_metadata.get("best_model_name")
best_feature_set = best_model_from_json.get("feature_set") or best_model_metadata.get("best_feature_set")
if best_model_name != "HistGradientBoostingRegressor": raise RuntimeError(f"Unexpected best model: {best_model_name}")
if best_feature_set != "baseline_condition_allowed": raise RuntimeError(f"Unexpected best feature set: {best_feature_set}")
best_metrics = {
    "test_rmse": safe_float(best_model_from_json.get("test_rmse", best_model_metadata.get("metrics", {}).get("test_rmse"))),
    "test_mae": safe_float(best_model_from_json.get("test_mae", best_model_metadata.get("metrics", {}).get("test_mae"))),
    "test_r2": safe_float(best_model_from_json.get("test_r2", best_model_metadata.get("metrics", {}).get("test_r2"))),
    "test_median_absolute_error": safe_float(best_model_metadata.get("metrics", {}).get("test_median_absolute_error")),
    "test_explained_variance": safe_float(best_model_metadata.get("metrics", {}).get("test_explained_variance")),
    "test_smape": safe_float(best_model_metadata.get("metrics", {}).get("test_smape")),
}
best_row = metrics_df.sort_values(["test_rmse", "test_mae"]).iloc[0].to_dict()
for key in list(best_metrics.keys()):
    if best_metrics[key] is None and key in best_row:
        best_metrics[key] = safe_float(best_row[key])
best_by_feature_set = {}
for fs in ["baseline_condition_allowed", "strict_no_prior_iri"]:
    sub = metrics_df[metrics_df["feature_set"].astype(str) == fs].sort_values(["test_rmse", "test_mae"])
    if len(sub) == 0: raise RuntimeError(f"Missing metrics for feature set {fs}")
    best_by_feature_set[fs] = sub.iloc[0].to_dict()
baseline_best = best_by_feature_set["baseline_condition_allowed"]
strict_best = best_by_feature_set["strict_no_prior_iri"]
iri_sensitivity = model_metrics.get("iri_mean_x_sensitivity_summary", {})
if not iri_sensitivity:
    iri_sensitivity = {"delta_rmse_strict_minus_baseline": safe_float(strict_best["test_rmse"] - baseline_best["test_rmse"]), "delta_mae_strict_minus_baseline": safe_float(strict_best["test_mae"] - baseline_best["test_mae"]), "delta_r2_strict_minus_baseline": safe_float(strict_best["test_r2"] - baseline_best["test_r2"])}
delta_rmse = safe_float(iri_sensitivity.get("delta_RMSE_strict_minus_baseline", iri_sensitivity.get("delta_rmse_strict_minus_baseline")))
delta_mae = safe_float(iri_sensitivity.get("delta_MAE_strict_minus_baseline", iri_sensitivity.get("delta_mae_strict_minus_baseline")))
delta_r2 = safe_float(iri_sensitivity.get("delta_R2_strict_minus_baseline", iri_sensitivity.get("delta_r2_strict_minus_baseline")))

approved_states = ["DISCOVER_DATASET", "TARGET_SELECT", "EDA_FEATURE_ENGINEERING_DESIGN", "MODELING_PIPELINE_DESIGN", "DOMAIN_INTERPRETATION"]
final_summary = {
    "dataset_id": DATASET_ID,
    "file_path": FILE_PATH,
    "target_column": TARGET_COLUMN,
    "task_type": TASK_TYPE,
    "approved_states": approved_states,
    "dataset_shape": {"rows": int(dataset_shape[0]), "columns": int(dataset_shape[1])},
    "dataset_columns": dataset_columns,
    "best_model_name": best_model_name,
    "best_feature_set": best_feature_set,
    "best_model": {"model_name": best_model_name, "feature_set": best_feature_set, "metrics": best_metrics},
    "strict_no_prior_iri_best_model": {"model_name": str(strict_best["model_name"]), "feature_set": "strict_no_prior_iri", "metrics": {"test_rmse": safe_float(strict_best["test_rmse"]), "test_mae": safe_float(strict_best["test_mae"]), "test_r2": safe_float(strict_best["test_r2"])}},
    "iri_mean_x_sensitivity": {"delta_rmse_strict_minus_baseline": delta_rmse, "delta_mae_strict_minus_baseline": delta_mae, "delta_r2_strict_minus_baseline": delta_r2, "interpretation": "Performance drops substantially when IRI_mean_x is excluded. This is engineering-plausible because pavement roughness persists over time, but deployment requires confirming that IRI_mean_x is true baseline/pre-outcome IRI."},
    "artifact_directories": {"metrics": str(metrics_dir), "models": str(models_dir), "plots": str(plots_dir), "predictions": str(predictions_dir)},
    "core_artifacts": {"final_pipeline_report": str(final_report_path), "final_pipeline_summary": str(final_summary_path), "artifact_manifest": str(artifact_manifest_path), "deployment_readiness_checklist": str(deployment_checklist_path), "metrics_long": str(required_input_artifacts["metrics_model_metrics_long"]), "best_model": str(required_input_artifacts["best_model"]), "best_model_metadata": str(required_input_artifacts["best_model_metadata"])},
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
if final_summary["target_column"] != TARGET_COLUMN or final_summary["task_type"] != TASK_TYPE: raise RuntimeError("Final summary target/task validation failed.")
if final_summary["best_model_name"] != "HistGradientBoostingRegressor": raise RuntimeError("Final summary best_model_name validation failed.")
if final_summary["best_feature_set"] != "baseline_condition_allowed": raise RuntimeError("Final summary best_feature_set validation failed.")
with open(final_summary_path, "w", encoding="utf-8") as f: json.dump(json_clean(final_summary), f, indent=2)

# Deployment checklist.
deployment_cautions = ["IRI_mean_x temporal provenance", "Future_AADT_mean_x availability", "treatment timing", "need for temporal/grouped validation"]
deployment_checklist_md = f"""# Deployment Readiness Checklist

Pipeline target: `{TARGET_COLUMN}`  
Task: `{TASK_TYPE}`  
Best approved model: `{best_model_name}`  
Best approved feature set: `{best_feature_set}`

## Temporal and Leakage Governance

- [ ] Confirm `IRI_mean_x` is measured before `IRI_mean_y`.
- [ ] Confirm baseline distress variables are measured before the target outcome period.
- [ ] Confirm `Treatment_type` timing relative to prediction target.
- [ ] Confirm `Last_Overlay_Thickness_mean_x` is available before prediction and is not post-outcome leakage.
- [ ] Confirm `Future_AADT_mean_x` is a forecast available at prediction/planning time.
- [ ] Confirm no target-derived variables, delta IRI, improvement, deterioration, or post-outcome condition variables enter the feature matrix.
- [ ] Maintain both feature-set views in governance documentation: `baseline_condition_allowed` and `strict_no_prior_iri`.

## Validation Before Operational Use

- [ ] Validate model on a later time period.
- [ ] Validate model using route/segment grouped split.
- [ ] Validate model by district, region, or climate zone.
- [ ] Review residuals by pavement type, treatment, traffic, climate, and distress category.
- [ ] Review errors for high-IRI and recently treated sections.
- [ ] Compare performance of baseline-IRI and no-prior-IRI models in the intended deployment setting.

## Agency Decision Integration

- [ ] Define agency-specific IRI good/fair/poor thresholds.
- [ ] Add uncertainty handling near good/fair/poor thresholds.
- [ ] Avoid using predictions as the sole basis for project-level treatment selection.
- [ ] Combine predictions with safety, structural condition, funding, equity, route importance, and engineering judgment.
- [ ] Establish model monitoring and retraining schedule.
- [ ] Document intended prediction horizon and data refresh cadence.

## Approved Performance Reference

Best model with baseline IRI:

- RMSE: `{best_metrics['test_rmse']:.4f}`
- MAE: `{best_metrics['test_mae']:.4f}`
- R²: `{best_metrics['test_r2']:.4f}`

Strict no-prior-IRI benchmark:

- RMSE: `{safe_float(strict_best['test_rmse']):.4f}`
- MAE: `{safe_float(strict_best['test_mae']):.4f}`
- R²: `{safe_float(strict_best['test_r2']):.4f}`

Primary caution:

> The best model depends materially on `IRI_mean_x`. This is pavement-engineering plausible, but operational use requires proving `IRI_mean_x` is true baseline/pre-outcome IRI available at prediction time.
"""
with open(deployment_checklist_path, "w", encoding="utf-8") as f: f.write(deployment_checklist_md)

# Final report.
try:
    metrics_table_md = metrics_df[["feature_set", "model_name", "cv_rmse_mean", "test_rmse", "test_mae", "test_r2", "test_smape"]].sort_values(["test_rmse", "test_mae"]).to_markdown(index=False)
except Exception:
    metrics_table_md = metrics_df[["feature_set", "model_name", "cv_rmse_mean", "test_rmse", "test_mae", "test_r2", "test_smape"]].sort_values(["test_rmse", "test_mae"]).to_csv(index=False)
final_report_md = f"""# Final Pavement Condition Prediction Pipeline Report

## 1. Executive Summary

This final package documents an approved pavement condition prediction pipeline for:

- Target: `{TARGET_COLUMN}`
- Task: `{TASK_TYPE}`
- Best model: `{best_model_name}`
- Best feature set: `{best_feature_set}`

The approved best model achieved:

- RMSE: `{best_metrics['test_rmse']:.4f}`
- MAE: `{best_metrics['test_mae']:.4f}`
- R²: `{best_metrics['test_r2']:.4f}`
- Median absolute error: `{best_metrics['test_median_absolute_error']:.4f}`
- Explained variance: `{best_metrics['test_explained_variance']:.4f}`
- SMAPE: `{best_metrics['test_smape']:.4f}`

The best model uses baseline/current IRI (`IRI_mean_x`) as a conditionally allowed feature. Deployment requires confirming that `IRI_mean_x` is truly measured before `IRI_mean_y`.

---

## 2. Dataset Governance

- `DATASET_ID`: `{DATASET_ID}`
- `FILE_PATH`: `{FILE_PATH}`
- Registry source: `mm.validate_registry(WORK_DIR)`
- File exists: `{os.path.exists(FILE_PATH)}`
- File path outside `WORK_DIR`: `True`
- `PINNED_FILE_PATH` matches registry: `{True if not pinned else str(Path(pinned).resolve()) == str(file_path_resolved)}`

Dataset shape:

```json
{json.dumps(final_summary['dataset_shape'], indent=2)}
```

Dataset columns as `List[str]`:

```python
{json.dumps(dataset_columns, indent=2)}
```

---

## 3. Target Selection

Selected target: `{TARGET_COLUMN}`  
Task type: `{TASK_TYPE}`

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
{json.dumps(modeling_feature_sets.get('leakage_checks', {}), indent=2)}
```

---

## 5. EDA and Feature Groups

Approved feature groups include traffic/loading, pavement structure, baseline condition, treatment, climate/environment, and age/system/context variables. They are documented in `{required_input_artifacts['feature_groups']}`.

---

## 6. Modeling Methodology

Train/test and CV strategy:

```json
{json.dumps(split_metadata, indent=2)}
```

Models optimized:

- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `HistGradientBoostingRegressor`

All preprocessing was implemented inside sklearn pipelines. Imputers and encoders were fit only inside training/CV folds and not globally during EDA.

---

## 7. Model Results

Model comparison summary:

{metrics_table_md}

Best approved model:

```json
{json.dumps(final_summary['best_model'], indent=2)}
```

Strict no-prior-IRI benchmark:

```json
{json.dumps(final_summary['strict_no_prior_iri_best_model'], indent=2)}
```

---

## 8. `IRI_mean_x` Sensitivity

Approved sensitivity summary:

```json
{json.dumps(final_summary['iri_mean_x_sensitivity'], indent=2)}
```

Removing `IRI_mean_x` worsened model performance substantially:

- ΔRMSE strict-minus-baseline: `{delta_rmse:.4f}`
- ΔMAE strict-minus-baseline: `{delta_mae:.4f}`
- ΔR² strict-minus-baseline: `{delta_r2:.4f}`

This does not automatically prove leakage. Pavement roughness is persistent over time, so prior IRI is expected to be highly predictive of future or post-period IRI. However, it is the primary deployment governance issue.

---

## 9. Artifact Organization

Metrics/XAI artifacts are organized under:

```text
{metrics_dir}
```

Model artifacts are organized under:

```text
{models_dir}
```

Plot artifacts are organized under:

```text
{plots_dir}
```

Prediction artifacts are organized under:

```text
{predictions_dir}
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
"""
with open(final_report_path, "w", encoding="utf-8") as f: f.write(final_report_md)

# Save complete executed code.
try:
    with open(__file__, "r", encoding="utf-8") as src:
        executed_source = src.read()
    with open(final_code_path, "w", encoding="utf-8") as dst:
        dst.write(executed_source)
except Exception as exc:
    raise RuntimeError(f"Failed to save complete executed code to {final_code_path}: {exc}")

# Artifact manifest after final artifacts exist.
manifest_entries = []
for name, dpath in {"metrics": metrics_dir, "models": models_dir, "plots": plots_dir, "predictions": predictions_dir}.items():
    manifest_entries.append(file_info(dpath, "directory", "MODELING_PIPELINE_DESIGN/FINAL_REPORTING_ARCHIVE", f"Organized {name} artifact directory.", name if name != "metrics" else "metrics_xai"))
group_map = {
    "discovery_dataset_profile": "discovery", "target_candidates": "target_selection", "target_selection": "target_selection", "target_diagnostics": "target_selection", "leakage_policy": "target_selection",
    "eda_feature_engineering_plan": "eda_feature_engineering", "feature_groups": "eda_feature_engineering", "feature_audit": "eda_feature_engineering", "leakage_screening": "eda_feature_engineering", "preprocessing_plan": "eda_feature_engineering", "eda_summary": "eda_feature_engineering", "eda_report": "eda_feature_engineering",
    "modeling_plan": "reports", "modeling_feature_sets": "reports", "train_test_split_metadata": "reports", "model_comparison_report": "reports", "root_model_metrics": "reports",
    "metrics_model_performance": "metrics_xai", "metrics_model_metrics": "metrics_xai", "metrics_model_metrics_long": "metrics_xai", "metrics_feature_importance_best": "metrics_xai", "best_model": "models", "best_model_metadata": "models",
}
for name, path in required_input_artifacts.items():
    manifest_entries.append(file_info(path, "file", "approved_upstream", name, group_map.get(name, "reports")))
for p in sorted(metrics_dir.glob("*")):
    if p.is_file(): manifest_entries.append(file_info(p, "file", "MODELING_PIPELINE_DESIGN", "Organized metrics/XAI artifact.", "metrics_xai"))
for p in sorted(models_dir.glob("*")):
    if p.is_file(): manifest_entries.append(file_info(p, "file", "MODELING_PIPELINE_DESIGN", "Trained model or model metadata artifact.", "models"))
for p in sorted(predictions_dir.glob("*.csv")):
    manifest_entries.append(file_info(p, "file", "MODELING_PIPELINE_DESIGN", "Holdout prediction artifact.", "predictions"))
for p in sorted(plots_dir.glob("*.png")):
    manifest_entries.append(file_info(p, "file", "DISCOVER/EDA/MODELING", "Diagnostic or model comparison plot.", "plots"))
for p, note in [(final_report_path, "Final human-readable pipeline report."), (final_summary_path, "Final machine-readable pipeline summary."), (artifact_manifest_path, "Artifact manifest."), (deployment_checklist_path, "Deployment readiness checklist."), (final_code_path, "Complete executed code for latest state.")]:
    manifest_entries.append(file_info(p, "file", STATE_NOW, note, "final_package"))
artifact_group_names = sorted(set(e["artifact_group"] for e in manifest_entries if e.get("artifact_group")))
artifact_manifest = {"dataset_id": DATASET_ID, "file_path": FILE_PATH, "state": STATE_NOW, "created_at_utc": datetime.now(timezone.utc).isoformat(), "artifact_group_names": artifact_group_names, "artifact_count": int(len(manifest_entries)), "artifacts": manifest_entries}
with open(artifact_manifest_path, "w", encoding="utf-8") as f: json.dump(json_clean(artifact_manifest), f, indent=2)

# Verify final artifacts.
required_final_artifacts = [final_report_path, final_summary_path, artifact_manifest_path, deployment_checklist_path, final_code_path]
missing_final = [str(p) for p in required_final_artifacts if not p.exists()]
if missing_final: raise RuntimeError("Missing required final artifacts: " + "; ".join(missing_final))
with open(final_summary_path, "r", encoding="utf-8") as f: final_summary_check = json.load(f)
if final_summary_check.get("target_column") != TARGET_COLUMN: raise RuntimeError("final_pipeline_summary.json target_column invalid.")
if final_summary_check.get("task_type") != TASK_TYPE: raise RuntimeError("final_pipeline_summary.json task_type invalid.")
if final_summary_check.get("best_model_name") != "HistGradientBoostingRegressor": raise RuntimeError("final_pipeline_summary.json best_model_name invalid.")
if final_summary_check.get("best_feature_set") != "baseline_condition_allowed": raise RuntimeError("final_pipeline_summary.json best_feature_set invalid.")
with open(artifact_manifest_path, "r", encoding="utf-8") as f: manifest_check = json.load(f)
manifest_paths = [entry["artifact_path"] for entry in manifest_check.get("artifacts", [])]
for required_dir in [str(metrics_dir), str(models_dir), str(plots_dir), str(predictions_dir)]:
    if required_dir not in manifest_paths: raise RuntimeError(f"artifact_manifest.json missing organized directory: {required_dir}")
report_text = final_report_path.read_text(encoding="utf-8")
for phrase in ["IRI_mean_y", "regression", "IRI_mean_x", "Pavement-Engineering Interpretation", str(metrics_dir), str(models_dir), str(plots_dir), str(predictions_dir), "Future Validation Recommendations"]:
    if phrase not in report_text: raise RuntimeError(f"final_pipeline_report.md missing required phrase: {phrase}")

# MM records.
final_artifacts_saved = [str(p) for p in required_final_artifacts]
mm.rag_add(kind="final_report", text="FINAL_REPORTING_ARCHIVE final pipeline report for IRI_mean_y regression.", meta={**base_meta, "target_column": TARGET_COLUMN, "task_type": TASK_TYPE, "best_model_name": best_model_name, "best_feature_set": best_feature_set, "final_pipeline_report_path": str(final_report_path), "artifacts_saved": final_artifacts_saved})
mm.rag_add(kind="artifact_manifest", text="Artifact manifest for final pavement prediction pipeline package.", meta={**base_meta, "artifact_manifest_path": str(artifact_manifest_path), "artifact_group_names": artifact_group_names, "artifact_count": int(len(manifest_check.get("artifacts", []))), "artifacts_saved": [str(artifact_manifest_path)]})
mm.rag_add(kind="deployment_readiness", text="Deployment readiness checklist for IRI_mean_y pavement prediction model.", meta={**base_meta, "deployment_readiness_checklist_path": str(deployment_checklist_path), "key_cautions": deployment_cautions, "artifacts_saved": [str(deployment_checklist_path)]})
mm.rag_add(kind="final_pipeline_summary", text="Final pipeline summary for approved IRI_mean_y regression model.", meta={**base_meta, "final_pipeline_summary_path": str(final_summary_path), "best_model_name": best_model_name, "best_feature_set": best_feature_set, "best_test_rmse": best_metrics["test_rmse"], "best_test_mae": best_metrics["test_mae"], "best_test_r2": best_metrics["test_r2"], "artifacts_saved": [str(final_summary_path)]})
final_report_hits = rag_search(kind="final_report", query="FINAL_REPORTING_ARCHIVE final pipeline report for IRI_mean_y regression", k=10)
artifact_manifest_hits = rag_search(kind="artifact_manifest", query="Artifact manifest for final pavement prediction pipeline package", k=10)
deployment_readiness_hits = rag_search(kind="deployment_readiness", query="Deployment readiness checklist for IRI_mean_y pavement prediction model", k=10)
final_summary_hits = rag_search(kind="final_pipeline_summary", query="Final pipeline summary for approved IRI_mean_y regression model", k=10)
if not final_report_hits: raise RuntimeError('Missing mm record kind="final_report".')
if not artifact_manifest_hits: raise RuntimeError('Missing mm record kind="artifact_manifest".')
if not deployment_readiness_hits: raise RuntimeError('Missing mm record kind="deployment_readiness".')
if not final_summary_hits: raise RuntimeError('Missing mm record kind="final_pipeline_summary".')
mm.rag_add(kind="pipeline_state", text="PIPELINE_STATE_COMPLETED=FINAL_REPORTING_ARCHIVE. Final pavement prediction pipeline report, summary, artifact manifest, and deployment readiness checklist completed.", meta={**base_meta, "target_column": TARGET_COLUMN, "task_type": TASK_TYPE, "best_model_name": best_model_name, "best_feature_set": best_feature_set, "artifacts_saved": final_artifacts_saved})
pipeline_state_hits_after = rag_search(kind="pipeline_state", query="PIPELINE_STATE_COMPLETED=FINAL_REPORTING_ARCHIVE", k=10)

# Final hard gate.
missing_gate_items = []
for p in required_final_artifacts:
    if not p.exists(): missing_gate_items.append(str(p))
if not final_report_hits: missing_gate_items.append('mm record kind="final_report"')
if not artifact_manifest_hits: missing_gate_items.append('mm record kind="artifact_manifest"')
if not deployment_readiness_hits: missing_gate_items.append('mm record kind="deployment_readiness"')
if not final_summary_hits: missing_gate_items.append('mm record kind="final_pipeline_summary"')
if not pipeline_state_hits_after: missing_gate_items.append("PIPELINE_STATE_COMPLETED=FINAL_REPORTING_ARCHIVE")
if final_summary_check.get("target_column") != TARGET_COLUMN: missing_gate_items.append("final_summary target_column")
if final_summary_check.get("task_type") != TASK_TYPE: missing_gate_items.append("final_summary task_type")
if final_summary_check.get("best_model_name") != "HistGradientBoostingRegressor": missing_gate_items.append("final_summary best_model_name")
if final_summary_check.get("best_feature_set") != "baseline_condition_allowed": missing_gate_items.append("final_summary best_feature_set")
if missing_gate_items:
    raise RuntimeError("FINAL_REPORTING_ARCHIVE hard gate failed. Missing/invalid: " + "; ".join(missing_gate_items))

gate_report = {
    "status": "FINAL_REPORTING_ARCHIVE_COMPLETED_VERIFIED_PENDING_REVIEWER_APPROVAL",
    "DATASET_ID": DATASET_ID,
    "FILE_PATH": FILE_PATH,
    "file_exists": os.path.exists(FILE_PATH),
    "file_path_outside_WORK_DIR": True,
    "PINNED_FILE_PATH_matches_registry": (True if not pinned else str(Path(pinned).resolve()) == str(file_path_resolved)),
    "state_already_completed_before_run": state_already_completed_before_run,
    "dataset_shape": [int(dataset_shape[0]), int(dataset_shape[1])],
    "dataset_columns_as_List_str": dataset_columns,
    "target_column": TARGET_COLUMN,
    "task_type": TASK_TYPE,
    "approved_states_verified": upstream_state_hits,
    "required_memory_retrieval_hit_counts": {"target_candidates_hits": len(target_candidates_memory_hits), "dataset_artifact_hits_for_COLUMNS_query": len(dataset_artifact_memory_hits), "model_metrics_hits": len(model_metrics_memory_hits), "model_metrics_per_model_hits": len(model_metrics_per_model_memory_hits)},
    "final_artifacts": {"final_pipeline_report_md": str(final_report_path), "final_pipeline_summary_json": str(final_summary_path), "artifact_manifest_json": str(artifact_manifest_path), "deployment_readiness_checklist_md": str(deployment_checklist_path), "final_code_py": str(final_code_path)},
    "artifact_manifest_summary": {"artifact_manifest_path": str(artifact_manifest_path), "artifact_count": int(len(manifest_check.get("artifacts", []))), "artifact_group_names": artifact_group_names, "organized_directories_confirmed": {"metrics": str(metrics_dir), "models": str(models_dir), "plots": str(plots_dir), "predictions": str(predictions_dir)}},
    "best_model_summary": final_summary["best_model"],
    "strict_no_prior_iri_best_model": final_summary["strict_no_prior_iri_best_model"],
    "iri_mean_x_sensitivity_summary": final_summary["iri_mean_x_sensitivity"],
    "deployment_cautions": deployment_cautions,
    "artifact_directories": final_summary["artifact_directories"],
    "mm_record_confirmation": {"final_report_hits": len(final_report_hits), "artifact_manifest_hits": len(artifact_manifest_hits), "deployment_readiness_hits": len(deployment_readiness_hits), "final_pipeline_summary_hits": len(final_summary_hits), "pipeline_state_hits": len(pipeline_state_hits_after)},
    "pipeline_state_completed_text": "PIPELINE_STATE_COMPLETED=FINAL_REPORTING_ARCHIVE",
}
print(json.dumps(json_clean(gate_report), indent=2))
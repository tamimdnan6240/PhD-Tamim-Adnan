from mm_runtime import mm
mm.health_check()

import os
import json
import hashlib
import pandas as pd
from datetime import datetime, timezone

WORK_DIR = "/content/pavement_agentic_workspace"
STATE_NOW = "FINAL_PACKAGE_CLOSURE"
TARGET = "IRI_mean_y"

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "final_package"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "models"), exist_ok=True)

# Registry is the ONLY source of dataset path / dataset_id
reg = mm.validate_registry(WORK_DIR)
FILE_PATH = reg["latest_path"]
DATASET_ID = reg["latest_dataset_id"]

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(FILE_PATH)

pinned = os.environ.get("PINNED_FILE_PATH")
if pinned and os.path.abspath(FILE_PATH) != os.path.abspath(pinned):
    raise RuntimeError(
        "Pinned dataset mismatch: "
        f"registry FILE_PATH={os.path.abspath(FILE_PATH)} != "
        f"PINNED_FILE_PATH={os.path.abspath(pinned)}"
    )

# Required memory-only retrievals for traceability/repeat-avoidance.
# Stale records are retrieved but not used as final evidence.
target_candidate_hits = mm.rag_search(
    kind="target_candidates",
    query="target_candidates.csv",
    k=3,
    meta_filters={"dataset_id": DATASET_ID}
)

dataset_column_hits = mm.rag_search(
    kind="dataset_artifact",
    query="COLUMNS",
    k=3,
    meta_filters={"dataset_id": DATASET_ID}
)

model_metrics_hits = mm.rag_search(
    kind="model_metrics",
    query="MODEL_METRICS",
    k=3,
    meta_filters={"dataset_id": DATASET_ID}
)

model_metrics_per_model_hits = mm.rag_search(
    kind="model_metrics_per_model",
    query="metrics",
    k=10,
    meta_filters={"dataset_id": DATASET_ID}
)

print({
    "memory_retrieval_counts": {
        "target_candidates": len(target_candidate_hits or []),
        "dataset_artifact_COLUMNS": len(dataset_column_hits or []),
        "model_metrics": len(model_metrics_hits or []),
        "model_metrics_per_model": len(model_metrics_per_model_hits or [])
    },
    "note": "Retrieved for traceability only. Final closure evidence is verified from approved root artifacts."
})

# DATA LOADING (MANDATORY): pandas read_csv from registry path.
df = pd.read_csv(FILE_PATH)
print("FILE_PATH:", FILE_PATH)
print("DATASET_ID:", DATASET_ID)
print("df.shape:", df.shape)
print("df.columns:", list(df.columns))
print("df.head(5):")
print(df.head(5))

# ARTIFACT CONSISTENCY CHECK before current dataset_artifact log.
art = mm.rag_get_latest(kind="dataset_artifact", meta_filters={"dataset_id": DATASET_ID})
if art and art.get("meta") and art["meta"].get("columns"):
    if art["meta"]["columns"] != df.columns.tolist():
        raise RuntimeError("Column mismatch vs prior artifact")

mm.rag_add(
    kind="dataset_artifact",
    text="Loaded dataset (see meta for path, shape, columns)",
    meta={
        "dataset_id": DATASET_ID,
        "file_path": FILE_PATH,
        "shape": list(df.shape),
        "ncols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "state": STATE_NOW
    }
)

# Repeat-avoidance check
state_hits = mm.rag_search(
    query=f"PIPELINE_STATE_COMPLETED={STATE_NOW}",
    kind="pipeline_state",
    k=5,
    meta_filters={"dataset_id": DATASET_ID}
)

# Approved final decision artifacts
final_model_selection_json_path = os.path.join(WORK_DIR, "model_selection", "final_model_selection.json")
final_model_card_json_path = os.path.join(WORK_DIR, "model_selection", "final_model_card.json")
final_model_card_md_path = os.path.join(WORK_DIR, "reports", "final_model_card.md")
executive_summary_md_path = os.path.join(WORK_DIR, "reports", "model_selection_executive_summary.md")
final_comparison_table_path = os.path.join(WORK_DIR, "metrics", "final_model_selection_comparison.csv")
final_xai_summary_table_path = os.path.join(WORK_DIR, "model_selection", "final_xai_summary_table.csv")
final_artifact_manifest_input_path = os.path.join(WORK_DIR, "model_selection", "final_artifact_manifest.json")
final_model_path = os.path.join(WORK_DIR, "models", "final_recommended_model.joblib")
final_model_metadata_path = os.path.join(WORK_DIR, "models", "final_recommended_model_metadata.json")

# Approved supporting artifacts
baseline_metrics_long_path = os.path.join(WORK_DIR, "metrics", "baseline_model_metrics_long.csv")
baseline_selected_path = os.path.join(WORK_DIR, "metrics", "selected_baseline_model.json")
advanced_metrics_long_path = os.path.join(WORK_DIR, "metrics", "advanced_model_metrics_long.csv")
advanced_selected_path = os.path.join(WORK_DIR, "metrics", "selected_advanced_model.json")
advanced_xai_csv_path = os.path.join(WORK_DIR, "xai", "advanced_selected_model_permutation_importance.csv")
advanced_xai_png_path = os.path.join(WORK_DIR, "xai", "advanced_selected_model_permutation_importance.png")
advanced_xai_summary_path = os.path.join(WORK_DIR, "xai", "advanced_xai_summary.json")

# Approved preprocessing/leakage artifacts
feature_policy_path = os.path.join(WORK_DIR, "features", "feature_policy_preprocess.json")
preprocessing_schema_path = os.path.join(WORK_DIR, "split_preprocess", "preprocessing_schema.json")
split_leakage_verification_path = os.path.join(WORK_DIR, "split_preprocess", "leakage_verification.json")
advanced_leakage_verification_path = os.path.join(
    WORK_DIR,
    "advanced_modeling",
    "reports",
    "advanced_modeling_leakage_verification.json"
)

# Closure outputs
final_package_dir = os.path.join(WORK_DIR, "final_package")
closure_manifest_path = os.path.join(final_package_dir, "final_package_manifest.json")
closure_report_json_path = os.path.join(final_package_dir, "final_closure_report.json")
closure_report_md_path = os.path.join(WORK_DIR, "reports", "final_closure_report.md")
closure_artifact_checksums_path = os.path.join(final_package_dir, "final_artifact_checksums.csv")
final_code_path = os.path.join(WORK_DIR, "final_code.py")

required_final_artifacts = [
    final_model_selection_json_path,
    final_model_card_json_path,
    final_model_card_md_path,
    executive_summary_md_path,
    final_comparison_table_path,
    final_xai_summary_table_path,
    final_artifact_manifest_input_path,
    final_model_path,
    final_model_metadata_path,
    baseline_metrics_long_path,
    baseline_selected_path,
    advanced_metrics_long_path,
    advanced_selected_path,
    advanced_xai_csv_path,
    advanced_xai_png_path,
    advanced_xai_summary_path,
    feature_policy_path,
    preprocessing_schema_path,
    split_leakage_verification_path,
    advanced_leakage_verification_path
]

closure_output_artifacts = [
    closure_manifest_path,
    closure_report_json_path,
    closure_report_md_path,
    closure_artifact_checksums_path
]

def verify_artifacts(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError("FINAL_PACKAGE_CLOSURE hard gate failed. Missing artifacts: " + "; ".join(missing))

def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_get_metric(metrics_dict, key):
    try:
        return metrics_dict.get(key)
    except Exception:
        return None

if state_hits and all(os.path.exists(p) for p in closure_output_artifacts):
    print(f"{STATE_NOW} already completed and closure artifacts exist. No rerun performed.")
    verify_artifacts(closure_output_artifacts)

    prior_closure = load_json(closure_report_json_path)

    print({
        "state": STATE_NOW,
        "dataset_id": DATASET_ID,
        "file_path": FILE_PATH,
        "target": prior_closure.get("target"),
        "final_recommended_model": prior_closure.get("final_recommended_model_name"),
        "closure_artifacts": closure_output_artifacts,
        "rerun_performed": False,
        "hard_gates_verified": True
    })

else:
    verify_artifacts(required_final_artifacts)

    # Load approved final artifacts
    final_selection = load_json(final_model_selection_json_path)
    final_model_card = load_json(final_model_card_json_path)
    final_model_metadata = load_json(final_model_metadata_path)
    feature_policy = load_json(feature_policy_path)
    preprocessing_schema = load_json(preprocessing_schema_path)
    split_leakage = load_json(split_leakage_verification_path)
    advanced_leakage = load_json(advanced_leakage_verification_path)

    if final_selection.get("target") != TARGET:
        raise RuntimeError("final_model_selection.json target mismatch.")

    if final_model_card.get("target") != TARGET:
        raise RuntimeError("final_model_card.json target mismatch.")

    if final_model_metadata.get("target") != TARGET:
        raise RuntimeError("final_recommended_model_metadata.json target mismatch.")

    if feature_policy.get("target") != TARGET:
        raise RuntimeError("feature_policy_preprocess.json target mismatch.")

    if preprocessing_schema.get("target") != TARGET:
        raise RuntimeError("preprocessing_schema.json target mismatch.")

    if not split_leakage.get("leakage_controls_passed", False):
        raise RuntimeError("split_preprocess leakage verification does not pass.")

    if not advanced_leakage.get("leakage_controls_passed", False):
        raise RuntimeError("advanced modeling leakage verification does not pass.")

    final_recommended_model_name = final_selection.get("recommended_model_name")
    final_recommended_model_stage = final_selection.get("recommended_model_stage")

    if final_recommended_model_name != "HistGradientBoostingRegressor_baseline":
        raise RuntimeError(
            f"Unexpected final recommended model: {final_recommended_model_name}. "
            "Expected HistGradientBoostingRegressor_baseline."
        )

    if final_recommended_model_stage != "BASELINE_MODELING":
        raise RuntimeError(
            f"Unexpected final recommended model stage: {final_recommended_model_stage}. "
            "Expected BASELINE_MODELING."
        )

    if not os.path.exists(final_model_path):
        raise FileNotFoundError(final_model_path)

    # Verify metrics schemas
    baseline_metrics_df = pd.read_csv(baseline_metrics_long_path)
    advanced_metrics_df = pd.read_csv(advanced_metrics_long_path)
    final_comparison_df = pd.read_csv(final_comparison_table_path)
    final_xai_df = pd.read_csv(final_xai_summary_table_path)

    required_long_schema = ["model_name", "split", "metric_name", "value", "timestamp"]
    if list(baseline_metrics_df.columns) != required_long_schema:
        raise RuntimeError("baseline_model_metrics_long.csv schema mismatch.")

    if list(advanced_metrics_df.columns) != required_long_schema:
        raise RuntimeError("advanced_model_metrics_long.csv schema mismatch.")

    if len(final_comparison_df) <= 0:
        raise RuntimeError("Final model selection comparison table is empty.")

    if "is_final_recommended_model" not in final_comparison_df.columns:
        raise RuntimeError("Final model comparison table missing is_final_recommended_model.")

    rec_col = final_comparison_df["is_final_recommended_model"]
    if rec_col.dtype == bool:
        rec_count = int(rec_col.sum())
    else:
        rec_count = int(rec_col.astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    if rec_count != 1:
        raise RuntimeError("Final comparison table must contain exactly one recommended model.")

    required_final_xai_cols = [
        "feature_name",
        "source_column_inferred",
        "engineering_family_if_available",
        "importance_mean",
        "importance_std",
        "rank",
        "timestamp",
        "interpretation_note"
    ]

    if list(final_xai_df.columns) != required_final_xai_cols:
        raise RuntimeError("final_xai_summary_table.csv schema mismatch.")

    if len(final_xai_df) <= 0:
        raise RuntimeError("final_xai_summary_table.csv is empty.")

    # Verify leakage exclusions in feature policy / final predictors
    final_predictors = feature_policy.get("final_predictors", [])
    forbidden_predictors = ["IRI_mean_y", "Treatment_type", "Future_AADT_mean_x"]

    leakage_issues = []

    for col in forbidden_predictors:
        if col in final_predictors:
            leakage_issues.append(f"Forbidden predictor present in final_predictors: {col}")

    non_target_y_predictors = [
        c for c in final_predictors
        if str(c).endswith("_y") and c != TARGET
    ]

    if non_target_y_predictors:
        leakage_issues.append(f"Non-target _y predictors present: {non_target_y_predictors}")

    if leakage_issues:
        raise RuntimeError("Final closure leakage verification failed: " + "; ".join(leakage_issues))

    # Checksums for final package artifacts
    checksum_rows = []
    for p in required_final_artifacts:
        checksum_rows.append({
            "artifact_path": p,
            "exists": bool(os.path.exists(p)),
            "bytes": int(os.path.getsize(p)) if os.path.exists(p) else None,
            "sha256": sha256_file(p) if os.path.exists(p) and os.path.isfile(p) else None
        })

    checksum_df = pd.DataFrame(checksum_rows)
    checksum_df.to_csv(closure_artifact_checksums_path, index=False)

    timestamp = datetime.now(timezone.utc).isoformat()

    recommended_metrics = final_selection.get("recommended_model_test_metrics", {})
    baseline_selected = final_selection.get("baseline_selected_model", {})
    advanced_selected = final_selection.get("advanced_selected_model", {})
    tradeoff = final_selection.get("baseline_vs_advanced_tradeoff", {})

    closure_manifest = {
        "dataset_id": DATASET_ID,
        "file_path": FILE_PATH,
        "state": STATE_NOW,
        "created_at_utc": timestamp,
        "target": TARGET,
        "final_recommended_model_name": final_recommended_model_name,
        "final_recommended_model_stage": final_recommended_model_stage,
        "final_recommended_model_artifact": final_model_path,
        "final_model_metadata": final_model_metadata_path,
        "approved_final_decision_artifacts": [
            final_model_selection_json_path,
            final_model_card_json_path,
            final_model_card_md_path,
            executive_summary_md_path,
            final_comparison_table_path,
            final_xai_summary_table_path,
            final_artifact_manifest_input_path
        ],
        "approved_metric_artifacts": [
            baseline_metrics_long_path,
            advanced_metrics_long_path,
            final_comparison_table_path
        ],
        "approved_xai_artifacts": [
            advanced_xai_csv_path,
            advanced_xai_png_path,
            advanced_xai_summary_path,
            final_xai_summary_table_path
        ],
        "approved_model_artifacts": [
            final_model_path,
            final_model_metadata_path
        ],
        "approved_governance_artifacts": [
            feature_policy_path,
            preprocessing_schema_path,
            split_leakage_verification_path,
            advanced_leakage_verification_path
        ],
        "artifact_checksums": closure_artifact_checksums_path,
        "closure_report_json": closure_report_json_path,
        "closure_report_md": closure_report_md_path,
        "leakage_controls": {
            "IRI_mean_y": "Target only; excluded from predictors.",
            "Treatment_type": "Excluded unless temporal provenance is reviewer-approved.",
            "Future_AADT_mean_x": "Excluded unless proven baseline-available.",
            "non_target_y_variables": "Excluded generically.",
            "IRI_mean_x": "Conditionally valid only under _x baseline-before-_y assumption.",
            "leakage_controls_passed": True
        },
        "deployment_use_restrictions": [
            "Use for network-level pavement roughness forecasting and decision-support screening only.",
            "Do not use as a standalone final treatment-programming system without agency validation.",
            "Confirm IRI units before applying condition thresholds.",
            "Confirm _x variables are temporally prior to IRI_mean_y.",
            "Review duplicate and asset-correlation risks.",
            "Use grouped or temporal validation if route/segment/year identifiers become available.",
            "Do not reintroduce Treatment_type or Future_AADT_mean_x without reviewer-approved provenance.",
            "Use approved root artifacts only; do not use stale MM metrics as evidence."
        ],
        "prohibited_actions_confirmed": [
            "No model training performed.",
            "No hyperparameter tuning performed.",
            "No preprocessing refit performed.",
            "No prediction changes performed.",
            "No metric changes performed.",
            "No model selection changes performed."
        ],
        "all_verified_artifacts": required_final_artifacts + closure_output_artifacts
    }

    with open(closure_manifest_path, "w", encoding="utf-8") as f:
        json.dump(closure_manifest, f, indent=2)

    closure_report = {
        "dataset_id": DATASET_ID,
        "file_path": FILE_PATH,
        "state": STATE_NOW,
        "created_at_utc": timestamp,
        "target": TARGET,
        "target_interpretation": (
            "Future/outcome pavement roughness represented by International Roughness Index. "
            "Higher IRI means rougher/worse pavement condition."
        ),
        "final_recommended_model_name": final_recommended_model_name,
        "final_recommended_model_stage": final_recommended_model_stage,
        "final_model_artifact": final_model_path,
        "final_metrics": {
            "rmse": safe_get_metric(recommended_metrics, "rmse"),
            "mae": safe_get_metric(recommended_metrics, "mae"),
            "r2": safe_get_metric(recommended_metrics, "r2"),
            "smape_pct": safe_get_metric(recommended_metrics, "smape_pct"),
            "median_absolute_error": safe_get_metric(recommended_metrics, "median_absolute_error"),
            "max_absolute_error": safe_get_metric(recommended_metrics, "max_absolute_error")
        },
        "selection_rationale": final_selection.get("selection_summary"),
        "baseline_vs_advanced_tradeoff": tradeoff,
        "baseline_selected_model": baseline_selected,
        "advanced_selected_model": advanced_selected,
        "xai_summary": {
            "final_xai_summary_table": final_xai_summary_table_path,
            "xai_row_count": int(len(final_xai_df)),
            "top_10_features": final_xai_df.head(10).to_dict(orient="records"),
            "interpretation_caution": "Permutation importance is a model-behavior diagnostic, not causal proof."
        },
        "engineering_feature_families": {
            "baseline_condition": ["IRI_mean_x", "Rutting_mean_x", "Cracking_Percent_mean_x", "Faulting_mean_x"],
            "traffic_loading": ["AADT_mean_x", "AADT_Single_Unit_mean_x", "AADT_Combination_mean_x"],
            "structure_materials": [
                "Thickness_Rigid_mean_x",
                "Thickness_Flexible_mean_x",
                "Base_Thickness_mean_x",
                "Last_Overlay_Thickness_mean_x",
                "Surface_Type_mode",
                "Base_Type_mode_x"
            ],
            "climate_environment": ["RHU_AV_x", "FRZ_IDX_x", "TEMP_AVG_x", "PRECIPITATION_x"],
            "functional_context": ["F_System_mode", "Urban_Type"],
            "treatment_history_retained": ["Age_x"],
            "excluded_treatment_future_variables": ["Treatment_type", "Future_AADT_mean_x"]
        },
        "leakage_controls": closure_manifest["leakage_controls"],
        "limitations": [
            "_x variables are assumed baseline/prior to _y; explicit dates were not available.",
            "IRI_mean_x is conditionally valid only under the _x baseline-before-_y assumption.",
            "Treatment_type was excluded to avoid treatment/intervention leakage.",
            "Future_AADT_mean_x was excluded to avoid future-information leakage.",
            "Random holdout split may not address spatial, corridor, or asset-level correlation.",
            "Prior EDA found exact duplicate rows.",
            "No route/segment/year/group identifier was available for grouped or temporal validation.",
            "IRI units should be confirmed before operational use of good/fair/poor thresholds.",
            "Maximum held-out error indicates some sections may have large prediction errors.",
            "Model is suitable for network-level screening, not standalone treatment programming without validation."
        ],
        "deployment_use_restrictions": closure_manifest["deployment_use_restrictions"],
        "artifact_manifest": closure_manifest_path,
        "artifact_checksums": closure_artifact_checksums_path,
        "prohibited_actions_confirmed": closure_manifest["prohibited_actions_confirmed"]
    }

    with open(closure_report_json_path, "w", encoding="utf-8") as f:
        json.dump(closure_report, f, indent=2)

    with open(closure_report_md_path, "w", encoding="utf-8") as f:
        f.write("# Final Pavement IRI Prediction Package Closure Report\n\n")
        f.write(f"Dataset ID: `{DATASET_ID}`\n\n")
        f.write(f"Dataset path: `{FILE_PATH}`\n\n")
        f.write(f"Target: `{TARGET}`\n\n")
        f.write("Target meaning: future/outcome pavement roughness. Higher IRI means rougher/worse condition.\n\n")
        f.write("## Final recommended model\n\n")
        f.write(f"`{final_recommended_model_name}` from `{final_recommended_model_stage}`\n\n")
        f.write(f"Model artifact: `{final_model_path}`\n\n")
        f.write("## Final held-out test metrics\n\n")
        f.write(f"- RMSE: {closure_report['final_metrics']['rmse']}\n")
        f.write(f"- MAE: {closure_report['final_metrics']['mae']}\n")
        f.write(f"- R²: {closure_report['final_metrics']['r2']}\n")
        f.write(f"- SMAPE (%): {closure_report['final_metrics']['smape_pct']}\n")
        f.write(f"- Median absolute error: {closure_report['final_metrics']['median_absolute_error']}\n")
        f.write(f"- Max absolute error: {closure_report['final_metrics']['max_absolute_error']}\n\n")
        f.write("## Selection rationale\n\n")
        f.write(str(final_selection.get("selection_summary")) + "\n\n")
        f.write("Advanced tuning produced negligible RMSE/R² gain and slightly worse MAE, so the simpler approved baseline model is recommended for reporting/deployment-readiness.\n\n")
        f.write("## Leakage controls\n\n")
        f.write("- `IRI_mean_y` is target-only and excluded from predictors.\n")
        f.write("- `Treatment_type` remains excluded unless temporal provenance is reviewer-approved.\n")
        f.write("- `Future_AADT_mean_x` remains excluded unless proven baseline-available.\n")
        f.write("- Non-target `_y` variables remain excluded.\n")
        f.write("- `IRI_mean_x` remains conditionally valid only under the `_x` baseline-before-`_y` assumption.\n\n")
        f.write("## XAI / interpretability\n\n")
        f.write(f"Final XAI summary table: `{final_xai_summary_table_path}`\n\n")
        f.write("Permutation importance is a model-behavior diagnostic, not causal proof of pavement deterioration mechanisms.\n\n")
        f.write("## Operational use restrictions\n\n")
        for item in closure_manifest["deployment_use_restrictions"]:
            f.write(f"- {item}\n")
        f.write("\n## Final package artifacts\n\n")
        f.write(f"- Closure manifest: `{closure_manifest_path}`\n")
        f.write(f"- Artifact checksums: `{closure_artifact_checksums_path}`\n")

    verify_artifacts(closure_output_artifacts)

    # Verify closure report integrity
    closure_check = load_json(closure_report_json_path)
    if closure_check.get("target") != TARGET:
        raise RuntimeError("final_closure_report.json target mismatch.")

    if closure_check.get("final_recommended_model_name") != "HistGradientBoostingRegressor_baseline":
        raise RuntimeError("final_closure_report.json final model mismatch.")

    manifest_check = load_json(closure_manifest_path)
    if manifest_check.get("final_recommended_model_artifact") != final_model_path:
        raise RuntimeError("final_package_manifest.json final model artifact mismatch.")

    checksum_check = pd.read_csv(closure_artifact_checksums_path)
    if len(checksum_check) != len(required_final_artifacts):
        raise RuntimeError("final_artifact_checksums.csv row count mismatch.")

    if not checksum_check["exists"].all():
        raise RuntimeError("final_artifact_checksums.csv indicates missing artifacts.")

    # MM lightweight records
    mm.rag_add(
        kind="final_package_closure",
        text="FINAL_PACKAGE_CLOSURE completed for target IRI_mean_y final pavement prediction package verified",
        meta={
            "dataset_id": DATASET_ID,
            "file_path": FILE_PATH,
            "state": STATE_NOW,
            "target": TARGET,
            "final_recommended_model_name": final_recommended_model_name,
            "final_recommended_model_stage": final_recommended_model_stage,
            "final_model_artifact": final_model_path,
            "closure_manifest_path": closure_manifest_path,
            "closure_report_json_path": closure_report_json_path,
            "closure_report_md_path": closure_report_md_path,
            "artifact_checksums_path": closure_artifact_checksums_path,
            "artifacts_saved": closure_output_artifacts,
            "final_rmse": closure_report["final_metrics"]["rmse"],
            "final_mae": closure_report["final_metrics"]["mae"],
            "final_r2": closure_report["final_metrics"]["r2"]
        }
    )

    mm.rag_add(
        kind="pipeline_state",
        text=f"PIPELINE_STATE_COMPLETED={STATE_NOW}",
        meta={
            "dataset_id": DATASET_ID,
            "file_path": FILE_PATH,
            "state": STATE_NOW,
            "target": TARGET,
            "final_recommended_model_name": final_recommended_model_name,
            "final_recommended_model_stage": final_recommended_model_stage,
            "artifacts_saved": closure_output_artifacts,
            "completed_at_utc": datetime.now(timezone.utc).isoformat()
        }
    )

    closure_hits = mm.rag_search(
        kind="final_package_closure",
        query="FINAL_PACKAGE_CLOSURE completed for target IRI_mean_y",
        k=5,
        meta_filters={"dataset_id": DATASET_ID}
    )

    pipeline_state_hits = mm.rag_search(
        kind="pipeline_state",
        query=f"PIPELINE_STATE_COMPLETED={STATE_NOW}",
        k=5,
        meta_filters={"dataset_id": DATASET_ID}
    )

    missing_or_errors = []

    if not closure_hits:
        missing_or_errors.append('mm record kind="final_package_closure" with completion text')

    if not pipeline_state_hits:
        missing_or_errors.append(f'mm record kind="pipeline_state" with PIPELINE_STATE_COMPLETED={STATE_NOW}')

    if missing_or_errors:
        raise RuntimeError("FINAL_PACKAGE_CLOSURE hard gate failed: " + "; ".join(missing_or_errors))

    print({
        "state": STATE_NOW,
        "dataset_id": DATASET_ID,
        "file_path": FILE_PATH,
        "target": TARGET,
        "final_recommended_model_name": final_recommended_model_name,
        "final_recommended_model_stage": final_recommended_model_stage,
        "final_model_artifact": final_model_path,
        "final_metrics": closure_report["final_metrics"],
        "closure_manifest": closure_manifest_path,
        "closure_report_json": closure_report_json_path,
        "closure_report_md": closure_report_md_path,
        "artifact_checksums": closure_artifact_checksums_path,
        "verified_final_artifact_count": int(len(required_final_artifacts)),
        "xai_summary_rows": int(len(final_xai_df)),
        "leakage_controls_passed": True,
        "IRI_mean_x_policy": "Conditionally valid under documented _x baseline-before-_y assumption.",
        "deployment_use_restrictions": closure_manifest["deployment_use_restrictions"],
        "prohibited_actions_confirmed": closure_manifest["prohibited_actions_confirmed"],
        "mm_audit_counts": {
            "final_package_closure": len(closure_hits or []),
            "pipeline_state_completed": len(pipeline_state_hits or [])
        },
        "hard_gates_verified": True
    })

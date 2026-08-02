# Final Pavement IRI Prediction Package Closure Report

Dataset ID: `9bc07dd2ac12`

Dataset path: `/content/drive/MyDrive/Agentic-AI-Asset-management/Tamim/codes/Data/Maintenance-data/2017-2019-maintenance-dataset.csv`

Target: `IRI_mean_y`

Target meaning: future/outcome pavement roughness. Higher IRI means rougher/worse condition.

## Final recommended model

`HistGradientBoostingRegressor_baseline` from `BASELINE_MODELING`

Model artifact: `/content/pavement_agentic_workspace/models/final_recommended_model.joblib`

## Final held-out test metrics

- RMSE: 22.22428583482637
- MAE: 14.352903541317088
- R²: 0.7886720224660457
- SMAPE (%): 15.927775716091189
- Median absolute error: 8.297007740607768
- Max absolute error: 116.6045513055996

## Selection rationale

Baseline selected model recommended because advanced tuning produced only negligible RMSE/R² improvement while MAE was slightly worse. For pavement management, this does not justify added advanced-model tuning complexity.

Advanced tuning produced negligible RMSE/R² gain and slightly worse MAE, so the simpler approved baseline model is recommended for reporting/deployment-readiness.

## Leakage controls

- `IRI_mean_y` is target-only and excluded from predictors.
- `Treatment_type` remains excluded unless temporal provenance is reviewer-approved.
- `Future_AADT_mean_x` remains excluded unless proven baseline-available.
- Non-target `_y` variables remain excluded.
- `IRI_mean_x` remains conditionally valid only under the `_x` baseline-before-`_y` assumption.

## XAI / interpretability

Final XAI summary table: `/content/pavement_agentic_workspace/model_selection/final_xai_summary_table.csv`

Permutation importance is a model-behavior diagnostic, not causal proof of pavement deterioration mechanisms.

## Operational use restrictions

- Use for network-level pavement roughness forecasting and decision-support screening only.
- Do not use as a standalone final treatment-programming system without agency validation.
- Confirm IRI units before applying condition thresholds.
- Confirm _x variables are temporally prior to IRI_mean_y.
- Review duplicate and asset-correlation risks.
- Use grouped or temporal validation if route/segment/year identifiers become available.
- Do not reintroduce Treatment_type or Future_AADT_mean_x without reviewer-approved provenance.
- Use approved root artifacts only; do not use stale MM metrics as evidence.

## Final package artifacts

- Closure manifest: `/content/pavement_agentic_workspace/final_package/final_package_manifest.json`
- Artifact checksums: `/content/pavement_agentic_workspace/final_package/final_artifact_checksums.csv`

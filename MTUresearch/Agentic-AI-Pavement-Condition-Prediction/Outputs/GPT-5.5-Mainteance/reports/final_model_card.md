# Pavement IRI Prediction Model Card

Dataset ID: `9bc07dd2ac12`

Target: `IRI_mean_y`

Recommended model: `HistGradientBoostingRegressor_baseline`

Recommended stage: `BASELINE_MODELING`

## Intended use

Network-level pavement roughness forecasting and decision-support screening. Not a standalone final treatment programming tool without agency validation.

## Performance summary

- RMSE: 22.22428583482637
- MAE: 14.352903541317088
- R²: 0.7886720224660457
- SMAPE (%): 15.927775716091189

## Selection rationale

Baseline selected model recommended because advanced tuning produced only negligible RMSE/R² improvement while MAE was slightly worse. For pavement management, this does not justify added advanced-model tuning complexity.

## Leakage controls

- `IRI_mean_y` is target only and excluded from predictors.
- `Treatment_type` excluded.
- `Future_AADT_mean_x` excluded.
- Non-target `_y` variables excluded.
- `IRI_mean_x` conditionally valid only under `_x` baseline-before-`_y` assumption.

## XAI summary

XAI summary table: `/content/pavement_agentic_workspace/model_selection/final_xai_summary_table.csv`

Permutation importance is a model-behavior diagnostic, not causal proof.

## Limitations

- _x variables are assumed baseline/prior to _y; this is not proven by explicit date fields.
- IRI_mean_x is conditionally valid only under the _x baseline-before-_y assumption.
- Treatment_type was excluded to avoid treatment/intervention leakage.
- Future_AADT_mean_x was excluded to avoid future-information leakage.
- Random holdout split may not address spatial, corridor, or asset-level correlation.
- Prior EDA found exact duplicate rows.
- No route/segment/year/group identifier was available for grouped or temporal validation.
- IRI units should be confirmed before operational use of good/fair/poor thresholds.
- Permutation importance describes model sensitivity, not causal pavement mechanisms.

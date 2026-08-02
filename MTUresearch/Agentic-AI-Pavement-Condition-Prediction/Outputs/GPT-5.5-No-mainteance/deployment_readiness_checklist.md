# Deployment Readiness Checklist

Pipeline target: `IRI_mean_y`  
Task: `regression`  
Best approved model: `HistGradientBoostingRegressor`  
Best approved feature set: `baseline_condition_allowed`

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

- RMSE: `21.9044`
- MAE: `14.5319`
- R²: `0.7957`

Strict no-prior-IRI benchmark:

- RMSE: `34.1715`
- MAE: `23.5992`
- R²: `0.5029`

Primary caution:

> The best model depends materially on `IRI_mean_x`. This is pavement-engineering plausible, but operational use requires proving `IRI_mean_x` is true baseline/pre-outcome IRI available at prediction time.

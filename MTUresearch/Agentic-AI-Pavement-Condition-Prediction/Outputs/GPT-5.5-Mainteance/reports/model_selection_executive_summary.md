# Model Selection Executive Summary

The approved target is `IRI_mean_y`, interpreted as future/outcome pavement roughness.

Recommended model: `HistGradientBoostingRegressor_baseline` from `BASELINE_MODELING`.

## Baseline vs advanced tradeoff

- Selected baseline RMSE: 22.22428583482637
- Selected baseline MAE: 14.352903541317088
- Selected baseline R²: 0.7886720224660457
- Selected advanced RMSE: 22.214036375043023
- Selected advanced MAE: 14.560707530835806
- Selected advanced R²: 0.7888668992024623

RMSE improvement of advanced vs baseline: 0.010249459783349124

MAE improvement of advanced vs baseline: -0.2078039895187178

## Recommendation

Baseline selected model recommended because advanced tuning produced only negligible RMSE/R² improvement while MAE was slightly worse. For pavement management, this does not justify added advanced-model tuning complexity.

The recommendation emphasizes pavement-management usefulness in IRI units, not only marginal statistical differences.

# MODELING_PIPELINE_DESIGN Report

## Target and Task
- Target: `IRI_mean_y`
- Task type: `regression`
- Dataset ID: `9bc07dd2ac12`
- Dataset shape used after dropping missing target rows: `5801` rows x `24` columns

## Leakage Controls
- `IRI_mean_y` was excluded from every feature matrix.
- `baseline_condition_allowed` includes `IRI_mean_x`.
- `strict_no_prior_iri` excludes `IRI_mean_x`.
- Final excluded columns:
```json
[
  "IRI_mean_y"
]
```

## Train/Test and CV Strategy
- Holdout test size: `0.2`
- Random seed: `42`
- CV: `5`-fold shuffled KFold on training data only
- Stratification used: `True`
- Stratification notes: `Target quantile-bin stratification used for train/test split.`

All preprocessing was implemented inside sklearn pipelines. Imputers and encoders were fit only inside train/CV folds during model selection and on the training split before final holdout evaluation.

## Models Optimized
```json
[
  "RandomForestRegressor",
  "ExtraTreesRegressor",
  "HistGradientBoostingRegressor"
]
```

Hyperparameter optimization used `RandomizedSearchCV` with `10` sampled configurations per model per feature set, optimizing negative RMSE.

## Model Performance
dataset_id,feature_set,model_name,cv_rmse_mean,cv_rmse_std,cv_mae_mean,cv_r2_mean,test_rmse,test_mae,test_r2,test_median_absolute_error,test_explained_variance,test_smape,best_params,model_artifact_path,prediction_artifact_path
9bc07dd2ac12,baseline_condition_allowed,HistGradientBoostingRegressor,23.050435214241922,0.9177649713156119,14.666956303363767,0.7852142244922531,21.904398318987035,14.531929524204388,0.7957420130339768,8.900613844655837,0.796140995434934,16.090872763505008,"{""model__min_samples_leaf"": 20, ""model__max_leaf_nodes"": 31, ""model__max_iter"": 300, ""model__learning_rate"": 0.03, ""model__l2_regularization"": 0.0}",/content/pavement_agentic_workspace/models/model__baseline_condition_allowed__HistGradientBoostingRegressor.joblib,/content/pavement_agentic_workspace/predictions/predictions__baseline_condition_allowed__HistGradientBoostingRegressor.csv
9bc07dd2ac12,baseline_condition_allowed,ExtraTreesRegressor,22.777702869606117,1.1143709694342252,14.423801567978234,0.7905184397957392,22.466275624158012,14.479722848787993,0.7851286283153007,8.181416666666607,0.7852079078560212,15.596839287136632,"{""model__n_estimators"": 200, ""model__min_samples_split"": 2, ""model__min_samples_leaf"": 2, ""model__max_features"": 0.6, ""model__max_depth"": 25}",/content/pavement_agentic_workspace/models/model__baseline_condition_allowed__ExtraTreesRegressor.joblib,/content/pavement_agentic_workspace/predictions/predictions__baseline_condition_allowed__ExtraTreesRegressor.csv
9bc07dd2ac12,baseline_condition_allowed,RandomForestRegressor,22.80006053858534,0.9134090266875552,14.840793117887042,0.7898875369329437,22.504808915540497,14.763631090128685,0.7843909179955373,8.639890476190466,0.7844438231565933,15.975567155475465,"{""model__n_estimators"": 200, ""model__min_samples_split"": 2, ""model__min_samples_leaf"": 2, ""model__max_features"": 0.6, ""model__max_depth"": 25}",/content/pavement_agentic_workspace/models/model__baseline_condition_allowed__RandomForestRegressor.joblib,/content/pavement_agentic_workspace/predictions/predictions__baseline_condition_allowed__RandomForestRegressor.csv
9bc07dd2ac12,strict_no_prior_iri,HistGradientBoostingRegressor,34.85040718735278,1.7692129704674524,23.766854865889858,0.5097830854515203,34.1714731679602,23.599168816190765,0.5028999633779379,15.983614358781509,0.5038649646230766,25.66062298821986,"{""model__min_samples_leaf"": 20, ""model__max_leaf_nodes"": 31, ""model__max_iter"": 300, ""model__learning_rate"": 0.03, ""model__l2_regularization"": 0.0}",/content/pavement_agentic_workspace/models/model__strict_no_prior_iri__HistGradientBoostingRegressor.joblib,/content/pavement_agentic_workspace/predictions/predictions__strict_no_prior_iri__HistGradientBoostingRegressor.csv
9bc07dd2ac12,strict_no_prior_iri,RandomForestRegressor,34.71724215091341,1.830259497096311,23.87582282871736,0.513664072193179,34.577843140794535,23.826481800905288,0.4910065535436996,16.521817886155688,0.49250504975460674,25.979032471617426,"{""model__n_estimators"": 200, ""model__min_samples_split"": 5, ""model__min_samples_leaf"": 2, ""model__max_features"": ""sqrt"", ""model__max_depth"": 12}",/content/pavement_agentic_workspace/models/model__strict_no_prior_iri__RandomForestRegressor.joblib,/content/pavement_agentic_workspace/predictions/predictions__strict_no_prior_iri__RandomForestRegressor.csv
9bc07dd2ac12,strict_no_prior_iri,ExtraTreesRegressor,34.916195052869895,2.0221455874649004,23.32146728020029,0.5081370581781514,34.64909485765422,23.187918210799072,0.48890671043661094,15.242583333333343,0.4897949298036922,24.618336179760057,"{""model__n_estimators"": 200, ""model__min_samples_split"": 2, ""model__min_samples_leaf"": 2, ""model__max_features"": 0.6, ""model__max_depth"": 25}",/content/pavement_agentic_workspace/models/model__strict_no_prior_iri__ExtraTreesRegressor.joblib,/content/pavement_agentic_workspace/predictions/predictions__strict_no_prior_iri__ExtraTreesRegressor.csv


## Best Model
```json
{
  "feature_set": "baseline_condition_allowed",
  "model_name": "HistGradientBoostingRegressor",
  "test_rmse": 21.904398318987035,
  "test_mae": 14.531929524204388,
  "test_r2": 0.7957420130339768,
  "model_artifact_path": "/content/pavement_agentic_workspace/models/best_model.joblib",
  "source_model_artifact_path": "/content/pavement_agentic_workspace/models/model__baseline_condition_allowed__HistGradientBoostingRegressor.joblib",
  "prediction_artifact_path": "/content/pavement_agentic_workspace/predictions/predictions__baseline_condition_allowed__HistGradientBoostingRegressor.csv"
}
```

## IRI_mean_x Sensitivity
```json
{
  "baseline_condition_allowed_best_model": "HistGradientBoostingRegressor",
  "strict_no_prior_iri_best_model": "HistGradientBoostingRegressor",
  "baseline_condition_allowed_test_rmse": 21.904398318987035,
  "strict_no_prior_iri_test_rmse": 34.1714731679602,
  "baseline_condition_allowed_test_mae": 14.531929524204388,
  "strict_no_prior_iri_test_mae": 23.599168816190765,
  "baseline_condition_allowed_test_r2": 0.7957420130339768,
  "strict_no_prior_iri_test_r2": 0.5028999633779379,
  "delta_RMSE_strict_minus_baseline": 12.267074848973166,
  "delta_MAE_strict_minus_baseline": 9.067239291986377,
  "delta_R2_strict_minus_baseline": -0.292842049656039,
  "interpretation": "A performance drop when IRI_mean_x is removed does not automatically prove leakage, but indicates dependence on prior IRI and requires deployment-time availability confirmation."
}
```

A performance drop when `IRI_mean_x` is removed does not automatically prove leakage, but it indicates reliance on prior IRI and requires confirmation that baseline IRI is available at prediction time.

## Best Model Feature Importance
feature_set,model_name,feature,importance,importance_std,importance_type
baseline_condition_allowed,HistGradientBoostingRegressor,IRI_mean_x,33.97156792148756,0.8517633085343291,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,AADT_Combination_mean_x,1.6176486382239055,0.29990661144892905,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,Cracking_Percent_mean_x,0.7555650597479427,0.21130793853329288,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,Future_AADT_mean_x,0.5604114704637148,0.26917656394594786,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,FRZ_IDX_x,0.5508410254403572,0.08751473032002928,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,Base_Thickness_mean_x,0.5431942718346541,0.21340871913330814,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,Urban_Type,0.47146701797281454,0.08105337857883542,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,PRECIPITATION_x,0.42711633381379244,0.14378794343506662,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,TEMP_AVG_x,0.41665984832988057,0.15158783106941315,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,Rutting_mean_x,0.33114995486711435,0.07997442067208177,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,F_System_mode,0.2841301189302456,0.06977566996381337,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,RHU_AV_x,0.2820301512913829,0.031384103525384496,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,Age_x,0.23559208270199933,0.22118546836549727,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,AADT_mean_x,0.23412255897079604,0.11471475594440012,permutation_importance_neg_rmse_on_test
baseline_condition_allowed,HistGradientBoostingRegressor,Last_Overlay_Thickness_mean_x,0.1932697645865673,0.05417496174100952,permutation_importance_neg_rmse_on_test


## Saved Artifacts
- Modeling plan: `/content/pavement_agentic_workspace/modeling_plan.json`
- Modeling feature sets: `/content/pavement_agentic_workspace/modeling_feature_sets.json`
- Train/test split metadata: `/content/pavement_agentic_workspace/train_test_split_metadata.json`
- Model performance metrics: `/content/pavement_agentic_workspace/model_performance_metrics.csv`
- Model metrics JSON: `/content/pavement_agentic_workspace/model_metrics.json`
- Best model: `/content/pavement_agentic_workspace/models/best_model.joblib`
- Best model metadata: `/content/pavement_agentic_workspace/models/best_model_metadata.json`
- Predictions directory: `/content/pavement_agentic_workspace/predictions`
- Models directory: `/content/pavement_agentic_workspace/models`
- Plots directory: `/content/pavement_agentic_workspace/plots`


---

## Reviewer Repair: Organized Metrics/XAI Artifacts

Repair timestamp UTC: `2026-07-28T13:06:37.883401+00:00`

No retraining was performed. Existing target, feature sets, split, predictions, trained model artifacts, and plots were preserved.

Organized artifact directories:

- Metrics/XAI: `/content/pavement_agentic_workspace/metrics`
- Models: `/content/pavement_agentic_workspace/models`
- Plots: `/content/pavement_agentic_workspace/plots`
- Predictions: `/content/pavement_agentic_workspace/predictions`

Metrics artifacts now include:

- `/content/pavement_agentic_workspace/metrics/model_performance_metrics.csv`
- `/content/pavement_agentic_workspace/metrics/model_metrics.json`
- `/content/pavement_agentic_workspace/metrics/model_metrics_long.csv`
- `/content/pavement_agentic_workspace/metrics/feature_importance_best_models.csv`

`model_metrics_long.csv` audit schema:

```text
dataset_id,file_path,state,feature_set,model_name,metric_name,value,timestamp
```

Long metrics row count: `60`

Best model remains:

```json
{
  "feature_set": "baseline_condition_allowed",
  "model_name": "HistGradientBoostingRegressor",
  "test_rmse": 21.904398318987035,
  "test_mae": 14.531929524204388,
  "test_r2": 0.7957420130339768,
  "model_artifact_path": "/content/pavement_agentic_workspace/models/best_model.joblib",
  "source_model_artifact_path": "/content/pavement_agentic_workspace/models/model__baseline_condition_allowed__HistGradientBoostingRegressor.joblib",
  "prediction_artifact_path": "/content/pavement_agentic_workspace/predictions/predictions__baseline_condition_allowed__HistGradientBoostingRegressor.csv"
}
```

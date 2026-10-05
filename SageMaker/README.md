# SageMaker

Notebooks on training and deploying models with Amazon SageMaker — mostly time-series forecasting (ARIMA / auto-ARIMA) plus one built-in-algorithm example. Migrated from an old local `prev_Prog/PYTHON/AWS` folder: outputs stripped, AWS account ID / IAM role / profile / bucket names replaced with placeholders.

## Concepts
- **Estimator** = trains a model (a training job). **Model** = trained artifact (`model.tar.gz` in S3). **Predictor** = client that calls a deployed endpoint.
- **Container types**: *built-in algorithm* (AWS image, e.g. XGBoost via `image_uris.retrieve`); *framework / bring-your-own-script* (AWS sklearn/other image + your `train.py`); *bring-your-own-container* (your own Docker image).
- **BYO-script hooks** in `train.py`: `model_fn` (load), `input_fn` (deserialize request), `predict_fn` (inference), `output_fn` (serialize response). Serialization can live here or in the Predictor's serializer/deserializer.

## Typical workflow
1. Init boto3 + SageMaker session, get execution role.
2. Explore / split data locally → train/test CSVs.
3. Upload data to S3 (`sess.upload_data`).
4. Write `train.py` (training + the inference hooks above).
5. `SKLearn(...)` / built-in estimator → `.fit(...)` (training job in a container).
6. Model artifact persists to S3.
7. Deploy an endpoint — from the estimator, or from the saved artifact via `SKLearnModel`.
8. Predict (real-time) or batch-transform; then **delete the endpoint** (it bills while live).

## Notebooks
- `00_workflow_and_references.ipynb` — the 9-step skeleton + doc links (SageMaker SDK, training/inference toolkits).
- `01_builtin_algorithm_xgboost.ipynb` — built-in XGBoost: S3 upload → train → deploy → predict → batch transform.
- `02_byo_script_training_job.ipynb` — minimal BYO `train.py` ARIMA training job + artifact → endpoint.
- `03_byo_script_train_and_deploy.ipynb` — fuller BYO-script ARIMA: two deploy paths (from estimator / from artifact), serializers.
- `04_byo_script_deploy_variants.ipynb` — same ARIMA workflow with extra experimental deploy/serialization variants.
- `05_autoarima_tuning.ipynb` — `pmdarima.auto_arima` tuning (seasonal, out-of-sample) as a BYO script.

## Caveats
- 02–04 overlap heavily (the same ARIMA BYO-script at different stages of completeness) — candidates to merge into one canonical notebook.
- These are learning/scratch notebooks: some cells have known typos/incomplete lines (e.g. `pritn`, a missing comma after a `dependencies=[...]` arg, `hyperparams` vs `hyperparameters`) and were left as-is on purpose. Fix before running.
- Set `<ACCOUNT_ID>`, `<SAGEMAKER_EXECUTION_ROLE>`, `<AWS_PROFILE>`, `<S3_BUCKET>` to your own values. Data CSVs referenced by the notebooks are not included.

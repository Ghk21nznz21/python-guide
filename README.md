# python-guide

A personal Python and data-science reference: language notes, style, performance tooling, and reusable data-science code.

## Language & tooling
- `Tutorial.py`, `AdvancedTutorial.py` — Python by example (classes, decorators, typing, comprehensions, …).
- `PEP8.md` — style cheat sheet.
- `Tools_perfomance_debbug.py` — profiling and debugging (cProfile, tracemalloc, traceback).

## BestCode/
- `Concurrency/` — threads, asyncio, multiprocessing, hybrid.
- `DataScience/PrepareData/` — data cleaning, numeric/categorical exploration.
- `DataScience/Evaluation/` — ROC, categorical-target evaluation.
- `DataScience/TypeOfModel/` — worked examples by model type: NN (Keras tuner/Optuna), time series (ARIMA/SARIMA/LSTM), RL (TF-Agents), computer vision.
- `DataScience_ready/` — drop-in utilities: `LSTM.py`, `ExcelToCsv.py`, `EfficientGroupLambda.py`, `prettyTable.py`.

## SageMaker/
Train/deploy notebooks for Amazon SageMaker (time-series forecasting + a built-in-algorithm example). See `SageMaker/README.md` for the workflow and notebook index. They sit a bit apart from the pure-Python material — keep, or split into their own repo later.

## Note

Bring-your-own test images for the vision demos go in `Computer_Vision_almost/images/` (gitignored).

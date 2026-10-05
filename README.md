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

## Note
`AWS/` holds SageMaker train/deploy notebooks. They sit a bit apart from the pure-Python material and overlap with separate AWS notes — keep, or split into their own repo later.

Bring-your-own test images for the vision demos go in `Computer_Vision_almost/images/` (gitignored).

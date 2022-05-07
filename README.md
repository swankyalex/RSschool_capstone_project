Сapstone project for RS School Machine Learning course 2022.

This demo uses [Forest Cover](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset.

## Usage
This package allows you to train model for the forest cover type (the predominant kind of tree cover) from strictly cartographic variables 
(as opposed to remotely sensed data).
1. Clone this repository to your machine.
2. Download [Forest Cover](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
or if you have [Make](https://www.gnu.org/software/make/) util
```sh
make venv
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
You can use make commands to train logistic regression and random forest with basic configure
```sh
make train-log / make train-forest
```
Your trained model will be saved to file defined in -s path

6. You can set models and parameters in file [model_setting.py](https://github.com/swankyalex/RSschool_capstone_project/blob/master/src/forest_model/model_settings.py)
```python
models: Dict[str, Model] = {
    "log": LogisticRegression(max_iter=300, solver="liblinear"),
    "forest": RandomForestClassifier(n_estimators=100),
}

model_params: Dict[str, Dict[str, Any]] = {
    "log": {
        "1": {"C": [0.01, 0.1, 1], "fit_intercept": [True, False]},
        "2": {"C": [0.0001, 0.001, 0.01], "fit_intercept": [True, False]},
        "3": {"C": [1, 10, 100], "fit_intercept": [True, False]},
    },
```
Just add needed model to models dict and parameters to model_params dict. And it's authomatically will be added to train script [click](https://click.palletsprojects.com/en/8.1.x/) parameters and you will may use it.

7. You can set different data processing techniques in file [data_processing.py](https://github.com/swankyalex/RSschool_capstone_project/blob/master/src/forest_model/data_processing.py).
Just write function for data processing that is returning pd.Dataframe and register it in the dict
```python
processing_types = {"1": processing_1, "2": processing_2}
```

8. You can make automathically observation of your data by [Pandas_profiling](https://pandas-profiling.ydata.ai/docs/master/index.html) with one of the following commands
```sh
poetry run python scripts/data_profiling.py --d <path to csv with data>
```
```sh
make profile
```
9. Run MLflow UI with one of the following commands to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
```sh
make mlflow
```
**You will look something like this**
![MLFlow experiments example](https://i.ibb.co/gbYNy9q/mlflow.png)

10. You can make submission file with one of the following commands:
```sh
poetry run submit
```
```sh
make submission
```
11. If you have connected [Kaggle API](https://www.kaggle.com/docs/api) you can send your submission directly to kaggle competition:
```sh
make submit
```

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment with one of the following commands:
```
poetry install
```
```
make venv-dev
```
Now you can use developer instruments:
1. Formatting with [black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/) and lint with [flake8](https://flake8.pycqa.org/en/latest/)
```
make format
```
or
```
poetry run [black/flake8/isort]
```
![Formatting example](https://i.ibb.co/tbfB5Vr/flake8.png)

2. Typechecking with [mypy](https://mypy.readthedocs.io/en/stable/) with one of the following commands:
```
make mypy
```
```
poetry run mypy
```
![Mypy example](https://i.ibb.co/pwx3Mjw/mypy.png)

3. Make tests with [pytest](https://docs.pytest.org/en/7.1.x/) with one of the following commands:
```
make test
```
```
poetry run pytest
```
![Tests example](https://i.ibb.co/wSZPSLS/tests.png)

More conveniently, to run all sessions of testing and formatting in a single command, you can use [nox](https://nox.thea.codes/en/stable/) with:
```
make nox
```
or
```
poetry run nox [-r]
```
![Nox example](https://i.ibb.co/0txKXqy/nox.png)




include ./Makefile.in.mk


.PHONY: format
format:
	$(call log, reorganizing imports & formatting code)
	$(RUN) isort "$(DIR_SRC)" "$(DIR_SCRIPTS)" "$(DIR_TESTS)" noxfile.py
	$(RUN) black "$(DIR_SRC)" "$(DIR_SCRIPTS)" "$(DIR_TESTS)" noxfile.py
	$(RUN) flake8 "$(DIR_SRC)" "$(DIR_SCRIPTS)" "$(DIR_TESTS)" noxfile.py
	$(call log, All good!)


.PHONY: mypy
mypy:
	$(call log, mypy is running)
	$(RUN) mypy "$(DIR_SRC)" noxfile.py
	$(call log, All good!)


.PHONY: format-full
format-full: format mypy
	$(call log, code formatted)


.PHONY: nox
nox:
	$(call log, running nox tests)
	$(RUN) nox -r


.PHONY: test
test:
	$(call log, running tests)
	$(RUN) pytest


.PHONY: profile
profile:
	$(call log, making pandas-profiling on data)
	$(PYTHON) "$(DIR_SCRIPTS)\data_profiling.py"


.PHONY: train-log
train-log:
	$(call log, training log regression)
	$(PYTHON) "$(DIR_TRAIN)\train.py" --model-name log


.PHONY: train-forest
train-forest:
	$(call log, training random forest)
	$(PYTHON) "$(DIR_TRAIN)\train.py" --model-name forest


.PHONY: submission
submission:
	$(call log, making forest submission)
	$(PYTHON) "$(DIR_TRAIN)\submission.py" --model-name forest


.PHONY: submission-log
submission-log:
	$(call log, making log legression submission)
	$(PYTHON) "$(DIR_TRAIN)\submission.py" --model-name log


.PHONY: submit
submit:
	$(call log, making submit to kaggle)
	kaggle competitions submit -c forest-cover-type-prediction -f "$(DIR_DATA)\submission.csv" -m "Message"


.PHONY: mlflow
mlflow:
	$(call log, ml flow is launched)
	$(RUN) mlflow ui


.PHONY: venv
venv:
	$(call log, installing packages)
	$(POETRY_INSTALL)  --no-dev


.PHONY: venv-dev
venv-dev:
	$(call log, installing development packages)
	$(POETRY_INSTALL)




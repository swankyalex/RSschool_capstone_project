include ./Makefile.in.mk


.PHONY: format
format:
	$(call log, reorganizing imports & formatting code)
	$(RUN) isort "$(DIR_SRC)" "$(DIR_SCRIPTS)" noxfile.py
	$(RUN) black "$(DIR_SRC)" "$(DIR_SCRIPTS)" noxfile.py
	$(RUN) flake8 "$(DIR_SRC)" "$(DIR_SCRIPTS)" noxfile.py
	$(call log, All good!)


.PHONY: mypy
mypy:
	$(call log, mypy is running)
	$(RUN) mypy "$(DIR_SRC)" noxfile.py
	$(call log, All good!)


.PHONY: full-format
full-format: format mypy
	$(call log, full formatting)


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


.PHONY: mlflow
mlflow:
	$(call log, ml flow is launched)
	$(RUN) mlflow ui


.PHONY: venv
venv:
	$(call log, installing packages)
	$(PIPENV_INSTALL)


.PHONY: venv-dev
venv-dev:
	$(call log, installing development packages)
	$(PIPENV_INSTALL) --dev




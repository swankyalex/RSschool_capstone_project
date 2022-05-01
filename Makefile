include ./Makefile.in.mk


.PHONY: format
format:
	$(call log, reorganizing imports & formatting code)
	$(RUN) isort "$(DIR_SRC)" "$(DIR_SCRIPTS)"
	$(RUN) black "$(DIR_SRC)" "$(DIR_SCRIPTS)"
	$(RUN) flake8 "$(DIR_SRC)" "$(DIR_SCRIPTS)"
	$(call log, All good!)


.PHONY: full-format
full-format:
	$(call log, reorganizing imports & formatting code)
	$(RUN) isort "$(DIR_SRC)" "$(DIR_SCRIPTS)"
	$(RUN) black "$(DIR_SRC)" "$(DIR_SCRIPTS)"
	$(RUN) flake8 "$(DIR_SRC)" "$(DIR_SCRIPTS)"
	$(RUN) mypy "$(DIR_SRC)"
	$(call log, All good!)


.PHONY: profile
profile:
	$(call log, making pandas-profiling on data)
	$(PYTHON) "$(DIR_SCRIPTS)\data_profiling.py"


.PHONY: train-log
train-log:
	$(call log, training log regression)
	$(PYTHON) "$(DIR_TRAIN)\train.py" --model log


.PHONY: train-forest
train-forest:
	$(call log, training random forest)
	$(PYTHON) "$(DIR_TRAIN)\train.py" --model forest


.PHONY: mlflow
mlflow:
	$(call log, ml flow is launched)
	$(RUN) mlflow ui


.PHONY: run-prod
run-prod:
	$(call log, starting local web server)
	$(RUN) gunicorn --config="$(DIR_SCRIPTS)/gunicorn.conf.py" project.wsgi:application

.PHONY: sh
sh:
	$(call log, starting Python shell)
	$(PYTHON) src/manage.py shell


.PHONY: venv
venv:
	$(call log, installing packages)
	$(PIPENV_INSTALL)


.PHONY: venv-dev
venv-dev:
	$(call log, installing development packages)
	$(PIPENV_INSTALL) --dev


.PHONY: data
data: static migrate
	$(call log, preparing data)


.PHONY: static
static:
	$(call log, collecting static)
	$(PYTHON) src/manage.py collectstatic --noinput


.PHONY: su
su:
	$(call log, starting Python shell)
	$(PYTHON) src/manage.py createsuperuser


.PHONY: migrations
migrations:
	$(call log, generating migrations)
	$(PYTHON) src/manage.py makemigrations


.PHONY: migrate
migrate:
	$(call log, applying migrations)
	$(PYTHON) src/manage.py migrate

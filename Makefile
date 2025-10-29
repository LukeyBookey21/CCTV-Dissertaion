.PHONY: install dev lint test run

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

lint:
	black --check .
	flake8

test:
	pytest -q

run:
	python -m cctv_dissertation.app

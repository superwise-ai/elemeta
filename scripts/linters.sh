#!/bin/bash -ex

poetry install --only lint
poetry run flake8 --max-line-length=99 elemeta 
poetry run python -m mypy ./elemeta

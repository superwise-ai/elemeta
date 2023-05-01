#!/bin/bash -ex

poetry install --only lint
poetry run isort elemeta --line-length 99 
poetry run black elemeta
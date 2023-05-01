#!/bin/bash -ex
# shellcheck disable=SC2164

poetry install --with docs
poetry run sphinx-apidoc -o ./docs ./elemeta --force
cd docs
poetry run make html

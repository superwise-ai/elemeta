#!/bin/bash -ex

poetry check
poetry lock --no-update
poetry install --only root
poetry build
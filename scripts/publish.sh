#!/bin/bash -ex
set -e

# Args
version=$1

# Bump version
poetry version $version

# Publish package
poetry publish --skip-existing --build --username $PYPI_USERNAME --password $PYPI_PASSWORD

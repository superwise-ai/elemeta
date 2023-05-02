#!/bin/bash -ex
# shellcheck disable=SC2164

# Install docs dependencies
poetry install --with docs

# Build docs
poetry run sphinx-apidoc -o ./docs ./elemeta --force
rm -rf ./docs/_build
poetry run sphinx-build -M html ./docs ./docs/_build -v --color -w /tmp/sphinx-build.log

# Check for errors in sphinx-build
SPHINX_ERRORS=$(grep -E '.*: ERROR:.*' /tmp/sphinx-build.log | wc -l | sed -e 's/ //g' || true)
if [ $SPHINX_ERRORS -gt 0 ]
then
  echo "Sphinx build had ${SPHINX_ERRORS} errors. Exiting."
  exit 1
fi

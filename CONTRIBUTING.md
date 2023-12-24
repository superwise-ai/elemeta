# Contributing guidelines

---

We're always open to contributions of any size! But before you start, read through our guidelines so your PR can be merged quickly.

## How to contribute:

1. If this is your first time here, check out our issues page and filter for ["good first issue"](https://github.com/superwise-ai/elemeta/labels/good%20first%20issue) for inspiration on where to get started.
2. Fork the repository.
3. Make your changes in your local branch.
4. Follow our repo [guidelines](#guidelines).
5. Check to make sure that your commits follow the package standards and then open a new PR request to the "main" branch.

---

## <a id="guidelines"></a>Guidelines:

### Formatting and linting

We use isort and black formatters in Elemeta. To apply these formats automatically using [pre-commit](https://pre-commit.com/#install), please run the following command from the repo root:

```sh
pre-commit run --all-files
```

To make sure that formatting and linting rules are applied, you can install a Git hook that will run `pre-commit` on each commit:

```sh
pre-commit install
```

### Documentation

Always make sure that you update the docs regardless if you applied code changes directly or if your contribution was direct to the package documentation! Please first run the documentation build step locally to ensure it looks good. To build the documentation run:

```sh
./scripts/docs.sh
open docs/_build/html/index.html
```

### Tests

Before committing any new code change, please run the full test suite to ensure its quality and coverage. To do this, please run

```sh
./scripts/test.sh
```

You can run the script directly by running the following: `Poetry run pytest`
but the _test.sh_ script is recommended as it will ensure that the test coverage is above the required coverage level.

### Commit format

We love a good and concise commit message 😉, and in addition, commit messages will automatically create the package versions and changelog based on [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/#summary) messages.
The full list of supported commit types can be found in the release [configuration](https://github.com/superwise-ai/elemeta/blob/main/.releaserc.json#L11C32-L17).

#### **When changing code**

When a code changes and the following commit types are used, a new version is created and published using [GitHub Actions](https://github.com/superwise-ai/elemeta/actions/workflows/release.yaml).

| Commit message                                                                                                                                                | Version type | When to use                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `fix(tests): Fix typo`                                                                                                                                        | `PATCH`      | A bug fix                                                                                                                                   |
| `feat(dataset): add IMDB reviews dataset`                                                                                                                     | `MINOR`      | A new feature                                                                                                                               |
| `perf(sentiment_polarity): Use new library to improve performance`                                                                                            | `PATCH`      | A code change that improves performance                                                                                                     |
| `refactor(dependencies): Remove unused dependencies`                                                                                                          | `PATCH`      | A code change that neither fixes a bug nor adds a feature                                                                                   |
| `refactor(extractors): replace unique_word_count with unique_words_count`<br><br>`BREAKING CHANGE: unique_word_count was renamed and will not work anymore .` | `MAJOR`      | A breaking release is counted when a `BREAKING CHANGE: <description>` message is in the end of the commit message (separated by a new line) |
| `deps(deps): Bump version of numpy to 1.21.0`                                                                                                                 | `PATCH`      | A dependency update                                                                                                                         |

#### **When changing other files**

When files in the repo, other than the code, are changed, no new version is released and the following commit types should be used:

| Commit message                        | When to use                                                               |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `chore(formatting): Format files`     | Generic changes in the repo that should be included in the release notes  |
| `ci(release): Update release options` | Changes in the CI workflows                                               |
| `docs(readme): Update logo`           | Updates to the documentation that should be included in the release notes |

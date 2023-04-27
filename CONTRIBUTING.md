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

### Formatting
We use isort and black formatters in Elemeta. To apply these formats automatically, please run the following command from the repo root 
```sh
scripts/formatters.sh
```

### Documentation
Always make sure that you update the docs regardless if you applied code changes directly or if your contribution was direct to the package documentation! Please first run the documentation build step locally to ensure it looks good. To build the documentation run: 
```sh
poetry install --with docs
./scripts/docs.sh
open docs/_build/html/index.html
```

### Tests
Before committing any new code change, please run the full test suite to ensure its quality and coverage. To do this, please run 
```sh
./script/test.sh
```

You can run the script directly by running the following: `Poetry run pytest`
but the *test.sh* script is recommended as it will ensure that the test coverage is above the required coverage level.

### Commit format
We love a good and concise commit message 😉, and in addition, commit messages will automatically create the repo changelog based on *"chore"* conventional commit messages, and new releases will be versioned based on [semantic release](https://github.com/semantic-release/semantic-release). For example:

To release a new hotfix release, the commit message should start with:
```sh
fix(topic): message
```
To release a new minor version, the commit message should start with 
```sh
feat(topic): message
```
For breaking changes, the commit message should start with 
```sh
BREAKING CHANGE: message
```


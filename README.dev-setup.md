# Workspace setup notes

This file lists recommended editor extensions, how to create a Python virtualenv, install deps, run tests, and enable pre-commit.

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

1. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

1. Install pre-commit hooks

```bash
pre-commit install
```

1. Running tests and lint

```bash
make test
make lint
```

## Notes

- `torch` may require platform-specific wheels. If you have a GPU, follow the official
 installation instructions for the appropriate `torch` wheel instead of relying on the
 generic `pip install` (the official docs show CPU vs CUDA wheel choices).
- For markdown linting you can install the CLI with `npm i -g markdownlint-cli` or use
 the `DavidAnson.vscode-markdownlint` extension for VS Code.

## Notes on imports

During development we use the `src/` layout and (optionally) run tests with
`PYTHONPATH=src` so the `cctv_dissertation` package is discoverable without a pip
install. This is a simple and quick approach for local development.

The preferred, more robust workflow is to install the package in editable mode
(`pip install -e .`) so imports work without modifying `PYTHONPATH`. CI should use
an editable install for reproducible test runs. See the "Example (run tests)" section
below for commands.

## Example (run tests)

Preferred: install the package in editable mode (recommended)

```bash
pip install -e .
.venv/bin/pytest -q
```

This installs the `cctv_dissertation` package into your virtualenv so imports work
without manipulating PYTHONPATH. If you prefer the PYTHONPATH approach you can still
use `PYTHONPATH=src .venv/bin/pytest -q`, but editable install is recommended for a
stable developer experience.

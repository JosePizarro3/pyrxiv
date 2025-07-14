<h4 align="center">

![CI](https://github.com/JosePizarro3/pyrxiv/actions/workflows/actions.yml/badge.svg)
![Coverage](https://coveralls.io/repos/github/JosePizarro3/pyrxiv/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<!-- [![PyPI version](https://img.shields.io/pypi/v/pyrxiv.svg)]
[![Python versions](https://img.shields.io/pypi/pyversions/pyrxiv.svg)] -->

</h4>

# pyrxiv

**pyrxiv** is a Python package for retrieving [arXiv](https://arxiv.org) papers, storing their metadata in [pydantic](https://docs.pydantic.dev/latest/)-like classes, and optionally filtering some of them out based on the specific content of the papers (matching a regex pattern).

While originally developed for the **Strongly Correlated Electron Systems** community in Condensed Matter Physics ([`cond-mat.str-el`](https://arxiv.org/list/cond-mat.str-el/recent)), it's designed to be flexible and applicable to **any arXiv category**.

Install the core package:
```bash
pip install pyrxiv
```

## Objective
**pyrxiv** main objective is to provide an easy command line interface (CLI) to search and download arXiv papers which contain a specific content string matched against a regex pattern. You can use the CLI and print the options after installing the package using:
```sh
pyrxiv --help
```

or directly:
```sh
pyrxiv search_and_download --help
```

For example:
```sh
pyrxiv search_and_download --category cond-mat.str-el --regex-pattern "DMFT|Hubbard" --n-papers 5
```

---

# Development

To contribute to `pyrxiv` or run it locally, follow these steps:


## Clone the Repository

```bash
git clone https://github.com/JosePizarro3/pyrxiv.git
cd pyrxiv
```

## Set Up a Virtual Environment

We recommend Python ≥ 3.10:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

Use [`uv`](https://docs.astral.sh/uv/) (faster than pip) to install the package in editable mode with `dev` extras:
```bash
pip install --upgrade pip
pip install uv
uv pip install -e .[dev]
```

## Run tests

Use `pytest` with verbosity to run all tests:
```bash
python -m pytest -sv tests
```


To check code coverage:
```bash
python -m pytest --cov=pyrxiv tests
```

### Code formatting and linting


We use [`Ruff`](https://docs.astral.sh/ruff/) for formatting and linting (configured via `pyproject.toml`).

Check linting issues:
```bash
ruff check .
```

Auto-format code:
```bash
ruff format . --check
```

Manually fix anything Ruff cannot handle automatically.

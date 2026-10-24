[![pipeline](https://github.com/michaelosthege/pyds/workflows/tests/badge.svg)](https://github.com/michaelosthege/pyds/actions)

A Python library for performing calculations in the Dempster-Shafer theory of evidence.

Written by https://github.com/reineking/pyds, refactored for a modern Python project structure.

## Features

- Support for normalized as well as unnormalized belief functions
- Different Monte-Carlo algorithms for combining belief functions
- Various methods related to the generalized Bayesian theorem
- Measures of uncertainty
- Methods for constructing belief functions from data

See [`examples.py`](examples.py) for how to use it.

## Development

Use [`uv`](https://astral.sh/uv) to run:

* `uv run pre-commit run -a`
* `uv run pytest -vs --cov=pyds tests`
* `uv run python examples.py`
* `uv run python stats.py`

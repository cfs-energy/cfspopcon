cfspopcon: 0D Plasma Calculations & Plasma OPerating CONtours
--------------------------------------------------------------

[![Build Status](https://github.com/cfs-energy/cfspopcon/actions/workflows/workflow_actions.yml/badge.svg)](https://github.com/cfs-energy/cfspopcon/actions)
[![Documentation Status](https://readthedocs.org/projects/cfspopcon/badge/?version=latest)](https://cfspopcon.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cfs-energy/cfspopcon/HEAD)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10054879.svg)](https://doi.org/10.5281/zenodo.10054879)

POPCONs (Plasma OPerating CONtours) is a tool developed to explore the performance and constraints of tokamak designs based on 0D scaling laws, model plasma kinetic profiles, and physics assumptions on the properties and behavior of the core plasma.

All of our documentation is available at [cfspopcon.readthedocs.io](https://cfspopcon.readthedocs.io/en/latest/). There, you can find installation instructions, instructions for how to run `cfsPOPCON` via the command-line-interface and also explanations of the example Jupyter notebooks in [docs/doc_sources](docs/doc_sources).

To re-generate the saved regression reference outputs in `tests/regression_results`, run the existing CLI helper from the repository root:

```bash
poetry run python tests/utils/regression_results.py
```

### Checks before opening a PR

```bash
poetry run pytest --cov-fail-under=84  # tests plus coverage; CI enforces the 84% floor, so a plain `pytest` or a partial run does not fail on it
poetry run ty check                          # type check (ty is pinned, since its diagnostics change between releases)
poetry run pre-commit run --all-files        # ruff check, ruff format and ty; `pre-commit install` (once) also rejects AI-attribution trailers in commit messages
```

CI runs the same checks.

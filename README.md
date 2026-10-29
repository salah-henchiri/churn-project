# churn-project

Example project of Section 2: the churn model of Section 1, packaged as a Python project.

    uv sync                                  # create .venv and install everything
    uv run churn-train --config configs/train.yaml
    uv run pytest
    uv run ruff check . && uv run ruff format --check .

Companion notebook: `uv run --with jupyter --with matplotlib jupyter lab ../../notebooks/02_using_the_package.ipynb`

# churn-project with MLflow (Section 3)

The project of Section 2, extended with experiment tracking and the model registry.

    uv sync

    # terminal 1: the tracking server (UI at http://127.0.0.1:5000, choose "Model training")
    uv run mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5000

    # terminal 2
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    uv run churn-train --run-name first-run        # one run
    uv run churn-experiments                       # the 8 configurations of configs/experiments.yaml
    uv run churn-register                          # best run -> churn-model, alias "champion"
    uv run pytest                                  # 24 tests (they use a temporary MLflow store)

Serve the champion for a quick test:

    uv run mlflow models serve -m "models:/churn-model@champion" --env-manager local -p 5001

Companion notebook: `uv run --with jupyter jupyter lab ../../notebooks/03_mlflow.ipynb`

Note: rename `RENAME_TO_.pre-commit-config.yaml.txt` to `.pre-commit-config.yaml` (Section 2.9).

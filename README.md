# churn-project with DVC (Section 4)

The project of Section 3, organised as a reproducible DVC pipeline
(prepare -> train -> evaluate) with versioned data and models.

## First use

    uv sync
    # the data are versioned by DVC; get them from the DVC remote...
    uv run dvc pull
    # ...or, the first time, download the original file (same bytes as data v2):
    curl -L -o data/raw/telco_churn.csv \
      https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
    uv run dvc status          # the raw file must match data/raw/telco_churn.csv.dvc

    uv run dvc repro           # runs only the stages whose inputs changed
    uv run dvc metrics show
    uv run pytest              # 36 tests

The DVC remote is the folder `../dvc-remote` (next to the project), set in `.dvc/config`.
Replace it with cloud storage when you have one, for instance:

    uv run dvc remote add -d storage s3://my-bucket/dvc
    uv run dvc push

## Useful commands

    uv run dvc dag                              # the pipeline as a graph
    uv run dvc params diff                      # parameters changed since the last commit
    uv run dvc metrics diff                     # metrics changed since the last commit
    uv run dvc exp run -n logreg-C10 -S model.params.C=10
    uv run dvc exp show

Each run of the `train` stage is also logged in MLflow (./mlflow.db, or the server given
by MLFLOW_TRACKING_URI), with the DVC hash of the data as the tag `data_dvc_md5`.

Companion notebook: `uv run --with jupyter jupyter lab ../../notebooks/04_data_versioning.ipynb`

Note: rename `RENAME_TO_.pre-commit-config.yaml.txt` to `.pre-commit-config.yaml` (Section 2.9).

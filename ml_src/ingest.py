# https://raw.githubusercontent.com/scikit-learn/scikit-learn/refs/heads/main/sklearn/datasets/data/breast_cancer.csv
# maybe have ingest pull from the URI for more realistic data ingest?
import os
import logging
import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer

LOGGER = logging.getLogger("mlflow")
LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO"))

def ingest_data(output_path:str, csv_src_uri:str|None=None):
    mlflow.set_experiment(os.getenv("DATA_5750_EXPERIMENT_NAME", default="give_me_a_name"))
    with mlflow.start_run(run_name="data_ingestion"):
        if csv_src_uri:
            df = pd.read_csv(csv_src_uri)
        else:
            data = load_breast_cancer(as_frame=True)
            df = data.frame
        os.makedirs("data", exist_ok=True)
        df.to_csv(output_path, index=False)
        mlflow.log_param("rows", len(df))
        mlflow.log_artifact(output_path)
        LOGGER.info(f"Data saved to: {output_path}")

if __name__ == "__main__":
    ingest_data(
        csv_src_uri=os.getenv("DATA_5750_INGEST_URI", default=None),
        output_path=os.getenv("DATA_5750_INGEST_DATA", default="data/bc_data.csv")
    )
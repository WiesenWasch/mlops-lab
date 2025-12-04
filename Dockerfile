# docker build . -t mlops1 --no-cache
# need to run with no-cache so the sqlite db gets generated anew each time

# running with high vulns as i am too lazy to build a non-slim image with mlflow reqs of git and bash
FROM python:3.12.12

ENV MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
ENV DATA_5750_EXPERIMENT_NAME="my_cool_model"
ENV DATA_5750_INGEST_URI=
ENV DATA_5750_INGEST_DATA="data/bc_data.csv"
# mlflow ui --backend-store-uri sqlite:///mlflow.db  <- how to launch ui

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ingest and train data
# i know this is better done in CI/CD pipeline step but i am lazy and this data set is small
COPY ./ml_src /ml_src
COPY MLproject .
COPY python_env.yaml .
RUN mlflow run -e ingest --env-manager=local --experiment-name=$DATA_5750_EXPERIMENT_NAME .
RUN mlflow run -e train --env-manager=local --experiment-name=$DATA_5750_EXPERIMENT_NAME .

WORKDIR /app
# don't need to copy the sqlite db if you are using a remote db
RUN mv /mlflow.db .
COPY ./api/app.py ./app.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
import time
from fastapi import FastAPI, Response
from pydantic import BaseModel
import mlflow
import numpy as np

class Features(BaseModel):
    data: list[dict]

class RunConfig(BaseModel):
    run_id: str

class MyModel:
    def __init__(self, model=None):
        self.model = model

MODEL = MyModel()

app = FastAPI(title="CancerClassificationAPI")
mlflow.set_experiment("mlflow_monitoring")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", status_code=200)
def predict(features: Features, _response: Response):
    if MODEL.model is None:
        top_model = mlflow.search_logged_models(
            experiment_ids=["1"],
            max_results=1,
        )
        model_id = top_model['model_id'].iloc[0]
        MODEL.model = mlflow.pyfunc.load_model(f"models:/{model_id}")

    # df = pd.DataFrame(features.data) # features need to be ordered but this is an OPs not dev project
    start = time.time()
    preds = MODEL.model.predict(features.data)
    latency = time.time() - start

    # logging model latency in code
    with mlflow.start_run(run_name="inference", nested=True):
        mlflow.log_metric("latency_ms", latency * 1000)
        mlflow.log_param("batch_size", len(features.data))

    return {"predictions": preds.tolist(), "prediciton_latency":latency}

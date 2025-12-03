from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="CancerClassificationAPI")
MODEL = None

class Features(BaseModel):
    data: list[dict]

class RunConfig(BaseModel):
    run_id: str

class MyModel:
    def __init__(self, model=None):
        self.model = model

MODEL = MyModel()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", status_code=200)
def predict(features: Features, response: Response):
    print(MODEL)
    if MODEL.model is not None:
        print(features.data)
        df = pd.DataFrame(features.data)
        print(df)
        preds = MODEL.model.predict(features.data)
        print(preds)
        return {"predictions": preds.tolist()}
    response.status_code = status.HTTP_428_PRECONDITION_REQUIRED
    return "NO MODEL LOADED"

@app.post("/set_model", status_code=200)
def set_model(run_config:RunConfig, response: Response):
    """
    loads the model associate with the given Run ID.
    Returns the Run ID if successful
    """
    model_uri = f"mlruns/1/models/{run_config.run_id}/artifacts"
    try:
        MODEL.model = mlflow.pyfunc.load_model(model_uri)
        response.status_code = status.HTTP_201_CREATED
        return model_uri
    except Exception as err:
        print(err)
        MODEL.model=None
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return f"Unable to load model for Run {run_config.run_id}"


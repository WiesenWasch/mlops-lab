from fastapi import FastAPI, Response
from pydantic import BaseModel
import mlflow

class Features(BaseModel):
    data: list[dict]

class RunConfig(BaseModel):
    run_id: str

class MyModel:
    def __init__(self, model=None):
        self.model = model

MODEL = MyModel()

app = FastAPI(title="CancerClassificationAPI")

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
    preds = MODEL.model.predict(features.data)
    return {"predictions": preds.tolist()}

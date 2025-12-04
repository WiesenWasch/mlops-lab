import io
import time
from fastapi import FastAPI, Response
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import mlflow
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

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
plt.switch_backend('Agg') # this needs to be set so matplotlib doesn't open a gui

@app.get("/health")
def health():
    return {"status": "healthy"}

def setup_model():
    top_model = mlflow.search_logged_models(
        experiment_ids=["1"],
        max_results=1,
    )
    model_id = top_model['model_id'].iloc[0]
    MODEL.model = mlflow.pyfunc.load_model(f"models:/{model_id}")

def model_predicition(data):
    # df = pd.DataFrame(data) # features need to be ordered but this is an OPs not dev project
    start = time.time()
    preds = MODEL.model.predict(data)
    latency = time.time() - start

    # logging model latency in code
    with mlflow.start_run(run_name="inference", nested=True):
        mlflow.log_metric("latency_ms", latency * 1000)
        mlflow.log_param("batch_size", len(data))

    return {"predictions": preds.tolist(), "prediciton_latency":latency}

@app.post("/predict", status_code=200)
def predict(features: Features, _response: Response):
    if MODEL.model is None:
        setup_model()

    result = model_predicition(features.data)

    return result

@app.get("/explain", status_code=200)
def explain():
    if MODEL.model is None:
        setup_model()

    mlflow.set_experiment("mlflow_interpretability")
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    explainer = shap.Explainer(MODEL.model.predict, X_test)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

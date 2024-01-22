from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
import mlflow
import os
import uvicorn

# if you are using an IDE like PyCharm, mark the capstone directory as 'Sources Root'
#   to avoid the line below being highlighted in red
from schemas.credit_card_transaction import ModelInput
import pickle
import logging


app = FastAPI(title="credit-card-fraud-detection")
logging.basicConfig(level=logging.INFO, format="%(message)s")

mlflow_enabled = os.getenv("MLFLOW_ENABLED")

if mlflow_enabled:
    MLFLOW_TRACKING_URI = f"http://{os.getenv('MLFLOW_HOST')}"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    RUN_ID = os.getenv("RUN_ID")


def load_model():
    if mlflow_enabled:
        logged_model = f"runs:/{RUN_ID}/model"
        model = mlflow.pyfunc.load_model(logged_model)
    else:
        with open("/ml_models/pipeline.pkl", "rb") as file_in:
            model = pickle.load(file_in)
    return model


def apply_model(input_data: dict[str, int | float]):
    logging.info("Loading the ML model artifacts...")
    model = load_model()
    logging.info("Successfully loaded the ML model artifacts")
    logging.info("Applying the model...")
    preds = model.predict(input_data)
    return preds[0]


@app.get("/health")
def health_check():
    return JSONResponse(status_code=200, content={"status": "healthy!"})


@app.post("/predict")
def predict_endpoint(model_input: ModelInput):
    logging.info("Validating the input data...")
    request = model_input.model_dump()
    logging.info("Successfully validated the input data")
    try:
        pred = apply_model(request)
        is_fraud = pred == 1
        if mlflow_enabled:
            result = {"is_fraud": is_fraud, "model_version": RUN_ID}
        else:
            result = {"is_fraud": is_fraud}
        return str(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.predict:app", host="0.0.0.0", port=8000, log_level="debug")

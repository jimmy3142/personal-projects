import mlflow
import os

MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

RUN_ID = os.getenv("RUN_ID", "ee2daf18edec4effb824d1b3e7bf2520")


def load_model(run_id: str):
    logged_model = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_data: dict[str, int | float]):
    model = load_model(RUN_ID)
    print(f"{input_data = }")
    preds = model.predict(input_data)
    return preds[0]


def predict_endpoint(transaction):
    try:
        pred = apply_model(transaction)
        is_fraud = pred == 1
        result = {"is_fraud": is_fraud, "model_version": RUN_ID}
        return result
    except ValueError as e:
        return {"detail": str(e)}

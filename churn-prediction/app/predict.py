from flask import Flask
from flask import request
from flask import jsonify
import pickle
import xgboost as xgb

app = Flask("customer_churn")

model_file = "xgboost_model.bin"
with open(model_file, "rb") as file_in:
    dv, model = pickle.load(file_in)


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()
    features = list(dv.get_feature_names_out())

    X = dv.transform([customer])
    d_matrix = xgb.DMatrix(X, feature_names=features)
    y_pred = model.predict(d_matrix)
    churn = y_pred >= 0.5

    result = {
        "customer_id": customer["customer_id"],
        "churn probability": float(y_pred),
        "churn": bool(churn),
    }
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)

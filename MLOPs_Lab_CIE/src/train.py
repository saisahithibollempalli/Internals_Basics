import pandas as pd
import numpy as np
import mlflow
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("../data/training_data.csv")

X = df.drop("land_stability_score", axis=1)
y = df["land_stability_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("geosurvey-land-stability-score")

models = {
    "Ridge": Ridge(),
    "GradientBoosting": GradientBoostingRegressor()
}

results = []
best_mae = float("inf")

for name, model in models.items():
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("priority", "high")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = name

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

joblib.dump(best_model, "../models/model.pkl")

output = {
    "experiment_name": "geosurvey-land-stability-score",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

with open("../results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)
import pandas as pd
import mlflow
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("../data/training_data.csv")

X = df.drop("land_stability_score", axis=1)
y = df["land_stability_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

mlflow.set_experiment("tuning-geosurvey")

model = GradientBoostingRegressor()

grid = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_absolute_error")

grid.fit(X_train, y_train)

best_params = grid.best_params_
best_mae = -grid.best_score_

output = {
    "search_type": "grid",
    "n_folds": 3,
    "total_trials": len(grid.cv_results_["params"]),
    "best_params": best_params,
    "best_mae": best_mae,
    "best_cv_mae": best_mae,
    "parent_run_name": "tuning-geosurvey"
}

with open("../results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)
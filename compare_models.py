# Train several regression models on the wine-quality data, log each as its own
# MLflow run, then register only the best one (lowest RMSE).
#
# Run:  python compare_models.py
# View: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
#       -> experiment "wine-quality-comparison" -> select runs -> Compare

import warnings

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

EXPERIMENT_NAME = "wine-quality-comparison"
REGISTERED_MODEL_NAME = "BestWineModel"
CSV_URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"

# (model name, class, list of hyperparameter settings to try)
CANDIDATES = [
    ("ElasticNet", ElasticNet, [
        {"alpha": 0.1, "l1_ratio": 0.5},
        {"alpha": 0.5, "l1_ratio": 0.5},
        {"alpha": 0.1, "l1_ratio": 0.9},
    ]),
    ("Ridge", Ridge, [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}]),
    ("Lasso", Lasso, [{"alpha": 0.01}, {"alpha": 0.1}]),
    ("RandomForest", RandomForestRegressor, [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
    ]),
    ("GradientBoosting", GradientBoostingRegressor, [
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
    ]),
    ("XGBoost", XGBRegressor, [
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4},
        {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6},
    ]),
]


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return rmse, mean_absolute_error(actual, pred), r2_score(actual, pred)


def main():
    warnings.filterwarnings("ignore")

    data = pd.read_csv(CSV_URL, sep=";")
    train, test = train_test_split(data, random_state=40)
    train_x, test_x = train.drop(columns="quality"), test.drop(columns="quality")
    train_y, test_y = train["quality"], test["quality"]

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []  # (rmse, model_name, params, run_id)
    for model_name, model_cls, param_grid in CANDIDATES:
        for params in param_grid:
            run_name = model_name + "-" + "-".join(f"{k}={v}" for k, v in params.items())
            with mlflow.start_run(run_name=run_name) as run:
                model = model_cls(**params, random_state=42)
                model.fit(train_x, train_y)
                preds = model.predict(test_x)
                rmse, mae, r2 = eval_metrics(test_y, preds)

                mlflow.set_tag("model_type", model_name)
                mlflow.log_params(params)
                mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

                signature = infer_signature(train_x, model.predict(train_x))
                if model_name == "XGBoost":
                    mlflow.xgboost.log_model(model, name="model", signature=signature,
                                             input_example=train_x.head(3))
                else:
                    # Tree models (RandomForest, GradientBoosting) store a
                    # sklearn Tree object; we trained it ourselves, so it's safe.
                    mlflow.sklearn.log_model(model, name="model", signature=signature,
                                             input_example=train_x.head(3),
                                             skops_trusted_types=["sklearn.tree._tree.Tree"])

                results.append((rmse, model_name, params, run.info.run_id))
                print(f"{run_name:<55} RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    # Register only the best model
    best_rmse, best_name, best_params, best_run_id = min(results, key=lambda r: r[0])
    print(f"\nBest: {best_name} {best_params}  RMSE={best_rmse:.4f}")
    mlflow.register_model(f"runs:/{best_run_id}/model", REGISTERED_MODEL_NAME)


if __name__ == "__main__":
    main()

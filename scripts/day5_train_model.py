"""
Day 5 — Model Training

Goal:
Train a model to predict the NEXT lap time using leakage-free features.

What this script does:
- Loads the feature-engineered CSV from Day 4
- Splits the data safely
- Trains baseline models
- Evaluates them using MAE, RMSE, and R²
- Saves the best model to the models folder
"""

import sys
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = Path.cwd().resolve().parent

sys.path.append(str(ROOT / "src"))

from config import PROCESSED_DIR, MODELS_DIR  # noqa: E402


def load_feature_data() -> pd.DataFrame:
    file_path = PROCESSED_DIR / "day4_feature_engineered_data.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Run Day 4 feature engineering first."
        )
    print(f"Loading feature data: {file_path.name}")
    return pd.read_csv(file_path)


def prepare_dataset(df: pd.DataFrame):
    df = df.copy()

    target_col = "Target_NextLapTimeSeconds"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    df = df[df[target_col].notna()].copy()

    # Drop only direct leakage columns
    leakage_cols = [
        target_col,
        "LapTimeSeconds",
        "LapTime",
    ]

    X = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    y = df[target_col]

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def main():
    print("## Day 5 — Model Training\n")

    df = load_feature_data()
    print(f"Loaded dataset shape: {df.shape}")

    X, y = prepare_dataset(df)

    print("\nFeature columns being used:")
    print(X.columns.tolist())
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # For this MVP, a simple random split is okay after leakage removal.
    # Later, you can replace this with time-aware splitting.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = build_preprocessor(X)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            random_state=42
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        results[name] = metrics
        fitted_pipelines[name] = pipeline

        print(f"{name} results:")
        print(json.dumps(metrics, indent=2))

    best_model_name = min(results, key=lambda k: results[k]["RMSE"])
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\nBest model: {best_model_name}")
    print("Best metrics:")
    print(json.dumps(best_metrics, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "day5_best_next_lap_time_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved best model to: {model_path}")

    metrics_path = MODELS_DIR / "day5_model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "all_models": results,
                "best_model": best_model_name,
                "best_metrics": best_metrics,
            },
            f,
            indent=2
        )
    print(f"Saved metrics to: {metrics_path}")

    print("\n## Day 5 complete")
    print("You now have a leakage-free baseline model for next-lap prediction.")


if __name__ == "__main__":
    main()
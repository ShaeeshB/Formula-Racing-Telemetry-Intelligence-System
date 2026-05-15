import sys
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = Path.cwd().resolve().parent

sys.path.append(str(ROOT / "src"))

from config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR


def load_feature_data() -> pd.DataFrame:
    file_path = PROCESSED_DIR / "day4_feature_engineered_data.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Run Day 4 first.")
    print(f"Loading feature data: {file_path.name}")
    return pd.read_csv(file_path)


def prepare_dataset(df: pd.DataFrame):
    df = df.copy()
    target_col = "Target_NextLapTimeSeconds"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    df = df[df[target_col].notna()].copy()
    leakage_cols = [target_col, "LapTimeSeconds", "LapTime"]
    X = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    y = df[target_col]
    return X, y, df


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
    return preds, {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def save_scatter_plot(y_test, preds):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, preds, alpha=0.5)
    min_val = min(float(np.min(y_test)), float(np.min(preds)))
    max_val = max(float(np.max(y_test)), float(np.max(preds)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.title("Actual vs Predicted Next Lap Time")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    path = FIGURES_DIR / "day6_actual_vs_predicted.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def save_residual_plot(y_test, preds):
    residuals = y_test - preds
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    path = FIGURES_DIR / "day6_residual_distribution.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def save_feature_importance_plot(model, X_test, y_test):
    try:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )
        importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        importances.head(10).sort_values().plot(kind="barh")
        plt.title("Top Feature Importance (Permutation)")
        plt.xlabel("Importance")
        plt.tight_layout()
        path = FIGURES_DIR / "day6_feature_importance.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")
        return importances
    except Exception as e:
        print(f"Could not compute permutation importance: {e}")
        return None


def build_strategy_recommendations(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    working = df.copy().reset_index(drop=True)
    working["PredictedNextLapTime"] = preds
    recs = []

    for _, row in working.iterrows():
        driver = row.get("Driver", "UNKNOWN")
        stint = row.get("Stint_filled", np.nan)
        predicted = row.get("PredictedNextLapTime", np.nan)
        prev_lap = row.get("PreviousLapTimeSeconds", np.nan)
        rolling_mean = row.get("PreviousLapTimeRollingMean3", np.nan)

        recommendation = "Keep going"
        reason = "No strong degradation signal detected."

        long_stint = pd.notna(stint) and stint >= 8
        pace_drop = pd.notna(predicted) and pd.notna(rolling_mean) and predicted > (rolling_mean + 1.5)

        if long_stint and pace_drop:
            recommendation = "Pit soon"
            reason = "Predicted next lap is slower than recent pace and stint is long."
        elif pd.notna(predicted) and pd.notna(prev_lap) and predicted > (prev_lap + 1.2):
            recommendation = "Consider pit window"
            reason = "Predicted next lap is noticeably slower than the previous lap."

        recs.append(
            {
                "Driver": driver,
                "LapNumber": row.get("LapNumber", np.nan),
                "Stint": stint,
                "PredictedNextLapTime": predicted,
                "PreviousLapTimeSeconds": prev_lap,
                "PreviousLapTimeRollingMean3": rolling_mean,
                "Recommendation": recommendation,
                "Reason": reason,
            }
        )

    return pd.DataFrame(recs)


def main():
    print("## Day 6 — Model Evaluation and Strategy\n")

    df = load_feature_data()
    print(f"Loaded dataset shape: {df.shape}")

    X, y, df_filtered = prepare_dataset(df)

    print("\nFeature columns being used:")
    print(X.columns.tolist())

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y,
        df_filtered.loc[X.index],
        test_size=0.2,
        random_state=42,
    )

    model_path = MODELS_DIR / "day5_best_next_lap_time_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run Day 5 first.")

    model = joblib.load(model_path)
    print(f"\nLoaded model: {model_path.name}")

    preds, metrics = evaluate_model(model, X_test, y_test)

    print("\nEvaluation metrics:")
    print(json.dumps(metrics, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODELS_DIR / "day6_evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    save_scatter_plot(y_test, preds)
    save_residual_plot(y_test, preds)

    importance = save_feature_importance_plot(model, X_test, y_test)
    if importance is not None:
        print("\nTop features:")
        print(importance.head(10))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    strategy_df = build_strategy_recommendations(df_test.reset_index(drop=True), preds)
    strategy_path = REPORTS_DIR / "day6_pit_recommendations.csv"
    strategy_df.to_csv(strategy_path, index=False)
    print(f"\nSaved pit recommendations to: {strategy_path}")

    print("\nSample recommendations:")
    print(strategy_df[["Driver", "LapNumber", "Recommendation", "Reason"]].head(10))

    plt.close("all")
    print("\n## Day 6 complete")


if __name__ == "__main__":
    main()
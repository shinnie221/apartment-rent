"""K-Nearest Neighbours regression model with Pipeline and GridSearchCV."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .pipeline import FeatureTransformer, prepare_data_and_split
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.pipeline import FeatureTransformer, prepare_data_and_split


def train_model(
    df: pd.DataFrame | None = None,
    X_train_trans: pd.DataFrame | None = None,
    X_test_trans: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    y_test: pd.Series | None = None,
    transformer: FeatureTransformer | None = None,
):
    """
    Train KNN Regressor using an sklearn Pipeline (StandardScaler + KNeighborsRegressor)
    and GridSearchCV so scaling is fitted independently within each CV fold.
    Refits best pipeline on full training data and evaluates on the FULL common test set.
    """
    if X_train_trans is None or X_test_trans is None or y_train is None or y_test is None:
        if df is None:
            raise ValueError("Must provide either pre-split data or df.")
        X_train, X_test, y_train, y_test, transformer, _ = prepare_data_and_split(df)
        X_train_trans = transformer.transform(X_train)
        X_test_trans = transformer.transform(X_test)

    feature_cols = X_train_trans.columns.tolist()
    y_train_log = np.log1p(y_train)

    # ── Pipeline with StandardScaler & KNN ──
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(algorithm="auto")),
    ])

    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    # Subsample training data for fast CV search if training set is large
    if len(X_train_trans) > 15000:
        np.random.seed(42)
        cv_idx = np.random.choice(len(X_train_trans), size=15000, replace=False)
        X_cv = X_train_trans.iloc[cv_idx]
        y_cv = y_train_log.iloc[cv_idx]
    else:
        X_cv = X_train_trans
        y_cv = y_train_log

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X_cv, y_cv)

    # Extract clean best params without 'knn__' prefix for reporting
    best_params = {k.replace("knn__", ""): v for k, v in grid_search.best_params_.items()}
    print(f"[KNN Regressor] Best Hyperparameters: {best_params}")

    # Refit final best pipeline on FULL training set
    best_pipe = grid_search.best_estimator_
    best_pipe.fit(X_train_trans, y_train_log)

    best_pipe.best_params_ = best_params
    best_pipe.cv_results_summary_ = grid_search.cv_results_

    # Predict on the FULL common test set
    y_pred_log = best_pipe.predict(X_test_trans)
    y_pred = np.maximum(0.0, np.expm1(y_pred_log))

    # Regression metrics on full test set
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print(f"[KNN Regressor] Evaluated on {len(y_test):,} test observations.")
    print(f"[KNN Regressor] R²: {r2:.4f}, MAE: ${mae:,.2f}, RMSE: ${rmse:,.2f}")

    return best_pipe, None, feature_cols, mae, rmse, r2, y_test, y_pred


def predict_property(
    model: Pipeline | KNeighborsRegressor,
    scaler_or_transformer: FeatureTransformer | StandardScaler | None,
    feature_names_or_scaler: list[str] | StandardScaler | None,
    input_data: dict,
) -> float:
    """Predict monthly rent in USD for a single listing using the KNN pipeline."""
    input_df = pd.DataFrame([input_data])

    if isinstance(scaler_or_transformer, FeatureTransformer):
        transformer = scaler_or_transformer
        X_in_trans = transformer.transform(input_df)
    else:
        feature_names = feature_names_or_scaler
        X_in_trans = pd.DataFrame(0.0, index=np.arange(1), columns=feature_names)
        for col in feature_names or []:
            if col in input_df.columns:
                X_in_trans[col] = float(pd.to_numeric(input_df[col].iloc[0], errors="coerce") or 0.0)

    predicted_rent_log = model.predict(X_in_trans)[0]
    predicted_rent = np.expm1(predicted_rent_log)

    return max(0.0, float(predicted_rent))


if __name__ == "__main__":
    from pathlib import Path
    import os

    print("=" * 60)
    print("RUNNING KNN MODEL TRAINING & HYPERPARAMETER SEARCH")
    print("=" * 60)

    # Locate dataset file
    possible_paths = [
        Path("apartments_for_rent_fully_prepared.csv"),
        Path("../apartments_for_rent_fully_prepared.csv"),
        Path(__file__).resolve().parent.parent / "apartments_for_rent_fully_prepared.csv",
    ]
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        raise FileNotFoundError("Could not find 'apartments_for_rent_fully_prepared.csv'. Please ensure it exists in the project root.")

    print(f"Loading data from: {data_path.name}")
    df = pd.read_csv(data_path)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    best_model, _, feature_names, mae, rmse, r2, y_test, y_pred = train_model(df=df)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 60)
    print("Best Hyperparameters found by GridSearchCV:")
    for param, val in best_model.best_params_.items():
        print(f"  - {param}: {val}")
    print("=" * 60)

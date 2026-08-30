"""Random Forest regression model with RandomizedSearchCV and shared test evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

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
    Train Random Forest Regressor using RandomizedSearchCV on training data.
    Fits the best estimator on the full training set and evaluates on the shared test set.
    """
    if X_train_trans is None or X_test_trans is None or y_train is None or y_test is None:
        if df is None:
            raise ValueError("Must provide either pre-split data or df.")
        X_train, X_test, y_train, y_test, transformer, _ = prepare_data_and_split(df)
        X_train_trans = transformer.transform(X_train)
        X_test_trans = transformer.transform(X_test)

    feature_cols = X_train_trans.columns.tolist()
    y_train_log = np.log1p(y_train)

    # ── RandomizedSearchCV for Random Forest ──
    # Subsample training data for fast CV iterations if training set is large
    if len(X_train_trans) > 20000:
        np.random.seed(42)
        cv_idx = np.random.choice(len(X_train_trans), size=20000, replace=False)
        X_cv = X_train_trans.iloc[cv_idx]
        y_cv = y_train_log.iloc[cv_idx]
    else:
        X_cv = X_train_trans
        y_cv = y_train_log

    param_distributions = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_leaf": [1, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
        "max_samples": [0.6, 0.8, 1.0],
    }

    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        return_train_score=True,
    )
    random_search.fit(X_cv, y_cv)

    best_params = random_search.best_params_
    print(f"[Random Forest] Best Hyperparameters: {best_params}")

    # Refit final Random Forest model on FULL training set
    model = RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_trans, y_train_log)

    model.best_params_ = best_params
    model.cv_results_summary_ = random_search.cv_results_

    # Predict on the full common test set and reverse log transform
    y_pred_log = model.predict(X_test_trans)
    y_pred = np.maximum(0.0, np.expm1(y_pred_log))

    # Regression metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print(f"[Random Forest] Evaluated on {len(y_test):,} test observations.")
    print(f"[Random Forest] R²: {r2:.4f}, MAE: ${mae:,.2f}, RMSE: ${rmse:,.2f}")

    return model, None, feature_cols, mae, rmse, r2, y_test, y_pred


def predict_property(
    model: RandomForestRegressor,
    scaler_or_transformer: FeatureTransformer | None,
    feature_names_or_none: list[str] | None,
    input_data: dict,
) -> float:
    """Predict monthly rent in USD for a single listing using the Random Forest model."""
    input_df = pd.DataFrame([input_data])
    if isinstance(scaler_or_transformer, FeatureTransformer):
        X_in_trans = scaler_or_transformer.transform(input_df)
    else:
        feature_names = feature_names_or_none
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
    print("RUNNING RANDOM FOREST MODEL TRAINING & HYPERPARAMETER SEARCH")
    print("=" * 60)

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
    print("RANDOM FOREST HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 60)
    print("Best Hyperparameters found by RandomizedSearchCV:")
    for param, val in best_model.best_params_.items():
        print(f"  - {param}: {val}")
    print("=" * 60)

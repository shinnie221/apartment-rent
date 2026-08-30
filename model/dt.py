"""Decision Tree regression model integrated into the shared evaluation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from .pipeline import FeatureTransformer, prepare_data_and_split


def _metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def train_model(
    df: pd.DataFrame | None = None,
    X_train_trans: pd.DataFrame | None = None,
    X_test_trans: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    y_test: pd.Series | None = None,
    transformer: FeatureTransformer | None = None,
    max_depth: int | None = None,
    max_leaf_nodes: int | None = None,
    min_samples_leaf: int = 10,
):
    """
    Train a controlled single Decision Tree model on the common split.
    Uses training-only validation split for candidate hyperparameter selection,
    refits on the full training set, and evaluates on the shared test set.
    """
    if X_train_trans is None or X_test_trans is None or y_train is None or y_test is None:
        if df is None:
            raise ValueError("Must provide either pre-split data or df.")
        X_train, X_test, y_train, y_test, transformer, _ = prepare_data_and_split(df)
        X_train_trans = transformer.transform(X_train)
        X_test_trans = transformer.transform(X_test)

    feature_cols = X_train_trans.columns.tolist()

    # Hyperparameter selection using only a validation subset of training data
    y_bins_train = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_trans, y_train, test_size=0.20, random_state=42, stratify=y_bins_train
    )

    candidates = [
        {"name": "Constrained tree (depth=20, leaves=512)", "max_depth": 20, "max_leaf_nodes": 512, "min_samples_leaf": 1},
        {"name": "Unpruned tree", "max_depth": None, "max_leaf_nodes": None, "min_samples_leaf": 1},
        {"name": "Min 5 per leaf", "max_depth": None, "max_leaf_nodes": None, "min_samples_leaf": 5},
        {"name": "Min 10 per leaf (depth=15)", "max_depth": 15, "max_leaf_nodes": None, "min_samples_leaf": 10},
        {"name": "Min 20 per leaf", "max_depth": None, "max_leaf_nodes": None, "min_samples_leaf": 20},
    ]

    selection_results = []
    for params in candidates:
        candidate = DecisionTreeRegressor(
            criterion="squared_error",
            random_state=42,
            max_depth=params["max_depth"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"],
        )
        candidate.fit(X_fit, y_fit)
        val_pred = candidate.predict(X_val)
        selection_results.append({**params, **_metrics(y_val, val_pred)})

    # Select best candidate configuration by validation R2
    best_candidate = max(selection_results, key=lambda r: r["r2"])

    if (max_depth, max_leaf_nodes, min_samples_leaf) != (None, None, 10):
        selected_params = {
            "name": "User-specified tree",
            "max_depth": max_depth,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
        }
    else:
        selected_params = best_candidate

    # Refit final Decision Tree on FULL training set
    model = DecisionTreeRegressor(
        criterion="squared_error",
        random_state=42,
        max_depth=selected_params["max_depth"],
        max_leaf_nodes=selected_params["max_leaf_nodes"],
        min_samples_leaf=selected_params["min_samples_leaf"],
    )
    model.fit(X_train_trans, y_train)

    train_pred = np.maximum(0.0, model.predict(X_train_trans))
    test_pred = np.maximum(0.0, model.predict(X_test_trans))

    train_r2 = float(r2_score(y_train, train_pred))
    test_metrics = _metrics(y_test, test_pred)
    r2_gap = float(train_r2 - test_metrics["r2"])

    # Store audit information for reporting
    model.training_r2_ = train_r2
    model.testing_r2_ = test_metrics["r2"]
    model.r2_gap_ = r2_gap
    model.best_params_ = {
        "max_depth": selected_params["max_depth"],
        "max_leaf_nodes": selected_params["max_leaf_nodes"],
        "min_samples_leaf": selected_params["min_samples_leaf"],
    }
    model.selected_parameters_ = selected_params
    model.selection_results_ = selection_results

    print(f"[Decision Tree] Best Hyperparameters: {model.best_params_}")
    print(f"[Decision Tree] Train R²: {train_r2:.4f}, Test R²: {test_metrics['r2']:.4f}, R² Gap: {r2_gap:.4f}")
    print(f"[Decision Tree] Test MAE: ${test_metrics['mae']:,.2f}, Test RMSE: ${test_metrics['rmse']:,.2f}")

    return (
        model,
        None,  # No scaler required for tree models
        feature_cols,
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["r2"],
        y_test,
        test_pred,
    )


def predict_property(
    model: DecisionTreeRegressor,
    scaler_or_transformer: FeatureTransformer | None,
    feature_names_or_none: list[str] | None,
    input_data: dict,
) -> float:
    """Predict monthly rent in USD for a single listing using the Decision Tree model."""
    input_df = pd.DataFrame([input_data])
    if isinstance(scaler_or_transformer, FeatureTransformer):
        X_in_trans = scaler_or_transformer.transform(input_df)
    else:
        feature_names = feature_names_or_none
        X_in_trans = pd.DataFrame(0.0, index=np.arange(1), columns=feature_names)
        for col in feature_names or []:
            if col in input_df.columns:
                X_in_trans[col] = float(pd.to_numeric(input_df[col].iloc[0], errors="coerce") or 0.0)

    predicted_rent = float(model.predict(X_in_trans)[0])
    return max(0.0, predicted_rent)


if __name__ == "__main__":
    from pathlib import Path
    import os

    print("=" * 60)
    print("RUNNING DECISION TREE MODEL TRAINING & HYPERPARAMETER SEARCH")
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
    print("DECISION TREE HYPERPARAMETER SELECTION RESULTS")
    print("=" * 60)
    for res in best_model.selection_results_:
        print(f"• Candidate: {res['name']:<45} | Val R²: {res['r2']:.4f} | Val MAE: ${res['mae']:,.2f}")
    print("-" * 60)
    print("Best Hyperparameters selected:")
    for param, val in best_model.best_params_.items():
        print(f"  - {param}: {val}")
    print("=" * 60)
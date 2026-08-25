"""Linear Regression baseline model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .pipeline import FeatureTransformer, prepare_data_and_split


def train_model(
    df: pd.DataFrame | None = None,
    X_train_trans: pd.DataFrame | None = None,
    X_test_trans: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    y_test: pd.Series | None = None,
    transformer: FeatureTransformer | None = None,
):
    """
    Train a true baseline Linear Regression model without GridSearchCV.
    Fits StandardScaler strictly on training features.
    """
    if X_train_trans is None or X_test_trans is None or y_train is None or y_test is None:
        if df is None:
            raise ValueError("Must provide either pre-split data or df.")
        X_train, X_test, y_train, y_test, transformer, _ = prepare_data_and_split(df)
        X_train_trans = transformer.transform(X_train)
        X_test_trans = transformer.transform(X_test)

    feature_cols = X_train_trans.columns.tolist()

    # Scale features using training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_trans)
    X_test_scaled = scaler.transform(X_test_trans)

    # Log target for training to reduce target skewness
    y_train_log = np.log1p(y_train)

    # Baseline Linear Regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_log)

    best_params = "Baseline model — no hyperparameter tuning"
    model.best_params_ = best_params

    # Predict on test set and return to original USD scale
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.maximum(0.0, np.expm1(y_pred_log))

    # Regression metrics on full test set
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print(f"[Linear Regression] Evaluated on {len(y_test):,} test observations.")
    print(f"[Linear Regression] R²: {r2:.4f}, MAE: ${mae:,.2f}, RMSE: ${rmse:,.2f}")

    return model, scaler, feature_cols, mae, rmse, r2, y_test, y_pred


def predict_property(
    model: LinearRegression,
    scaler_or_transformer: StandardScaler | FeatureTransformer,
    feature_names_or_scaler: list[str] | StandardScaler | None,
    input_data: dict,
) -> float:
    """
    Predict monthly rent in USD for a single apartment listing.
    Supports flexible signature for compatibility with main.py.
    """
    input_df = pd.DataFrame([input_data])

    if isinstance(scaler_or_transformer, FeatureTransformer):
        transformer = scaler_or_transformer
        scaler = feature_names_or_scaler
        X_in_trans = transformer.transform(input_df)
        X_in_scaled = scaler.transform(X_in_trans) if scaler is not None else X_in_trans
    else:
        # Fallback if scaler and feature_names are passed
        scaler = scaler_or_transformer
        feature_names = feature_names_or_scaler
        X_in = pd.DataFrame(0.0, index=np.arange(1), columns=feature_names)
        for col in feature_names:
            if col in input_df.columns:
                X_in[col] = float(pd.to_numeric(input_df[col].iloc[0], errors="coerce") or 0.0)
        X_in_scaled = scaler.transform(X_in)

    predicted_rent_log = model.predict(X_in_scaled)[0]
    predicted_rent = np.expm1(predicted_rent_log)

    return max(0.0, float(predicted_rent))

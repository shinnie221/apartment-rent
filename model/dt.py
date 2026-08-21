"""Decision Tree regression model with training-only feature preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "pets_allowed_bin",
    "amenities_count",
    "square_feet",
    "log_sqft",
    "total_rooms",
    "latitude",
    "longitude",
    "sqft_per_room",
    "bath_bed_ratio",
    "city_frequency",
]


class DecisionTreeFeatureTransformer:
    """Create non-target-derived features and one-hot encode state."""

    def fit(self, frame: pd.DataFrame) -> "DecisionTreeFeatureTransformer":
        city = frame.get("cityname", pd.Series("Unknown", index=frame.index))
        city = city.fillna("Unknown").astype(str)
        self.city_frequencies_ = city.value_counts(normalize=True).to_dict()

        state = frame.get("state", pd.Series("Unknown", index=frame.index))
        state = state.fillna("Unknown").astype(str)
        self.state_categories_ = sorted(state.unique().tolist())
        self.feature_names_ = NUMERIC_FEATURES + [
            f"state_{category}" for category in self.state_categories_
        ]
        return self

    @staticmethod
    def _numeric(
        frame: pd.DataFrame,
        column: str,
        default: float,
    ) -> pd.Series:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
        else:
            values = pd.Series(default, index=frame.index, dtype=float)
        return values.fillna(default).astype(float)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("DecisionTreeFeatureTransformer must be fitted first")

        bedrooms = self._numeric(frame, "bedrooms", 2.0)
        bathrooms = self._numeric(frame, "bathrooms", 1.0)
        square_feet = self._numeric(frame, "square_feet", 900.0)

        transformed = pd.DataFrame(index=frame.index)
        transformed["bedrooms"] = bedrooms
        transformed["bathrooms"] = bathrooms
        transformed["pets_allowed_bin"] = self._numeric(
            frame, "pets_allowed_bin", 0.0
        )
        transformed["amenities_count"] = self._numeric(
            frame, "amenities_count", 0.0
        )
        transformed["square_feet"] = square_feet
        transformed["log_sqft"] = np.log1p(square_feet.clip(lower=0))
        transformed["total_rooms"] = bedrooms + bathrooms
        transformed["latitude"] = self._numeric(frame, "latitude", 39.8163)
        transformed["longitude"] = self._numeric(frame, "longitude", -98.5576)
        transformed["sqft_per_room"] = square_feet / (
            transformed["total_rooms"] + 1.0
        )
        transformed["bath_bed_ratio"] = bathrooms / (bedrooms + 1.0)

        city = frame.get("cityname", pd.Series("Unknown", index=frame.index))
        city = city.fillna("Unknown").astype(str)
        transformed["city_frequency"] = (
            city.map(self.city_frequencies_).fillna(0.0).astype(float)
        )

        state = frame.get("state", pd.Series("Unknown", index=frame.index))
        state = state.fillna("Unknown").astype(str)
        for category in self.state_categories_:
            transformed[f"state_{category}"] = (state == category).astype(float)

        return transformed.reindex(columns=self.feature_names_, fill_value=0.0)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)


def _price_bins(values: pd.Series) -> pd.Series:
    """Temporary price quantiles used only to preserve the split distribution."""
    return pd.qcut(values, q=10, labels=False, duplicates="drop")


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def train_model(
    df: pd.DataFrame,
    max_depth: int | None = None,
    max_leaf_nodes: int | None = None,
    min_samples_leaf: int = 10,
):
    """Train a controlled single tree while keeping the application API stable."""
    model_df = df.copy()
    model_df["price"] = pd.to_numeric(model_df["price"], errors="coerce")
    model_df = model_df.dropna(subset=["price"])

    # Apply the same model-stage anomaly rule used by the project models.
    outlier_threshold = float(model_df["price"].quantile(0.995))
    model_df = model_df[model_df["price"] <= outlier_threshold].copy()

    source_features = [
        "bedrooms",
        "bathrooms",
        "square_feet",
        "amenities_count",
        "pets_allowed_bin",
        "state",
        "cityname",
        "latitude",
        "longitude",
    ]
    X = model_df[source_features].copy()
    y = model_df["price"].astype(float).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=_price_bins(y),
    )

    # Hyperparameter refinement uses only a validation subset of training data.
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=_price_bins(y_train),
    )
    tuning_transformer = DecisionTreeFeatureTransformer()
    X_fit_transformed = tuning_transformer.fit_transform(X_fit)
    X_validation_transformed = tuning_transformer.transform(X_validation)

    candidates = [
        {
            "name": "Initial constrained tree",
            "max_depth": 20,
            "max_leaf_nodes": 512,
            "min_samples_leaf": 1,
        },
        {
            "name": "Unpruned tree",
            "max_depth": None,
            "max_leaf_nodes": None,
            "min_samples_leaf": 1,
        },
        {
            "name": "Minimum 5 per leaf",
            "max_depth": None,
            "max_leaf_nodes": None,
            "min_samples_leaf": 5,
        },
        {
            "name": "Minimum 10 per leaf",
            "max_depth": None,
            "max_leaf_nodes": None,
            "min_samples_leaf": 10,
        },
        {
            "name": "Minimum 20 per leaf",
            "max_depth": None,
            "max_leaf_nodes": None,
            "min_samples_leaf": 20,
        },
    ]

    selection_results = []
    for parameters in candidates:
        candidate = DecisionTreeRegressor(
            criterion="squared_error",
            random_state=42,
            max_depth=parameters["max_depth"],
            max_leaf_nodes=parameters["max_leaf_nodes"],
            min_samples_leaf=parameters["min_samples_leaf"],
        )
        candidate.fit(X_fit_transformed, y_fit)
        validation_prediction = candidate.predict(X_validation_transformed)
        result = {**parameters, **_metrics(y_validation, validation_prediction)}
        selection_results.append(result)

    selected = max(selection_results, key=lambda result: result["r2"])
    # Explicit arguments allow controlled experiments while defaults use validation choice.
    if (max_depth, max_leaf_nodes, min_samples_leaf) != (None, None, 10):
        selected = {
            "name": "User-specified tree",
            "max_depth": max_depth,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
        }

    transformer = DecisionTreeFeatureTransformer()
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    baseline_parameters = candidates[0]
    baseline_model = DecisionTreeRegressor(
        criterion="squared_error",
        random_state=42,
        max_depth=baseline_parameters["max_depth"],
        max_leaf_nodes=baseline_parameters["max_leaf_nodes"],
        min_samples_leaf=baseline_parameters["min_samples_leaf"],
    )
    baseline_model.fit(X_train_transformed, y_train)
    baseline_metrics = _metrics(y_test, baseline_model.predict(X_test_transformed))

    model = DecisionTreeRegressor(
        criterion="squared_error",
        random_state=42,
        max_depth=selected["max_depth"],
        max_leaf_nodes=selected["max_leaf_nodes"],
        min_samples_leaf=selected["min_samples_leaf"],
    )
    model.fit(X_train_transformed, y_train)

    training_prediction = np.maximum(0.0, model.predict(X_train_transformed))
    testing_prediction = np.maximum(0.0, model.predict(X_test_transformed))
    test_metrics = _metrics(y_test, testing_prediction)

    # Store audit information needed by the report runtime and tree diagram.
    model.training_r2_ = float(r2_score(y_train, training_prediction))
    model.selection_results_ = selection_results
    model.selected_parameters_ = selected
    model.validation_r2_ = float(selected.get("r2", np.nan))
    model.baseline_metrics_ = baseline_metrics
    model.outlier_threshold_ = outlier_threshold
    model.prepared_records_ = int(len(df))
    model.model_records_ = int(len(model_df))
    model.training_records_ = int(len(X_train))
    model.testing_records_ = int(len(X_test))
    model.validation_records_ = int(len(X_validation))
    model.feature_count_ = int(X_train_transformed.shape[1])
    model.feature_names_report_ = transformer.get_feature_names_out().tolist()
    model.quartile_edges_ = np.quantile(y_train, [0.25, 0.50, 0.75])
    training_tiers = np.digitize(y_train, model.quartile_edges_, right=True)
    model.quartile_centers_ = np.asarray(
        [float(y_train[training_tiers == tier].median()) for tier in range(4)]
    )

    return (
        model,
        transformer,
        model.feature_names_report_,
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["r2"],
        y_test,
        testing_prediction,
    )


def predict_property(model, scaler, feature_names, input_data):
    """Predict monthly rent from the application input dictionary."""
    del feature_names  # Retained in the signature for compatibility with main.py.
    input_frame = pd.DataFrame([input_data])
    transformed = scaler.transform(input_frame)
    predicted_rent = float(model.predict(transformed)[0])
    return max(0.0, predicted_rent)
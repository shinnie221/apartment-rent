"""Data preprocessing, feature engineering, and shared dataset splitting pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_NUMERIC_FEATURES = [
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


class FeatureTransformer:
    """Feature transformer fitted STRICTLY on training data to avoid data leakage."""

    def __init__(self):
        self.city_frequencies_: dict[str, float] = {}
        self.state_categories_: list[str] = []
        self.feature_names_: list[str] = []
        self.medians_: dict[str, float] = {}
        self.state_lat_medians_: dict[str, float] = {}
        self.state_lon_medians_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> "FeatureTransformer":
        """Learn encoding mappings and imputations strictly from training frame."""
        # 1. State categories
        state_series = frame.get("state", pd.Series(dtype=str)).dropna().astype(str)
        self.state_categories_ = sorted([s for s in state_series.unique().tolist() if s != "Unknown"])

        # 2. City frequencies from training data only
        city_series = frame.get("cityname", pd.Series(dtype=str)).fillna("Unknown").astype(str)
        self.city_frequencies_ = city_series.value_counts(normalize=True).to_dict()

        # 3. Numeric medians from training data only (no global dataset leakage)
        beds_s = pd.to_numeric(frame.get("bedrooms"), errors="coerce").dropna()
        self.medians_["bedrooms"] = float(beds_s.median()) if len(beds_s) > 0 else 2.0

        baths_s = pd.to_numeric(frame.get("bathrooms"), errors="coerce").dropna()
        self.medians_["bathrooms"] = float(baths_s.median()) if len(baths_s) > 0 else 1.0

        sqft_s = pd.to_numeric(frame.get("square_feet"), errors="coerce").dropna()
        self.medians_["square_feet"] = float(sqft_s.median()) if len(sqft_s) > 0 else 1000.0

        lat_s = pd.to_numeric(frame.get("latitude"), errors="coerce").dropna()
        self.medians_["latitude"] = float(lat_s.median()) if len(lat_s) > 0 else 37.0

        lon_s = pd.to_numeric(frame.get("longitude"), errors="coerce").dropna()
        self.medians_["longitude"] = float(lon_s.median()) if len(lon_s) > 0 else -95.0

        # State-level coordinate medians from training data
        if "state" in frame.columns and "latitude" in frame.columns:
            valid_lat = frame.dropna(subset=["state", "latitude"])
            self.state_lat_medians_ = valid_lat.groupby("state")["latitude"].median().to_dict()
        if "state" in frame.columns and "longitude" in frame.columns:
            valid_lon = frame.dropna(subset=["state", "longitude"])
            self.state_lon_medians_ = valid_lon.groupby("state")["longitude"].median().to_dict()

        # Define full feature names
        self.feature_names_ = list(BASE_NUMERIC_FEATURES) + [
            f"state_{cat}" for cat in self.state_categories_
        ]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform features using pre-fitted training statistics."""
        if not self.feature_names_:
            raise RuntimeError("FeatureTransformer must be fitted on training data first.")

        df = frame.copy()

        # Extract numeric inputs, imputing with training-only medians
        beds = pd.to_numeric(df.get("bedrooms"), errors="coerce").fillna(self.medians_.get("bedrooms", 2.0)).astype(float)
        baths = pd.to_numeric(df.get("bathrooms"), errors="coerce").fillna(self.medians_.get("bathrooms", 1.0)).astype(float)
        sqft = pd.to_numeric(df.get("square_feet"), errors="coerce").fillna(self.medians_.get("square_feet", 1000.0)).astype(float)
        pets = pd.to_numeric(df.get("pets_allowed_bin"), errors="coerce").fillna(0.0).astype(float)
        amen = pd.to_numeric(df.get("amenities_count"), errors="coerce").fillna(0.0).astype(float)

        # Coordinate handling: impute using training state medians first, then global training median
        raw_lat = pd.to_numeric(df.get("latitude"), errors="coerce")
        raw_lon = pd.to_numeric(df.get("longitude"), errors="coerce")
        state_col = df.get("state", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)

        fallback_state_lat = state_col.map(self.state_lat_medians_).fillna(self.medians_.get("latitude", 37.0))
        fallback_state_lon = state_col.map(self.state_lon_medians_).fillna(self.medians_.get("longitude", -95.0))

        lat = raw_lat.fillna(fallback_state_lat).fillna(self.medians_.get("latitude", 37.0)).astype(float)
        lon = raw_lon.fillna(fallback_state_lon).fillna(self.medians_.get("longitude", -95.0)).astype(float)

        # Output frame
        out = pd.DataFrame(index=df.index)
        out["bedrooms"] = beds
        out["bathrooms"] = baths
        out["pets_allowed_bin"] = pets
        out["amenities_count"] = amen
        out["square_feet"] = sqft
        out["log_sqft"] = np.log1p(sqft.clip(lower=0.0))
        out["total_rooms"] = beds + baths
        out["latitude"] = lat
        out["longitude"] = lon
        out["sqft_per_room"] = sqft / (out["total_rooms"] + 1.0)
        out["bath_bed_ratio"] = baths / (beds + 1.0)

        # City frequency: 0.0 for unseen cities
        city = df.get("cityname", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
        out["city_frequency"] = city.map(self.city_frequencies_).fillna(0.0).astype(float)

        # State one-hot dummy variables: 0.0 for unseen states
        for cat in self.state_categories_:
            out[f"state_{cat}"] = (state_col == cat).astype(float)

        return out.reindex(columns=self.feature_names_, fill_value=0.0)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)


def prepare_data_and_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, FeatureTransformer, dict]:
    """
    1. Drop missing target prices (never impute price).
    2. Filter out price outliers (> 99.5th percentile).
    3. Split into 80% train / 20% test using stratified price bins with random_state=42.
    4. Fit FeatureTransformer strictly on training set.
    """
    df_clean = df.copy()
    if "price" in df_clean.columns:
        df_clean["price"] = pd.to_numeric(df_clean["price"], errors="coerce")
        df_clean = df_clean.dropna(subset=["price"])

    original_records = len(df_clean)
    q995_threshold = float(df_clean["price"].quantile(0.995)) if original_records > 0 else 5000.0
    df_filtered = df_clean[df_clean["price"] <= q995_threshold].copy()
    modelling_records = len(df_filtered)
    removed_records = original_records - modelling_records

    print("=" * 60)
    print("DATASET PREPARATION & OUTLIER FILTERING (99.5th PERCENTILE)")
    print("=" * 60)
    print(f"Original number of records:            {original_records:,}")
    print(f"99.5th percentile price threshold:     ${q995_threshold:,.2f}")
    print(f"Number of outlier records removed:     {removed_records:,}")
    print(f"Final number of modelling records:     {modelling_records:,}")

    # Separate target
    y = df_filtered["price"].astype(float)
    X = df_filtered.drop(columns=["price"])

    # Common stratified train-test split (80/20, random_state=42)
    stratify_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify_bins
    )

    print(f"Common Training records (80%):         {len(X_train):,}")
    print(f"Common Testing records (20%):          {len(X_test):,}")
    print("=" * 60)

    # Fit FeatureTransformer on TRAINING data only
    transformer = FeatureTransformer()
    transformer.fit(X_train)

    outlier_info = {
        "original_records": original_records,
        "q995_threshold": q995_threshold,
        "removed_records": removed_records,
        "modelling_records": modelling_records,
        "train_records": len(X_train),
        "test_records": len(X_test),
    }

    return X_train, X_test, y_train, y_test, transformer, outlier_info


def get_training_quartile_edges(y_train: pd.Series | np.ndarray) -> np.ndarray:
    """
    Compute quartile boundaries strictly from TRAINING set prices.
    Returns [-np.inf, Q1, Q2, Q3, np.inf] so no observations become NaN.
    """
    q1, q2, q3 = np.quantile(y_train, [0.25, 0.50, 0.75])
    return np.array([-np.inf, q1, q2, q3, np.inf])

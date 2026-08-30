import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# Set page config
st.set_page_config(page_title="Apartment Rental Price Prediction & Valuation System", page_icon="🏢", layout="wide")

# Import shared pipeline & models
from model.pipeline import prepare_data_and_split, get_training_quartile_edges, FeatureTransformer
from model.lr import train_model as train_lr, predict_property as predict_lr
from model.knn import train_model as train_knn, predict_property as predict_knn
from model.dt import train_model as train_dt, predict_property as predict_dt
from model.rf import train_model as train_rf, predict_property as predict_rf

# ──────────────────────────────────────────────
#  SECTION 1: LOAD RAW DATASET & MISSING VALUE ANALYSIS
# ──────────────────────────────────────────────
def analyze_raw_dataset(raw_file='apartments_for_rent_classified_100K.csv'):
    """Load raw dataset, inspect shape, and perform missing value analysis."""
    df_raw = pd.read_csv(raw_file, sep=';', encoding='latin-1', low_memory=False)

    print("=" * 60)
    print("SECTION 1: RAW DATASET MISSING VALUE ANALYSIS")
    print("=" * 60)
    print(f"Dataset Shape: {df_raw.shape[0]:,} rows and {df_raw.shape[1]} columns\n")
    print(df_raw[['id', 'category', 'title', 'price', 'bedrooms', 'bathrooms',
                   'square_feet', 'cityname', 'state']].head(3))

    total_rows = len(df_raw)
    print(f"\nTotal Rows in Raw Dataset: {total_rows:,}\n")

    # Compute missing values and percentages
    missing_count = df_raw.isnull().sum()
    missing_percent = (missing_count / total_rows) * 100
    missing_summary = pd.DataFrame({
        'Missing Count': missing_count,
        'Percentage (%)': missing_percent
    })
    missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values(
        by='Missing Count', ascending=False
    )
    for column, row in missing_summary.iterrows():
        print(f"{column}: {int(row['Missing Count']):,} missing entries ({row['Percentage (%)']:.2f}% missing)")

    return df_raw


# ──────────────────────────────────────────────
#  SECTION 2: DATA CLEANING & PREPROCESSING (WITHOUT GLOBAL NUMERICAL IMPUTATION)
# ──────────────────────────────────────────────
def clean_apartment_dataset(df_raw):
    """
    Clean raw dataset: remove duplicates, strip whitespace, clean text fields.
    Does NOT globally impute numerical modelling fields (price, bedrooms, bathrooms,
    square_feet, latitude, longitude) to avoid data leakage before splitting.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: EXECUTING PREPROCESSING PIPELINE (NO NUMERICAL LEAKAGE)")
    print("=" * 60)

    # 1. Remove exact duplicate rows
    df_clean = df_raw.drop_duplicates().copy()
    print(f"Rows after removing exact duplicates: {len(df_clean):,}")

    # 2. Strip leading/trailing whitespaces from object/string columns
    string_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 3. Handle Categorical / String Missing Values cleanly
    df_clean["amenities"] = df_clean["amenities"].fillna("None").replace(["nan", ""], "None")
    df_clean["pets_allowed"] = df_clean["pets_allowed"].fillna("None").replace(["nan", ""], "None")
    df_clean["address"] = df_clean["address"].fillna("Not Specified").replace(["nan", ""], "Not Specified")
    df_clean["cityname"] = df_clean["cityname"].fillna("Unknown").replace(["nan", ""], "Unknown")
    df_clean["state"] = df_clean["state"].fillna("Unknown").replace(["nan", ""], "Unknown")

    # Format price_display string representation without mutating numeric price
    if "price" in df_clean.columns:
        df_clean["price"] = pd.to_numeric(df_clean["price"], errors="coerce")
        df_clean["price_display"] = df_clean["price"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "$0"
        )

    # Cast numeric columns cleanly while keeping NaNs intact for training-only imputation
    for num_col in ["bedrooms", "bathrooms", "square_feet", "latitude", "longitude"]:
        if num_col in df_clean.columns:
            df_clean[num_col] = pd.to_numeric(df_clean[num_col], errors="coerce")

    print(f"Cleaned Shape: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")
    return df_clean


# ──────────────────────────────────────────────
#  SECTION 3: FEATURE ENGINEERING & PREPARATION
# ──────────────────────────────────────────────
def prepare_apartment_dataset(df_clean):
    """
    Engineer base features (amenities_count, pets_allowed_bin) and
    select only the modeling columns for the final prepared dataset.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: FEATURE ENGINEERING & PREPARATION")
    print("=" * 60)

    # Feature Engineering: Create 'amenities_count'
    def count_amenities(x):
        if pd.isna(x) or str(x).strip().lower() in ["none", "nan", ""]:
            return 0
        return len(str(x).split(","))

    df_clean["amenities_count"] = df_clean["amenities"].apply(count_amenities)

    # Feature Engineering: Create 'pets_allowed_bin'
    def has_pets(x):
        if pd.isna(x) or str(x).strip().lower() in ["none", "nan", ""]:
            return 0
        return 1

    df_clean["pets_allowed_bin"] = df_clean["pets_allowed"].apply(has_pets)

    # Feature Selection: Retain only modeling columns
    cols_to_keep = [
        "price",            # Continuous rent price
        "bedrooms",         # Structural feature
        "bathrooms",        # Structural feature
        "square_feet",      # Structural feature
        "amenities_count",  # Engineered amenity feature
        "pets_allowed_bin", # Engineered pet feature
        "state",            # Location feature
        "cityname",         # Location feature
        "latitude",         # Coordinate feature
        "longitude",        # Coordinate feature
    ]

    df_prepared = df_clean[cols_to_keep].copy()
    print(f"Final Prepared Shape: {df_prepared.shape[0]:,} rows, {df_prepared.shape[1]} columns\n")

    return df_prepared


# ──────────────────────────────────────────────
#  COMBINED PIPELINE: RAW → CLEAN → PREPARE → SAVE
# ──────────────────────────────────────────────
def run_full_pipeline(raw_file='apartments_for_rent_classified_100K.csv',
                      output_file='apartments_for_rent_fully_prepared.csv'):
    """Full pipeline: load raw data → analyze missing values → clean → feature engineer → save CSV."""
    df_raw = analyze_raw_dataset(raw_file)
    df_cleaned = clean_apartment_dataset(df_raw)
    df_prepared = prepare_apartment_dataset(df_cleaned)
    df_prepared.to_csv(output_file, index=False)
    print(f"✅ Fully prepared dataset saved to: {output_file}")
    return df_prepared


# ──────────────────────────────────────────────
#  DATA LOADING & SHARED MODEL TRAINING (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    prepared_file = "apartments_for_rent_fully_prepared.csv"
    raw_file = "apartments_for_rent_classified_100K.csv"
    try:
        if not os.path.exists(prepared_file) and os.path.exists(raw_file):
            run_full_pipeline(raw_file, prepared_file)

        df = pd.read_csv(prepared_file)
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_resource
def get_trained_models(_df):
    """
    Execute common data preparation and train all 4 models on identical train-test observations.
    """
    # 1. Drop missing target prices, filter outliers (99.5th percentile), stratified split (80/20, random_state=42)
    X_train, X_test, y_train, y_test, transformer, outlier_info = prepare_data_and_split(_df)

    # 2. Transform train and test using training-fitted transformer only (training medians applied here)
    X_train_trans = transformer.transform(X_train)
    X_test_trans = transformer.transform(X_test)

    # 3. Calculate frozen quartile boundaries from TRAINING prices only [-inf, Q1, Q2, Q3, inf]
    training_quartile_edges = get_training_quartile_edges(y_train)

    results = {}

    # ── Model 1: Linear Regression (Baseline) ──
    lr_model, lr_scaler, lr_features, lr_mae, lr_rmse, lr_r2, lr_yt, lr_yp = train_lr(
        X_train_trans=X_train_trans, X_test_trans=X_test_trans,
        y_train=y_train, y_test=y_test, transformer=transformer
    )
    results['Linear Regression'] = {
        'model': lr_model,
        'transformer': transformer,
        'scaler': lr_scaler,
        'features': lr_features,
        'predict_fn': predict_lr,
        'metrics': {'MAE': lr_mae, 'RMSE': lr_rmse, 'R²': lr_r2},
        'y_test': lr_yt,
        'y_pred': lr_yp,
        'best_params': "Baseline model — no hyperparameter tuning",
    }

    # ── Model 2: KNN Regressor (Pipeline with StandardScaler + KNN) ──
    knn_model, knn_scaler, knn_features, knn_mae, knn_rmse, knn_r2, knn_yt, knn_yp = train_knn(
        X_train_trans=X_train_trans, X_test_trans=X_test_trans,
        y_train=y_train, y_test=y_test, transformer=transformer
    )
    results['KNN Regressor'] = {
        'model': knn_model,
        'transformer': transformer,
        'scaler': knn_scaler,
        'features': knn_features,
        'predict_fn': predict_knn,
        'metrics': {'MAE': knn_mae, 'RMSE': knn_rmse, 'R²': knn_r2},
        'y_test': knn_yt,
        'y_pred': knn_yp,
        'best_params': knn_model.best_params_,
    }

    # ── Model 3: Decision Tree ──
    dt_model, dt_scaler, dt_features, dt_mae, dt_rmse, dt_r2, dt_yt, dt_yp = train_dt(
        X_train_trans=X_train_trans, X_test_trans=X_test_trans,
        y_train=y_train, y_test=y_test, transformer=transformer
    )
    results['Decision Tree'] = {
        'model': dt_model,
        'transformer': transformer,
        'scaler': dt_scaler,
        'features': dt_features,
        'predict_fn': predict_dt,
        'metrics': {'MAE': dt_mae, 'RMSE': dt_rmse, 'R²': dt_r2},
        'y_test': dt_yt,
        'y_pred': dt_yp,
        'best_params': dt_model.best_params_,
        'training_r2': getattr(dt_model, 'training_r2_', None),
        'r2_gap': getattr(dt_model, 'r2_gap_', None),
    }

    # ── Model 4: Random Forest ──
    rf_model, rf_scaler, rf_features, rf_mae, rf_rmse, rf_r2, rf_yt, rf_yp = train_rf(
        X_train_trans=X_train_trans, X_test_trans=X_test_trans,
        y_train=y_train, y_test=y_test, transformer=transformer
    )
    results['Random Forest'] = {
        'model': rf_model,
        'transformer': transformer,
        'scaler': rf_scaler,
        'features': rf_features,
        'predict_fn': predict_rf,
        'metrics': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R²': rf_r2},
        'y_test': rf_yt,
        'y_pred': rf_yp,
        'best_params': rf_model.best_params_,
    }

    meta = {
        'outlier_info': outlier_info,
        'transformer': transformer,
        'training_quartile_edges': training_quartile_edges,
        'y_train': y_train,
        'test_count': len(y_test),
        'train_count': len(y_train),
    }

    return results, meta


# ──────────────────────────────────────────────
#  HELPER: quartile‑based classification metrics
# ──────────────────────────────────────────────
def _make_labels(edges):
    """Create human‑readable labels from bin edges [-inf, Q1, Q2, Q3, inf]."""
    tag_names = ['Budget', 'Moderate', 'Premium', 'Luxury']
    labels = []
    for i in range(len(edges) - 1):
        name = tag_names[i] if i < len(tag_names) else f"Q{i+1}"
        lower_str = f"${edges[i]:,.0f}" if np.isfinite(edges[i]) else "Min"
        upper_str = f"${edges[i+1]:,.0f}" if np.isfinite(edges[i+1]) else "Max"
        labels.append(f"{name} ({lower_str}–{upper_str})")
    return labels


def price_quartile_metrics(y_test, y_pred, frozen_training_edges):
    """
    Bin actual and predicted test prices using FROZEN quartile boundaries from the TRAINING set.
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    quartile_edges = frozen_training_edges
    labels = _make_labels(quartile_edges)

    y_test_binned = pd.cut(y_test, bins=quartile_edges, labels=labels, include_lowest=True)
    y_pred_binned = pd.cut(y_pred, bins=quartile_edges, labels=labels, include_lowest=True)

    return labels, y_test_binned, y_pred_binned, quartile_edges


# ──────────────────────────────────────────────
#  LOAD DATA & TRAIN MODELS
# ──────────────────────────────────────────────
df_raw = load_data()
if df_raw.empty:
    st.stop()

with st.spinner("Training models using common train-test split... This may take a moment."):
    models_dict, pipeline_meta = get_trained_models(df_raw)

# Basic preprocessing for exploration dashboard display
df_explore = df_raw.copy()
for col in ['bedrooms', 'bathrooms']:
    df_explore[col] = pd.to_numeric(df_explore[col], errors='coerce').fillna(0)
if 'square_feet' in df_explore.columns:
    df_explore['square_feet'] = pd.to_numeric(df_explore['square_feet'], errors='coerce')
    df_explore['square_feet'] = df_explore['square_feet'].fillna(df_explore['square_feet'].median())
if 'amenities_count' not in df_explore.columns:
    df_explore['amenities_count'] = 0
if 'pets_allowed_bin' not in df_explore.columns:
    df_explore['pets_allowed_bin'] = 0

df_explore['price_per_sqft'] = np.where(
    df_explore['square_feet'] > 0,
    df_explore['price'] / df_explore['square_feet'],
    np.nan
)
df_explore['lat_lon_diff'] = df_explore['latitude'] - df_explore['longitude']

# ──────────────────────────────────────────────
#  SIDEBAR – GLOBAL FILTERS
# ──────────────────────────────────────────────
st.sidebar.header("Filters")

all_states = sorted(df_explore['state'].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("Filter by State", all_states, default=all_states)

bed_min = int(df_explore['bedrooms'].min())
bed_max = int(df_explore['bedrooms'].max())
bed_range = st.sidebar.slider(
    "Bedrooms Range", min_value=bed_min, max_value=bed_max,
    value=(bed_min, bed_max), step=1
)

bath_min = float(df_explore['bathrooms'].min())
bath_max = float(df_explore['bathrooms'].max())
bath_range = st.sidebar.slider(
    "Bathrooms Range", min_value=bath_min, max_value=bath_max,
    value=(bath_min, bath_max), step=0.5
)

pet_options = ["All", "Pet-Friendly Only", "No Pets Only"]
selected_pet = st.sidebar.radio("Pet Policy", pet_options, index=0)

all_amenity_options = ["Gym", "Pool", "Parking", "Washer/Dryer", "AC", "Balcony", "Dishwasher", "Patio/Deck", "Storage"]
selected_amenities = st.sidebar.multiselect(
    "Select Amenities",
    all_amenity_options,
    default=[],
    help="Select required amenities. Apartments with at least this many amenities will be included."
)

mask = (
    (df_explore['state'].isin(selected_states)) &
    (df_explore['bedrooms'] >= bed_range[0]) &
    (df_explore['bedrooms'] <= bed_range[1]) &
    (df_explore['bathrooms'] >= bath_range[0]) &
    (df_explore['bathrooms'] <= bath_range[1]) &
    (df_explore['amenities_count'] >= len(selected_amenities))
)
if selected_pet == "Pet-Friendly Only":
    mask = mask & (df_explore['pets_allowed_bin'] == 1)
elif selected_pet == "No Pets Only":
    mask = mask & (df_explore['pets_allowed_bin'] == 0)

df_filtered = df_explore[mask].copy()

st.sidebar.markdown(f"**Showing {len(df_filtered):,} / {len(df_explore):,} apartments**")

# ──────────────────────────────────────────────
#  TITLE & TABS
# ──────────────────────────────────────────────
st.title("Apartment Rental Price Prediction & Valuation System")

tab1, tab2, tab3, tab4 = st.tabs([
    "Visualisation Dashboard",
    "Model Comparison",
    "Data Explorer",
    "Price Predictor",
])

chart_config = {'displayModeBar': False}

# ═══════════════════════════════════════════════
#  TAB 1 — VISUALISATION DASHBOARD (17 charts)
# ═══════════════════════════════════════════════
with tab1:
    st.header("Visualisation Dashboard")

    subtab_loc, subtab_feat, subtab_layout = st.tabs([
        "Location & Geographic",
        "Single Feature vs Price",
        "Physical Layout & Floorplan",
    ])

    def make_kde_histogram(df, col, title, x_label, y_label='Frequency', nbins=50, color='#87CEEB'):
        data = df[col].dropna()
        if len(data) == 0:
            return go.Figure()

        fig = px.histogram(data, x=col, nbins=nbins,
                           labels={col: x_label, 'count': y_label},
                           color_discrete_sequence=[color])
        try:
            from scipy.stats import gaussian_kde
            kde_data = data.sample(5000, random_state=42) if len(data) > 5000 else data
            kde = gaussian_kde(kde_data)
            x_range = np.linspace(data.min(), data.max(), 200)
            kde_values = kde(x_range)
            bin_width = (data.max() - data.min()) / nbins
            kde_scaled = kde_values * len(data) * bin_width

            fig.add_trace(go.Scatter(
                x=x_range, y=kde_scaled,
                mode='lines',
                name='KDE',
                line=dict(color='#1f77b4', width=2.5),
                hovertemplate=f'{x_label}: %{{x:,.2f}}<br>KDE Density: %{{y:,.2f}}<extra></extra>'
            ))
        except Exception:
            pass

        fig.update_layout(height=450, xaxis_title=x_label, yaxis_title=y_label, showlegend=False, hoverlabel=dict(font_size=13))
        return fig

    # ── SUB‑TAB 1 — Location & Geographic Drivers ──
    with subtab_loc:
        st.subheader("Location & Geographic Drivers")

        st.markdown("##### Distribution of Apartment Prices (Histogram with KDE)")
        df_price_hist = df_filtered.dropna(subset=['price']).copy()
        q99_price = df_price_hist['price'].quantile(0.99)
        df_price_hist = df_price_hist[df_price_hist['price'] <= q99_price]
        fig1 = make_kde_histogram(df_price_hist, 'price', 'Distribution of Apartment Prices', 'Price ($)', 'Frequency')
        st.plotly_chart(fig1, use_container_width=True, config=chart_config, key="chart_1")

        st.markdown("##### Bar Chart of Average Apartment Price by State (Top 10)")
        state_counts = df_filtered['state'].value_counts()
        top_10_states = state_counts.head(10).index.tolist()
        df_top10 = df_filtered[df_filtered['state'].isin(top_10_states)]
        avg_price_state = df_top10.groupby('state', as_index=False)['price'].mean().sort_values('price', ascending=False)
        fig2 = px.bar(avg_price_state, x='state', y='price',
                      labels={'price': 'Average Price ($)', 'state': 'State'},
                      color='price', color_continuous_scale='Viridis')
        fig2.update_traces(hovertemplate='<b>State: %{x}</b><br>Average Price: $%{y:,.2f}<extra></extra>')
        fig2.update_layout(height=450, xaxis_tickangle=-45, hoverlabel=dict(font_size=13))
        st.plotly_chart(fig2, use_container_width=True, config=chart_config, key="chart_2")

        st.markdown("##### Distribution of Latitude-Longitude Difference (Histogram with KDE)")
        df_latlon = df_filtered.dropna(subset=['lat_lon_diff']).copy()
        fig3 = make_kde_histogram(df_latlon, 'lat_lon_diff', 'Distribution of Latitude-Longitude Difference',
                                  'Latitude − Longitude Difference', 'Number of Apartments', color='#87CEEB')
        st.plotly_chart(fig3, use_container_width=True, config=chart_config, key="chart_3")

        col_loc_a, col_loc_b = st.columns(2)
        with col_loc_a:
            st.markdown("##### Top 10 Cities by Number of Apartments (Line)")
            top_cities = df_filtered['cityname'].value_counts().head(10).reset_index()
            top_cities.columns = ['City', 'Count']
            fig4 = px.line(top_cities, x='City', y='Count', markers=True,
                           labels={'City': 'City Name', 'Count': 'Number of Apartments'},
                           color_discrete_sequence=['purple'])
            fig4.update_traces(hovertemplate='<b>City: %{x}</b><br>Apartments: %{y:,}<extra></extra>')
            fig4.update_layout(height=420, xaxis_tickangle=-45, hoverlabel=dict(font_size=13))
            st.plotly_chart(fig4, use_container_width=True, config=chart_config, key="chart_4")

        with col_loc_b:
            st.markdown("##### Top 10 States by Median Apartment Size (Horizontal Bar)")
            agg10 = df_filtered.groupby('state', as_index=False)['square_feet'].median() \
                .nlargest(10, 'square_feet').sort_values('square_feet', ascending=True)
            fig10 = px.bar(agg10, y='state', x='square_feet', orientation='h',
                           labels={'square_feet': 'Median Sq Ft', 'state': 'State'},
                           color='square_feet', color_continuous_scale='Peach')
            fig10.update_traces(hovertemplate='<b>State: %{y}</b><br>Median Size: %{x:,.0f} sq ft<extra></extra>')
            fig10.update_layout(height=420, yaxis={'categoryorder': 'total ascending'}, hoverlabel=dict(font_size=13))
            st.plotly_chart(fig10, use_container_width=True, config=chart_config, key="chart_10")

    # ── SUB‑TAB 2 — Single Feature vs Price Drivers ──
    with subtab_feat:
        st.subheader("Single Feature vs Price Drivers")

        st.markdown("##### Distribution of Rental Price by Pets Allowed Category (Grouped Histogram)")
        df_p6 = df_filtered.dropna(subset=['price']).copy()
        q99_p6 = df_p6['price'].quantile(0.99)
        df_p6 = df_p6[df_p6['price'] <= q99_p6].copy()
        df_p6['Pet Policy'] = df_p6['pets_allowed_bin'].map({0: 'No Pets', 1: 'Pet-Friendly'})
        fig6 = px.histogram(df_p6, x='price', color='Pet Policy', barmode='group', nbins=50,
                            labels={'price': 'Rental Price ($)', 'count': 'Count'},
                            color_discrete_sequence=['#EF553B', '#00CC96'])
        fig6.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Price Range: %{x}<br>Count: %{y:,}<extra></extra>')
        fig6.update_layout(height=450, xaxis_title='Rental Price ($)', yaxis_title='Count', hoverlabel=dict(font_size=13))
        st.plotly_chart(fig6, use_container_width=True, config=chart_config, key="chart_6")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("##### Bedrooms vs Price (Violin Plot)")
            df_v7 = df_filtered[df_filtered['bedrooms'] > 0].copy()
            df_v7['bedrooms_str'] = df_v7['bedrooms'].astype(int).astype(str) + ' Bed'
            fig7 = px.violin(df_v7, x='bedrooms_str', y='price', box=True,
                             labels={'bedrooms_str': 'Bedrooms', 'price': 'Price ($)'},
                             color_discrete_sequence=['#636EFA'])
            fig7.update_layout(height=420, hoverlabel=dict(font_size=13))
            st.plotly_chart(fig7, use_container_width=True, config=chart_config, key="chart_7")

        with col_f2:
            st.markdown("##### Bathrooms vs Price Distribution (Box Plot)")
            fig8 = px.box(df_filtered, x='bathrooms', y='price',
                          labels={'price': 'Price ($)', 'bathrooms': 'Bathrooms'},
                          color_discrete_sequence=['#EF553B'])
            fig8.update_layout(height=420, hoverlabel=dict(font_size=13))
            st.plotly_chart(fig8, use_container_width=True, config=chart_config, key="chart_8")

        col_f3, col_f4 = st.columns(2)
        with col_f3:
            st.markdown("##### Size Tiers vs Average Rent (Area Graph)")
            df_bin9 = df_filtered.dropna(subset=['square_feet']).copy()
            df_bin9['sqft_tier'] = pd.cut(df_bin9['square_feet'],
                                          bins=[0, 500, 800, 1200, 1800, 100000],
                                          labels=['<500', '500–800', '800–1200', '1200–1800', '1800+'])
            agg9 = df_bin9.groupby('sqft_tier', as_index=False, observed=True)['price'].mean()
            fig9 = px.area(agg9, x='sqft_tier', y='price',
                           labels={'price': 'Avg Rent ($)', 'sqft_tier': 'Size Tier'},
                           color_discrete_sequence=['#00CC96'])
            fig9.update_traces(hovertemplate='<b>Tier: %{x}</b><br>Average Rent: $%{y:,.2f}<extra></extra>')
            fig9.update_layout(height=420, hoverlabel=dict(font_size=13))
            st.plotly_chart(fig9, use_container_width=True, config=chart_config, key="chart_9")

        with col_f4:
            st.markdown("##### Pet Policy Market Split (Donut Chart)")
            df_pets11 = df_filtered.copy()
            df_pets11['Pet Policy'] = df_pets11['pets_allowed_bin'].map({0: 'No Pets', 1: 'Pet-Friendly'})
            agg11 = df_pets11['Pet Policy'].value_counts().reset_index()
            agg11.columns = ['Pet Policy', 'Count']
            fig11 = go.Figure(data=[go.Pie(
                labels=agg11['Pet Policy'], values=agg11['Count'],
                hole=0.5, marker_colors=['#EF553B', '#00CC96'],
                textinfo='percent+label', textposition='auto',
                hovertemplate='<b>Policy: %{label}</b><br>Apartments: %{value:,}<br>Share: %{percent}<extra></extra>'
            )])
            fig11.update_layout(height=450, margin=dict(t=30, b=30, l=30, r=30), showlegend=True, hoverlabel=dict(font_size=14))
            st.plotly_chart(fig11, use_container_width=True, config=chart_config, key="chart_11")

        st.markdown("##### Amenities Count vs Average Rent (Line Graph)")
        agg12 = df_filtered.groupby('amenities_count', as_index=False)['price'].mean().sort_values('amenities_count')
        fig12 = px.line(agg12, x='amenities_count', y='price', markers=True,
                        labels={'amenities_count': 'Amenities Count', 'price': 'Average Price ($)'},
                        color_discrete_sequence=['#FFA15A'])
        fig12.update_traces(hovertemplate='<b>Amenities Count: %{x}</b><br>Average Rent: $%{y:,.2f}<extra></extra>')
        fig12.update_layout(height=420, hoverlabel=dict(font_size=13))
        st.plotly_chart(fig12, use_container_width=True, config=chart_config, key="chart_12")

    # ── SUB‑TAB 3 — Physical Layout & Floorplan ──
    with subtab_layout:
        st.subheader("Physical Layout & Floorplan Mechanics")

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.markdown("##### Bedrooms Distribution (Pie Chart)")
            bed_counts5 = df_filtered['bedrooms'].value_counts().sort_index().reset_index()
            bed_counts5.columns = ['Bedrooms', 'Count']
            bed_counts5['Bedrooms'] = bed_counts5['Bedrooms'].astype(int).astype(str) + ' Bed'
            fig5_pie = px.pie(bed_counts5, names='Bedrooms', values='Count',
                              color_discrete_sequence=px.colors.qualitative.Pastel1)
            fig5_pie.update_traces(
                textinfo='percent+label', textposition='auto', insidetextorientation='horizontal',
                hovertemplate='<b>Category: %{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
            )
            fig5_pie.update_layout(height=450, margin=dict(t=30, b=30, l=30, r=30), showlegend=True, hoverlabel=dict(font_size=14))
            st.plotly_chart(fig5_pie, use_container_width=True, config=chart_config, key="chart_5")

        with col_l2:
            st.markdown("##### Top 10 Cities by Number of Apartments (Line Graph)")
            top_cities13 = df_filtered['cityname'].value_counts().head(10).reset_index()
            top_cities13.columns = ['City', 'Count']
            fig13 = px.line(top_cities13, x='City', y='Count', markers=True,
                            labels={'City': 'City Name', 'Count': 'Number of Apartments'},
                            color_discrete_sequence=['#AB63FA'])
            fig13.update_traces(hovertemplate='<b>City: %{x}</b><br>Apartments: %{y:,}<extra></extra>')
            fig13.update_layout(height=450, xaxis_tickangle=-45, hoverlabel=dict(font_size=13))
            st.plotly_chart(fig13, use_container_width=True, config=chart_config, key="chart_13")

        col_l3, col_l4 = st.columns(2)
        with col_l3:
            st.markdown("##### Bedroom Count Market Share (Pie Chart)")
            df_bed16 = df_filtered.copy()
            df_bed16['Bedrooms'] = df_bed16['bedrooms'].apply(
                lambda x: f"{int(x)} Bed" if x < 4 else "4+ Bed"
            )
            agg16 = df_bed16['Bedrooms'].value_counts().reset_index()
            agg16.columns = ['Bedrooms', 'Count']
            fig16 = px.pie(agg16, names='Bedrooms', values='Count',
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig16.update_traces(
                textinfo='percent+label', textposition='auto', insidetextorientation='horizontal',
                hovertemplate='<b>Category: %{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>'
            )
            fig16.update_layout(height=450, margin=dict(t=30, b=30, l=30, r=30), showlegend=True, hoverlabel=dict(font_size=14))
            st.plotly_chart(fig16, use_container_width=True, config=chart_config, key="chart_16")

        with col_l4:
            st.markdown("##### Multi-Attribute Profile by Bedrooms (Radar Chart)")
            bed_vals = sorted(df_filtered['bedrooms'].dropna().unique())
            bed_vals = [b for b in bed_vals if b > 0][:5]
            radar_attrs = ['price', 'square_feet', 'bathrooms', 'amenities_count']
            radar_labels = ['Price', 'Sq Ft', 'Bathrooms', 'Amenities']

            agg_radar = df_filtered[df_filtered['bedrooms'].isin(bed_vals)] \
                .groupby('bedrooms')[radar_attrs].mean()

            radar_norm = agg_radar.copy()
            for rc in radar_attrs:
                rc_min, rc_max = radar_norm[rc].min(), radar_norm[rc].max()
                if rc_max > rc_min:
                    radar_norm[rc] = (radar_norm[rc] - rc_min) / (rc_max - rc_min)
                else:
                    radar_norm[rc] = 0.5

            fig17 = go.Figure()
            radar_colors = px.colors.qualitative.Set2
            for idx, bed in enumerate(bed_vals):
                vals = radar_norm.loc[bed].tolist()
                vals.append(vals[0])
                fig17.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=radar_labels + [radar_labels[0]],
                    fill='toself',
                    name=f'{int(bed)} Bed',
                    line_color=radar_colors[idx % len(radar_colors)],
                    opacity=0.7,
                    hovertemplate='<b>%{fullData.name}</b><br>Attribute: %{theta}<br>Normalized Value: %{r:.2f}<extra></extra>'
                ))
            fig17.update_layout(
                height=420,
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                hoverlabel=dict(font_size=13)
            )
            st.plotly_chart(fig17, use_container_width=True, config=chart_config, key="chart_17")


# ═══════════════════════════════════════════════
#  TAB 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════
with tab2:
    st.header("Model Comparison")

    # ── 2a. Regression Metrics Summary Table ──────────────────
    st.subheader("Regression Metrics Summary")

    metrics_rows = []
    for name, info in models_dict.items():
        m = info['metrics']
        best_p = info.get('best_params', 'N/A')
        if isinstance(best_p, dict):
            formatted_params = ", ".join(f"{k}={v}" for k, v in best_p.items())
        else:
            formatted_params = str(best_p)

        metrics_rows.append({
            'Model': name,
            'R² Score': round(m['R²'], 4),
            'MAE ($)': round(m['MAE'], 2),
            'RMSE ($)': round(m['RMSE'], 2),
            'Best Hyperparameters': formatted_params,
        })
    df_metrics = pd.DataFrame(metrics_rows)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # Automatically identify best model
    best_row = df_metrics.sort_values(by=['R² Score', 'MAE ($)', 'RMSE ($)'], ascending=[False, True, True]).iloc[0]
    best_model_name = best_row['Model']
    best_r2 = float(best_row['R² Score'])
    best_mae = float(best_row['MAE ($)'])
    best_rmse = float(best_row['RMSE ($)'])

    st.success(
        f"**Best Performing Model**: **{best_model_name}** achieved the highest R² of {best_r2:.4f}, "
        f"with an MAE of ${best_mae:,.2f} and RMSE of ${best_rmse:,.2f}, providing the strongest overall regression performance."
    )

    # Decision Tree Generalization Audit
    dt_info = models_dict.get('Decision Tree', {})
    if 'training_r2' in dt_info and dt_info['training_r2'] is not None:
        c_dt1, c_dt2, c_dt3 = st.columns(3)
        c_dt1.metric("Decision Tree Training R²", f"{dt_info['training_r2']:.4f}")
        c_dt2.metric("Decision Tree Testing R²", f"{dt_info['metrics']['R²']:.4f}")
        c_dt3.metric("Decision Tree Generalization Gap (R²)", f"{dt_info.get('r2_gap', 0.0):.4f}")

    # ── 2b. Multi‑Metric Score Comparison Bar Chart ──────────
    st.subheader("Multi‑Metric Score Comparison Bar Chart")
    df_melt = df_metrics[['Model', 'R² Score', 'MAE ($)', 'RMSE ($)']].melt(id_vars='Model', var_name='Metric', value_name='Value')
    fig_multi = px.bar(df_melt, x='Model', y='Value', color='Metric', barmode='group',
                       text_auto='.2f',
                       color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'])
    fig_multi.update_layout(height=450, yaxis_title="Score / Error", hoverlabel=dict(font_size=13))
    st.plotly_chart(fig_multi, use_container_width=True, config=chart_config, key="chart_tab2_multi")

    st.divider()

    # ── 2c. Classification Metrics (Price Quartiles) ─────────
    st.subheader("Secondary Business Analysis: Price Quartile Classification")

    frozen_edges = pipeline_meta['training_quartile_edges']
    q1_val, q2_val, q3_val = frozen_edges[1], frozen_edges[2], frozen_edges[3]

    st.markdown(f"""
| Price Category | Definition | Interpretation | Cutoff (Training Data) |
| :--- | :--- | :--- | :--- |
| **Budget** | Price ≤ Q1 | Lowest rental-price range | ≤ ${q1_val:,.0f} |
| **Moderate** | Q1 < Price ≤ Q2 | Lower-middle rental-price range | ${q1_val:,.0f} – ${q2_val:,.0f} |
| **Premium** | Q2 < Price ≤ Q3 | Upper-middle rental-price range | ${q2_val:,.0f} – ${q3_val:,.0f} |
| **Luxury** | Price > Q3 | Highest rental-price range | > ${q3_val:,.0f} |
""")

    cls_rows = []
    for name, info in models_dict.items():
        labels, yt_bin, yp_bin, _ = price_quartile_metrics(info['y_test'], info['y_pred'], frozen_training_edges=frozen_edges)
        cls_rows.append({
            'Model': name,
            'Accuracy': round(accuracy_score(yt_bin, yp_bin), 4),
            'Precision (Weighted)': round(precision_score(yt_bin, yp_bin, average='weighted', zero_division=0), 4),
            'Recall (Weighted)': round(recall_score(yt_bin, yp_bin, average='weighted', zero_division=0), 4),
            'F1 Score (Weighted)': round(f1_score(yt_bin, yp_bin, average='weighted', zero_division=0), 4),
        })
    df_cls = pd.DataFrame(cls_rows)
    st.dataframe(df_cls, use_container_width=True, hide_index=True)

    st.markdown("#### Model Classification Performance Comparison (Accuracy & F1-Score)")
    df_cls_melt = df_cls.melt(id_vars=['Model'], value_vars=['Accuracy', 'F1 Score (Weighted)'],
                              var_name='Metric', value_name='Score')
    fig_cls_bar = px.bar(df_cls_melt, x='Model', y='Score', color='Metric', barmode='group',
                         text_auto='.4f', color_discrete_sequence=['#636EFA', '#00CC96'])
    fig_cls_bar.update_layout(height=420, yaxis_range=[0, 1.05], yaxis_title="Score", hoverlabel=dict(font_size=13))
    st.plotly_chart(fig_cls_bar, use_container_width=True, config=chart_config, key="chart_tab2_cls_bar")

    st.divider()

    # ── 2d. Inspect Individual Model ─────────────────────────
    st.subheader("Inspect Individual Model")
    model_choice = st.selectbox("Select a model to inspect", list(models_dict.keys()))

    info = models_dict[model_choice]
    labels, yt_bin, yp_bin, q_edges = price_quartile_metrics(info['y_test'], info['y_pred'], frozen_training_edges=frozen_edges)

    # Actual vs Predicted Scatter
    st.markdown("#### Actual vs Predicted Price Scatter Plot")
    df_scatter = pd.DataFrame({
        'Actual': info['y_test'],
        'Predicted': info['y_pred']
    })
    if len(df_scatter) > 5000:
        df_scatter = df_scatter.sample(5000, random_state=42)

    fig_ap = px.scatter(
        df_scatter,
        x='Actual',
        y='Predicted',
        opacity=0.35,
        labels={'Actual': 'Actual Price ($)', 'Predicted': 'Predicted Price ($)'},
        color_discrete_sequence=['#AB63FA']
    )
    line_min = min(df_scatter['Actual'].min(), df_scatter['Predicted'].min())
    line_max = max(df_scatter['Actual'].max(), df_scatter['Predicted'].max())
    fig_ap.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    fig_ap.update_layout(height=500, hoverlabel=dict(font_size=13))
    st.plotly_chart(fig_ap, use_container_width=True, config=chart_config, key="chart_tab2_scatter")

    # Residual Plot
    st.markdown("#### Residual Plot")
    df_residual = pd.DataFrame({
        'Predicted': info['y_pred'],
        'Residual': np.array(info['y_test']) - np.array(info['y_pred'])
    })
    if len(df_residual) > 5000:
        df_residual = df_residual.sample(5000, random_state=42)

    fig_res = px.scatter(
        df_residual,
        x='Predicted',
        y='Residual',
        opacity=0.35,
        labels={'Predicted': 'Predicted Price ($)', 'Residual': 'Residual ($)'},
        color_discrete_sequence=['#FF69B4']
    )
    line_min_res = df_residual['Predicted'].min()
    line_max_res = df_residual['Predicted'].max()
    fig_res.add_trace(
        go.Scatter(
            x=[line_min_res, line_max_res],
            y=[0, 0],
            mode='lines',
            name='Zero Residual',
            line=dict(color='red', dash='dash')
        )
    )
    fig_res.update_layout(height=500, hoverlabel=dict(font_size=13))
    st.plotly_chart(fig_res, use_container_width=True, config=chart_config, key="chart_tab2_residual")

    # Class‑level metrics table
    st.markdown("#### Class‑Level Metrics (per Price Quartile)")
    report = classification_report(yt_bin, yp_bin, output_dict=True, zero_division=0)
    class_rows = []
    for lbl in labels:
        if lbl in report:
            r = report[lbl]
            class_rows.append({
                'Price Range': lbl,
                'Accuracy': round(accuracy_score(yt_bin == lbl, yp_bin == lbl), 4),
                'Precision': round(r['precision'], 4),
                'Recall': round(r['recall'], 4),
                'F1 Score': round(r['f1-score'], 4),
            })
    df_class = pd.DataFrame(class_rows)
    st.dataframe(df_class, use_container_width=True, hide_index=True)

    # Class-Level Performance Bar Chart
    df_class_melt = df_class.melt(id_vars=['Price Range'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                  var_name='Metric', value_name='Score')
    fig_class_bar = px.bar(df_class_melt, x='Price Range', y='Score', color='Metric', barmode='group',
                           text_auto='.4f', color_discrete_sequence=['#AB63FA', '#FFA15A', '#19D3F3', '#00CC96'],
                           title=f"Quartile Performance Breakdown — {model_choice}")
    fig_class_bar.update_layout(height=420, yaxis_range=[0, 1.05], yaxis_title="Score", hoverlabel=dict(font_size=13))
    st.plotly_chart(fig_class_bar, use_container_width=True, config=chart_config, key="chart_tab2_class_bar")

    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(yt_bin, yp_bin, labels=labels)
    fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, aspect='auto',
                       color_continuous_scale='Blues',
                       labels={'x': 'Predicted Quartile', 'y': 'Actual Quartile', 'color': 'Count'})
    fig_cm.update_layout(height=500, hoverlabel=dict(font_size=13))
    st.plotly_chart(fig_cm, use_container_width=True, config=chart_config, key="chart_tab2_cm")


# ═══════════════════════════════════════════════
#  TAB 3 — DATA EXPLORER
# ═══════════════════════════════════════════════
with tab3:
    st.header("Data Explorer")

    subtab_raw, subtab_cleaned, subtab_transformed = st.tabs([
        "Raw Dataset",
        "Cleaned Dataset",
        "Data Transformed",
    ])

    with subtab_raw:
        st.subheader("Raw Dataset")
        if os.path.exists("apartments_for_rent_classified_100K.csv"):
            @st.cache_data
            def load_raw_dataset():
                return pd.read_csv("apartments_for_rent_classified_100K.csv",
                                   sep=';', encoding='latin-1', low_memory=False)
            df_raw_preview = load_raw_dataset()

            c_r1, c_r2 = st.columns(2)
            c_r1.metric("Total Rows", f"{len(df_raw_preview):,}")
            c_r2.metric("Total Columns", df_raw_preview.shape[1])

            st.divider()

            st.markdown("##### Missing Value Analysis")
            total_rows_raw = len(df_raw_preview)
            missing_count = df_raw_preview.isnull().sum()
            missing_pct = (missing_count / total_rows_raw) * 100
            missing_df = pd.DataFrame({
                'Column': missing_count.index,
                'Missing Count': missing_count.values,
                'Percentage (%)': missing_pct.values.round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
                by='Missing Count', ascending=False
            ).reset_index(drop=True)
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.success("No missing values found in the raw dataset.")

            st.divider()
            st.markdown("##### Dataset Preview")
            raw_n = st.slider("Number of rows to display", min_value=5, max_value=100,
                              value=10, step=5, key="raw_preview_n")
            st.dataframe(df_raw_preview.head(raw_n), use_container_width=True, hide_index=True)
        else:
            st.info("Raw file `apartments_for_rent_classified_100K.csv` not found.")

    with subtab_cleaned:
        st.subheader("Cleaned Dataset")
        st.caption("This view shows the dataset after duplicate removal, whitespace stripping, and text cleaning.")

        if os.path.exists("apartments_for_rent_classified_100K.csv"):
            @st.cache_data
            def load_cleaned_dataset():
                df_raw = pd.read_csv("apartments_for_rent_classified_100K.csv",
                                     sep=';', encoding='latin-1', low_memory=False)
                return clean_apartment_dataset(df_raw)
            df_cleaned_preview = load_cleaned_dataset()

            c_c1, c_c2 = st.columns(2)
            c_c1.metric("Total Rows", f"{len(df_cleaned_preview):,}")
            c_c2.metric("Total Columns", df_cleaned_preview.shape[1])

            st.divider()
            st.markdown("##### Column Overview")
            col_info = pd.DataFrame({
                'Column': df_cleaned_preview.columns,
                'Data Type': df_cleaned_preview.dtypes.astype(str).values,
                'Non-Null Count': df_cleaned_preview.notnull().sum().values,
                'Null Count': df_cleaned_preview.isnull().sum().values,
            }).reset_index(drop=True)
            st.dataframe(col_info, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("##### Dataset Preview")
            clean_n = st.slider("Number of rows to display", min_value=5, max_value=100,
                                value=10, step=5, key="clean_preview_n")
            st.dataframe(df_cleaned_preview.head(clean_n), use_container_width=True, hide_index=True)
        else:
            st.info("Raw file `apartments_for_rent_classified_100K.csv` not found.")

    with subtab_transformed:
        st.subheader("Data Transformed")
        st.caption("Prepared dataset containing structural features, location attributes, and engineered feature columns.")

        c_t1, c_t2 = st.columns(2)
        c_t1.metric("Total Rows", f"{len(df_explore):,}")
        c_t2.metric("Total Columns", df_explore.shape[1])

        st.divider()
        st.markdown("Columns Added After Data Transformation")
        col_eng1, col_eng2 = st.columns(2)
        with col_eng1:
            st.markdown("**amenities_count** — Number of amenities per listing")
            amen_stats = df_explore['amenities_count'].describe().round(2)
            st.dataframe(amen_stats.to_frame("amenities_count"), use_container_width=True)
        with col_eng2:
            st.markdown("**pets_allowed_bin** — Pet policy binary flag")
            pet_counts = df_explore['pets_allowed_bin'].value_counts().reset_index()
            pet_counts.columns = ['Value', 'Count']
            pet_counts['Label'] = pet_counts['Value'].map({0: 'No Pets / Unspecified', 1: 'Pets Allowed'})
            pet_counts['Percentage'] = (pet_counts['Count'] / pet_counts['Count'].sum() * 100).round(2)
            st.dataframe(pet_counts[['Label', 'Count', 'Percentage']], use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("##### Dataset Preview")
        trans_n = st.slider("Number of rows to display", min_value=5, max_value=100,
                            value=20, step=5, key="trans_preview_n")
        st.dataframe(df_explore.head(trans_n), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
#  TAB 4 — PRICE PREDICTOR
# ═══════════════════════════════════════════════
@st.fragment
def render_price_predictor():
    st.header("Price Predictor")
    st.markdown("Enter apartment features and select a model to predict the estimated monthly rent.")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        pred_state = st.selectbox("State", sorted(df_explore['state'].dropna().unique().tolist()), key='pred_state')
        pred_beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1, key='pred_beds')
        pred_baths = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5, key='pred_baths')

    with col_p2:
        pred_sqft = st.number_input("Square Feet", min_value=100, max_value=10000, value=1000, step=50, key='pred_sqft')
        pred_pets = st.checkbox("Pets Allowed", value=True, key='pred_pets')
        amenity_options = ["Gym", "Pool", "Parking", "Washer/Dryer", "AC", "Balcony"]
        pred_amenities = st.multiselect("Amenities", amenity_options, default=["Parking", "AC"], key='pred_amen')

    st.divider()

    pred_model_name = st.selectbox("Choose Prediction Model", list(models_dict.keys()), key='pred_model')

    if st.button("Predict Rent", type="primary", use_container_width=True):
        transformer = pipeline_meta['transformer']
        state_lat = transformer.state_lat_medians_.get(pred_state, transformer.medians_.get("latitude", 37.0))
        state_lon = transformer.state_lon_medians_.get(pred_state, transformer.medians_.get("longitude", -95.0))

        input_data = {
            'bedrooms': pred_beds,
            'bathrooms': pred_baths,
            'pets_allowed_bin': 1 if pred_pets else 0,
            'amenities_count': len(pred_amenities),
            'square_feet': pred_sqft,
            'state': pred_state,
            'cityname': 'Unknown',
            'latitude': state_lat,
            'longitude': state_lon,
        }

        sel_info = models_dict[pred_model_name]
        sel_pred = sel_info['predict_fn'](
            sel_info['model'],
            sel_info['transformer'],
            sel_info['scaler'],
            input_data
        )

        st.success(f"### Estimated Monthly Rent ({pred_model_name}): **${sel_pred:,.2f}**")


with tab4:
    render_price_predictor()

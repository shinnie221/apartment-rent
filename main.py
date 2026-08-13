import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# Set page config
st.set_page_config(page_title="Apartment Rental Price Prediction & Valuation System", page_icon="🏢", layout="wide")

# Import models
from model.lr import train_model as train_lr, predict_property as predict_lr
from model.knn import train_model as train_knn, predict_property as predict_knn
from model.dt import train_model as train_dt, predict_property as predict_dt
from model.rf import train_model as train_rf, predict_property as predict_rf


# ──────────────────────────────────────────────
#  DATA LOADING & MODEL TRAINING (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("apartments_for_rent_fully_prepared.csv")
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_resource
def get_trained_models(_df, knn_n_neighbors=5, dt_max_depth=10, rf_n_estimators=100, rf_max_depth=10):
    """Train all 4 models and return results dict (including y_test / y_pred)."""
    results = {}

    # Linear Regression
    lr_model, lr_scaler, lr_features, lr_mae, lr_rmse, lr_r2, lr_yt, lr_yp = train_lr(_df)
    results['Linear Regression'] = {
        'model': lr_model, 'scaler': lr_scaler, 'features': lr_features,
        'predict_fn': predict_lr,
        'metrics': {'MAE': lr_mae, 'RMSE': lr_rmse, 'R²': lr_r2},
        'y_test': lr_yt, 'y_pred': lr_yp,
    }

    # KNN
    knn_model, knn_scaler, knn_features, knn_mae, knn_rmse, knn_r2, knn_yt, knn_yp = train_knn(_df, n_neighbors=knn_n_neighbors)
    results['KNN Regressor'] = {
        'model': knn_model, 'scaler': knn_scaler, 'features': knn_features,
        'predict_fn': predict_knn,
        'metrics': {'MAE': knn_mae, 'RMSE': knn_rmse, 'R²': knn_r2},
        'y_test': knn_yt, 'y_pred': knn_yp,
    }

    # Decision Tree
    dt_model, dt_scaler, dt_features, dt_mae, dt_rmse, dt_r2, dt_yt, dt_yp = train_dt(_df, max_depth=dt_max_depth)
    results['Decision Tree'] = {
        'model': dt_model, 'scaler': dt_scaler, 'features': dt_features,
        'predict_fn': predict_dt,
        'metrics': {'MAE': dt_mae, 'RMSE': dt_rmse, 'R²': dt_r2},
        'y_test': dt_yt, 'y_pred': dt_yp,
    }

    # Random Forest
    rf_model, rf_scaler, rf_features, rf_mae, rf_rmse, rf_r2, rf_yt, rf_yp = train_rf(_df, n_estimators=rf_n_estimators, max_depth=rf_max_depth)
    results['Random Forest'] = {
        'model': rf_model, 'scaler': rf_scaler, 'features': rf_features,
        'predict_fn': predict_rf,
        'metrics': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R²': rf_r2},
        'y_test': rf_yt, 'y_pred': rf_yp,
    }

    return results


# ──────────────────────────────────────────────
#  HELPER: quartile‑based classification metrics
# ──────────────────────────────────────────────
def price_quartile_metrics(y_test, y_pred):
    """
    Bin *actual* test prices into 4 equal‑frequency quartiles,
    then assign predicted prices to the same bin edges.
    Returns (bin_labels, y_test_binned, y_pred_binned, quartile_edges).
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Create 4 equal‑frequency bins from actual prices
    quartile_edges = np.quantile(y_test, [0, 0.25, 0.50, 0.75, 1.0])
    # Ensure unique edges (add tiny epsilon if needed)
    quartile_edges = np.unique(quartile_edges)
    if len(quartile_edges) < 3:
        quartile_edges = np.linspace(y_test.min(), y_test.max(), 5)

    labels = _make_labels(quartile_edges)

    y_test_binned = pd.cut(y_test, bins=quartile_edges, labels=labels, include_lowest=True)
    y_pred_binned = pd.cut(y_pred, bins=quartile_edges, labels=labels, include_lowest=True)

    # Handle predictions outside the range – assign to nearest bin
    y_pred_binned = y_pred_binned.fillna(
        pd.cut(np.clip(y_pred, quartile_edges[0], quartile_edges[-1]),
               bins=quartile_edges, labels=labels, include_lowest=True)
    )
    # Still NaN → assign to closest label
    y_pred_binned = y_pred_binned.fillna(labels[-1])

    return labels, y_test_binned, y_pred_binned, quartile_edges


def _make_labels(edges):
    """Create human‑readable labels from bin edges."""
    tag_names = ['Budget', 'Moderate', 'Premium', 'Luxury']
    labels = []
    for i in range(len(edges) - 1):
        name = tag_names[i] if i < len(tag_names) else f"Q{i+1}"
        labels.append(f"{name} (${edges[i]:,.0f}–${edges[i+1]:,.0f})")
    return labels


# ──────────────────────────────────────────────
#  LOAD DATA & TRAIN MODELS
# ──────────────────────────────────────────────
df_raw = load_data()
if df_raw.empty:
    st.stop()

# ──────────────────────────────────────────────
#  SIDEBAR – MODEL HYPERPARAMETERS
# ──────────────────────────────────────────────
st.sidebar.header("🧠 Model Hyperparameters")
st.sidebar.caption("Adjust parameters below and models will retrain automatically.")

# KNN
st.sidebar.markdown("**KNN Regressor**")
sidebar_knn_k = st.sidebar.slider(
    "n_neighbors (K)", min_value=1, max_value=20, value=5, step=1,
    help="Number of neighbors to use for KNN."
)

# Decision Tree
st.sidebar.markdown("**Decision Tree**")
sidebar_dt_depth = st.sidebar.slider(
    "max_depth (DT)", min_value=1, max_value=30, value=10, step=1,
    help="Maximum depth of the decision tree."
)

# Random Forest
st.sidebar.markdown("**Random Forest**")
sidebar_rf_estimators = st.sidebar.slider(
    "n_estimators (RF)", min_value=10, max_value=300, value=100, step=10,
    help="Number of trees in the random forest."
)
sidebar_rf_depth = st.sidebar.slider(
    "max_depth (RF)", min_value=1, max_value=30, value=10, step=1,
    help="Maximum depth of each tree in the random forest."
)

st.sidebar.markdown("**Linear Regression** — _no tunable hyperparameters_")

st.sidebar.divider()

with st.spinner("Training models... This may take a moment."):
    models_dict = get_trained_models(
        df_raw,
        knn_n_neighbors=sidebar_knn_k,
        dt_max_depth=sidebar_dt_depth,
        rf_n_estimators=sidebar_rf_estimators,
        rf_max_depth=sidebar_rf_depth,
    )

# Basic preprocessing for exploration
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

# Derived column for chart #3
df_explore['price_per_sqft'] = np.where(
    df_explore['square_feet'] > 0,
    df_explore['price'] / df_explore['square_feet'],
    np.nan
)

# Derived column for lat-lon difference chart
df_explore['lat_lon_diff'] = df_explore['latitude'] - df_explore['longitude']

# ──────────────────────────────────────────────
#  SIDEBAR – GLOBAL FILTERS
# ──────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

# State filter
all_states = sorted(df_explore['state'].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("Filter by State", all_states, default=all_states)

# Bedrooms range
bed_min = int(df_explore['bedrooms'].min())
bed_max = int(df_explore['bedrooms'].max())
bed_range = st.sidebar.slider(
    "Bedrooms Range", min_value=bed_min, max_value=bed_max,
    value=(bed_min, bed_max), step=1
)

# Bathrooms range
bath_min = float(df_explore['bathrooms'].min())
bath_max = float(df_explore['bathrooms'].max())
bath_range = st.sidebar.slider(
    "Bathrooms Range", min_value=bath_min, max_value=bath_max,
    value=(bath_min, bath_max), step=0.5
)

# Pet Allowed filter
pet_options = ["All", "Pet-Friendly Only", "No Pets Only"]
selected_pet = st.sidebar.radio("Pet Policy", pet_options, index=0)

# Amenities count range
amen_min = int(df_explore['amenities_count'].min())
amen_max = int(df_explore['amenities_count'].max())
amen_range = st.sidebar.slider(
    "Amenities Count Range", min_value=amen_min, max_value=amen_max,
    value=(amen_min, amen_max), step=1
)

# Apply filters (without price filter)
mask = (
    (df_explore['state'].isin(selected_states)) &
    (df_explore['bedrooms'] >= bed_range[0]) &
    (df_explore['bedrooms'] <= bed_range[1]) &
    (df_explore['bathrooms'] >= bath_range[0]) &
    (df_explore['bathrooms'] <= bath_range[1]) &
    (df_explore['amenities_count'] >= amen_range[0]) &
    (df_explore['amenities_count'] <= amen_range[1])
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
st.title("🏢 Apartment Rental Price Prediction & Valuation System")

tab1, tab2, tab3, tab4 = st.tabs([
    "Visualisation Dashboard",
    "Model Comparison",
    "Data Explorer",
    "Price Predictor",
])

# Chart config – disable zoom/pan/toolbar
static_config = {'staticPlot': True, 'displayModeBar': False}


# ═══════════════════════════════════════════════
#  TAB 1 — VISUALISATION DASHBOARD (15 charts)
#  Organised into 3 sub‑tabs
# ═══════════════════════════════════════════════
with tab1:
    st.header("Visualisation Dashboard")
    st.caption(f"All charts reflect current sidebar filters ({len(df_filtered):,} records).")

    subtab_loc, subtab_feat, subtab_layout = st.tabs([
        "Location & Geographic",
        "Single Feature vs Price",
        "Physical Layout & Floorplan",
    ])

    # ═════════════════════════════════════════════
    #  SUB‑TAB 1 — Location & Geographic Drivers
    # ═════════════════════════════════════════════
    with subtab_loc:
        st.subheader("Location & Geographic Drivers")

        # Chart 1: Distribution of Apartment Prices — Histogram
        st.markdown("##### Chart 1 · Distribution of Apartment Prices")
        df_price_hist = df_filtered.dropna(subset=['price']).copy()
        q99_price = df_price_hist['price'].quantile(0.99)
        df_price_hist = df_price_hist[df_price_hist['price'] <= q99_price]
        fig3b = px.histplot(df_price_hist, x='price', nbins=50,
                             labels={'price': 'Price ($)', 'count': 'Frequency'},
                             color_discrete_sequence=['#87CEEB'])
        fig3b.update_layout(height=450, xaxis_title='Price ($)', yaxis_title='Frequency')
        st.plotly_chart(fig3b, use_container_width=True, config=static_config)

        # Chart 2: Average Apartment Price by State (Top 10 by listing count)
        st.markdown("##### Chart 2 · Average Price by State (Top 10 by Listing Count)")
        state_counts = df_filtered['state'].value_counts()
        top_10_states_by_count = state_counts.head(10).index.tolist()
        df_top10_listing = df_filtered[df_filtered['state'].isin(top_10_states_by_count)]
        avg_price_by_state = df_top10_listing.groupby('state', as_index=False)['price'].mean() \
            .sort_values('price', ascending=False)
        fig3c = px.bar(avg_price_by_state, x='state', y='price',
                       labels={'price': 'Average Price ($)', 'state': 'State'},
                       color='price', color_continuous_scale='Viridis')
        fig3c.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig3c, use_container_width=True, config=static_config)

        # Chart 3D: Distribution of Latitude-Longitude Difference — Histogram
        st.markdown("##### Chart 3 · Distribution of Latitude–Longitude Difference")
        df_latlon = df_filtered.dropna(subset=['lat_lon_diff']).copy()
        fig3d = px.histogram(df_latlon, x='lat_lon_diff', nbins=50,
                             labels={'lat_lon_diff': 'Latitude − Longitude Difference', 'count': 'Number of Apartments'},
                             color_discrete_sequence=['#87CEEB'])
        fig3d.update_layout(height=450, xaxis_title='Latitude − Longitude Difference',
                            yaxis_title='Number of Apartments')
        st.plotly_chart(fig3d, use_container_width=True, config=static_config)

    # ═════════════════════════════════════════════
    #  SUB‑TAB 2 — Single Feature vs Price Drivers
    # ═════════════════════════════════════════════
    with subtab_feat:
        st.subheader("Single Feature vs Price Drivers")

        col_a, col_b = st.columns(2)

        # Chart 4: bedrooms vs price — Violin Plot
        with col_a:
            st.markdown("##### Chart 4 · Bedrooms vs Price (Violin)")
            df_v4 = df_filtered[df_filtered['bedrooms'] > 0].copy()
            df_v4['bedrooms_str'] = df_v4['bedrooms'].astype(int).astype(str)
            fig4 = px.violin(df_v4, x='bedrooms_str', y='price', box=True,
                             labels={'bedrooms_str': 'Bedrooms', 'price': 'Price ($)'},
                             color_discrete_sequence=['#636EFA'])
            fig4.update_layout(height=450)
            st.plotly_chart(fig4, use_container_width=True, config=static_config)

        # Chart 5: bathrooms vs price — Box Plot
        with col_b:
            st.markdown("##### Chart 5 · Bathrooms vs Price Distribution (Box)")
            fig5 = px.box(df_filtered, x='bathrooms', y='price',
                          labels={'price': 'Price ($)', 'bathrooms': 'Bathrooms'},
                          color_discrete_sequence=['#EF553B'])
            fig5.update_layout(height=450)
            st.plotly_chart(fig5, use_container_width=True, config=static_config)

        # Chart 6: square_feet vs price — Scatter with OLS trendline
        st.markdown("##### Chart 6 · Square Feet vs Price (Scatter + OLS Trend)")
        sample6 = df_filtered.dropna(subset=['square_feet', 'price'])
        if len(sample6) > 5000:
            sample6 = sample6.sample(5000, random_state=42)
        fig6 = px.scatter(sample6, x='square_feet', y='price', opacity=0.35,
                          trendline='ols',
                          labels={'square_feet': 'Square Feet', 'price': 'Monthly Rent ($)'},
                          color_discrete_sequence=['#AB63FA'])
        fig6.update_layout(height=450)
        st.plotly_chart(fig6, use_container_width=True, config=static_config)

        col_c, col_d = st.columns(2)

        # Chart 7: square_feet (Binned) vs price — Area Graph
        with col_c:
            st.markdown("##### Chart 7 · Size Tiers vs Average Rent (Area)")
            df_bin7 = df_filtered.dropna(subset=['square_feet']).copy()
            df_bin7['sqft_tier'] = pd.cut(df_bin7['square_feet'],
                                          bins=[0, 500, 800, 1200, 1800, 100000],
                                          labels=['<500', '500–800', '800–1200', '1200–1800', '1800+'])
            agg7 = df_bin7.groupby('sqft_tier', as_index=False, observed=True)['price'].mean()
            fig7 = px.area(agg7, x='sqft_tier', y='price',
                           labels={'price': 'Avg Rent ($)', 'sqft_tier': 'Size Tier'},
                           color_discrete_sequence=['#00CC96'])
            fig7.update_layout(height=400)
            st.plotly_chart(fig7, use_container_width=True, config=static_config)

        # Chart 8: pets_allowed_bin Market Split — Donut Chart
        with col_d:
            st.markdown("##### Chart 8 · Pet Policy Market Split (Donut)")
            df_pets8 = df_filtered.copy()
            df_pets8['Pet Policy'] = df_pets8['pets_allowed_bin'].map({0: 'No Pets', 1: 'Pet-Friendly'})
            agg8 = df_pets8['Pet Policy'].value_counts().reset_index()
            agg8.columns = ['Pet Policy', 'Count']
            fig8 = go.Figure(data=[go.Pie(
                labels=agg8['Pet Policy'], values=agg8['Count'],
                hole=0.5, marker_colors=['#EF553B', '#00CC96'],
                textinfo='label+percent', textposition='outside'
            )])
            fig8.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig8, use_container_width=True, config=static_config)

        # Chart 9: amenities_count vs price — Line Graph with Markers
        st.markdown("##### Chart 9 · Amenities Count vs Average Rent (Line)")
        agg9 = df_filtered.groupby('amenities_count', as_index=False)['price'].mean().sort_values('amenities_count')
        fig9 = px.line(agg9, x='amenities_count', y='price', markers=True,
                       labels={'amenities_count': 'Amenities Count', 'price': 'Average Price ($)'},
                       color_discrete_sequence=['#FFA15A'])
        fig9.update_layout(height=400)
        st.plotly_chart(fig9, use_container_width=True, config=static_config)

        # Chart 9B: Top 10 Cities by Number of Apartments — Line Chart
        st.markdown("##### Chart 9B · Top 10 Cities by Number of Apartments (Line)")
        top_cities_count = df_filtered['cityname'].value_counts().head(10).reset_index()
        top_cities_count.columns = ['City', 'Count']
        fig9b = px.line(top_cities_count, x='City', y='Count', markers=True,
                        labels={'City': 'City Name', 'Count': 'Number of Apartments'},
                        color_discrete_sequence=['#AB63FA'])
        fig9b.update_layout(height=450, xaxis_tickangle=-45)
        st.plotly_chart(fig9b, use_container_width=True, config=static_config)

    # ═════════════════════════════════════════════
    #  SUB‑TAB 3 — Physical Layout & Floorplan
    # ═════════════════════════════════════════════
    with subtab_layout:
        st.subheader("3️⃣ Physical Layout & Floorplan Mechanics")

        # Chart 10: square_feet × bedrooms — 2D Pivot Heatmap (Median SqFt)
        st.markdown("##### Chart 10 · Square Feet Tier × Bedrooms (Median Sq Ft Heatmap)")
        df_h10 = df_filtered.dropna(subset=['square_feet']).copy()
        df_h10['sqft_tier'] = pd.cut(df_h10['square_feet'],
                                      bins=[0, 500, 800, 1200, 1800, 100000],
                                      labels=['<500', '500–800', '800–1200', '1200–1800', '1800+'])
        pivot10 = df_h10.pivot_table(index='sqft_tier', columns='bedrooms', values='square_feet',
                                      aggfunc='median', fill_value=0, observed=True)
        fig10 = px.imshow(pivot10, text_auto='.0f', aspect='auto',
                          color_continuous_scale='YlGnBu',
                          labels={'x': 'Bedrooms', 'y': 'Square Feet Tier', 'color': 'Median Sq Ft'})
        fig10.update_layout(height=420)
        st.plotly_chart(fig10, use_container_width=True, config=static_config)

        col_e, col_f = st.columns(2)

        # Chart 11: bedrooms Inventory Split — Pie Chart
        with col_e:
            st.markdown("##### Chart 11 · Bedroom Count Market Share (Pie)")
            df_bed11 = df_filtered.copy()
            df_bed11['Bedrooms'] = df_bed11['bedrooms'].apply(
                lambda x: f"{int(x)} Bed" if x < 4 else "4+ Bed"
            )
            agg11 = df_bed11['Bedrooms'].value_counts().reset_index()
            agg11.columns = ['Bedrooms', 'Count']
            fig11 = px.pie(agg11, names='Bedrooms', values='Count',
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig11.update_layout(height=420)
            st.plotly_chart(fig11, use_container_width=True, config=static_config)

        # Chart 11B: Bedrooms Distribution — Pie Chart
        st.markdown("##### Chart 11B · Bedrooms Distribution (Pie Chart)")
        bed_counts_pie = df_filtered['bedrooms'].value_counts().sort_index().reset_index()
        bed_counts_pie.columns = ['Bedrooms', 'Count']
        bed_counts_pie['Bedrooms'] = bed_counts_pie['Bedrooms'].astype(int).astype(str) + ' Bed'
        fig11b = px.pie(bed_counts_pie, names='Bedrooms', values='Count',
                        color_discrete_sequence=px.colors.qualitative.Pastel1)
        fig11b.update_traces(textinfo='label+percent', textposition='outside')
        fig11b.update_layout(height=450)
        st.plotly_chart(fig11b, use_container_width=True, config=static_config)

        # Chart 12: state vs square_feet (Top 10) — Horizontal Bar (Median)
        with col_f:
            st.markdown("##### Chart 12 · Top 10 States by Median Apartment Size")
            agg12 = df_filtered.groupby('state', as_index=False)['square_feet'].median() \
                .nlargest(10, 'square_feet').sort_values('square_feet', ascending=True)
            fig12 = px.bar(agg12, y='state', x='square_feet', orientation='h',
                           labels={'square_feet': 'Median Sq Ft', 'state': 'State'},
                           color='square_feet', color_continuous_scale='Peach')
            fig12.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig12, use_container_width=True, config=static_config)

        st.divider()

        # ── Multi‑Variable Feature Interactions ──────────────
        st.subheader("4️⃣ Multi‑Variable Feature Interactions")

        # Chart 13: bedrooms × bathrooms vs price — 2D Matrix Heatmap
        st.markdown("##### Chart 13 · Bedrooms × Bathrooms → Average Rent Heatmap")
        pivot13 = df_filtered.pivot_table(index='bathrooms', columns='bedrooms', values='price',
                                           aggfunc='mean', observed=True)
        fig13 = px.imshow(pivot13, text_auto='$,.0f', aspect='auto',
                          color_continuous_scale='RdYlGn',
                          labels={'x': 'Bedrooms', 'y': 'Bathrooms', 'color': 'Avg Rent ($)'})
        fig13.update_layout(height=420)
        st.plotly_chart(fig13, use_container_width=True, config=static_config)

        col_g, col_h = st.columns(2)

        # Chart 14: pets_allowed_bin × bedrooms vs price — Grouped Multi‑Bar
        with col_g:
            st.markdown("##### Chart 14 · Pet Policy × Bedrooms → Average Rent")
            df_p14 = df_filtered.copy()
            df_p14['Pet Policy'] = df_p14['pets_allowed_bin'].map({0: 'No Pets', 1: 'Pets Allowed'})
            agg14 = df_p14.groupby(['bedrooms', 'Pet Policy'], as_index=False)['price'].mean()
            fig14 = px.bar(agg14, x='bedrooms', y='price', color='Pet Policy', barmode='group',
                           labels={'price': 'Avg Price ($)', 'bedrooms': 'Bedrooms'},
                           color_discrete_sequence=['#EF553B', '#00CC96'])
            fig14.update_layout(height=420)
            st.plotly_chart(fig14, use_container_width=True, config=static_config)

        # Chart 15: bedrooms × Multi‑Attribute Profile — Radar Chart
        with col_h:
            st.markdown("##### Chart 15 · Multi‑Attribute Profile by Bedrooms (Radar)")
            bed_vals = sorted(df_filtered['bedrooms'].dropna().unique())
            bed_vals = [b for b in bed_vals if b > 0][:5]  # first 5 non‑zero
            radar_attrs = ['price', 'square_feet', 'bathrooms', 'amenities_count']
            radar_labels = ['Price', 'Sq Ft', 'Bathrooms', 'Amenities']

            agg_radar = df_filtered[df_filtered['bedrooms'].isin(bed_vals)] \
                .groupby('bedrooms')[radar_attrs].mean()

            # Normalise each attribute to 0‑1
            radar_norm = agg_radar.copy()
            for rc in radar_attrs:
                rc_min, rc_max = radar_norm[rc].min(), radar_norm[rc].max()
                if rc_max > rc_min:
                    radar_norm[rc] = (radar_norm[rc] - rc_min) / (rc_max - rc_min)
                else:
                    radar_norm[rc] = 0.5

            fig15 = go.Figure()
            radar_colors = px.colors.qualitative.Set2
            for idx, bed in enumerate(bed_vals):
                vals = radar_norm.loc[bed].tolist()
                vals.append(vals[0])  # close polygon
                fig15.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=radar_labels + [radar_labels[0]],
                    fill='toself',
                    name=f'{int(bed)} Bed',
                    line_color=radar_colors[idx % len(radar_colors)],
                    opacity=0.7,
                ))
            fig15.update_layout(
                height=420,
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            )
            st.plotly_chart(fig15, use_container_width=True, config=static_config)


# ═══════════════════════════════════════════════
#  TAB 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════
with tab2:
    st.header("🤖 Model Comparison")

    # ── 2a. Regression Metrics Summary Table ──────────────────
    st.subheader("Regression Metrics Summary")
    metrics_rows = []
    for name, info in models_dict.items():
        m = info['metrics']
        metrics_rows.append({
            'Model': name,
            'R² Score': round(m['R²'], 4),
            'MAE ($)': round(m['MAE'], 2),
            'RMSE ($)': round(m['RMSE'], 2),
        })
    df_metrics = pd.DataFrame(metrics_rows)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # ── 2b. Multi‑Metric Score Comparison Bar Chart ──────────
    st.subheader("Multi‑Metric Score Comparison Bar Chart")
    df_melt = df_metrics.melt(id_vars='Model', var_name='Metric', value_name='Value')
    fig_multi = px.bar(df_melt, x='Model', y='Value', color='Metric', barmode='group',
                       text_auto='.2f',
                       color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'])
    fig_multi.update_layout(height=450, yaxis_title="Score / Error")
    st.plotly_chart(fig_multi, use_container_width=True, config=static_config)

    st.divider()

    # ── 2c. Classification Metrics (Price Quartiles) ─────────
    st.subheader("Classification Metrics on Price Quartiles")
    st.caption("Prices are split into 4 equal‑frequency quartiles (25 % each). "
               "Predicted prices are binned with the same edges to compute classification metrics.")

    # Overall classification table for all models
    cls_rows = []
    for name, info in models_dict.items():
        labels, yt_bin, yp_bin, _ = price_quartile_metrics(info['y_test'], info['y_pred'])
        cls_rows.append({
            'Model': name,
            'Accuracy': round(accuracy_score(yt_bin, yp_bin), 4),
            'F1 Score (weighted)': round(f1_score(yt_bin, yp_bin, average='weighted', zero_division=0), 4),
            'Precision (weighted)': round(precision_score(yt_bin, yp_bin, average='weighted', zero_division=0), 4),
            'Recall (weighted)': round(recall_score(yt_bin, yp_bin, average='weighted', zero_division=0), 4),
        })
    df_cls = pd.DataFrame(cls_rows)
    st.dataframe(df_cls, use_container_width=True, hide_index=True)

    st.divider()

    # ── 2d. Inspect Individual Model ─────────────────────────
    st.subheader("🔎 Inspect Individual Model")
    model_choice = st.selectbox("Select a model to inspect", list(models_dict.keys()))

    info = models_dict[model_choice]
    labels, yt_bin, yp_bin, q_edges = price_quartile_metrics(info['y_test'], info['y_pred'])

    # Actual vs Predicted Scatter
    st.markdown("#### Actual vs Predicted Price Scatter Plot")
    df_scatter = pd.DataFrame({'Actual': info['y_test'], 'Predicted': info['y_pred']})
    if len(df_scatter) > 5000:
        df_scatter = df_scatter.sample(5000, random_state=42)
    fig_ap = px.scatter(df_scatter, x='Actual', y='Predicted', opacity=0.35,
                        labels={'Actual': 'Actual Price ($)', 'Predicted': 'Predicted Price ($)'},
                        color_discrete_sequence=['#AB63FA'])
    # Perfect prediction line
    line_min = min(df_scatter['Actual'].min(), df_scatter['Predicted'].min())
    line_max = max(df_scatter['Actual'].max(), df_scatter['Predicted'].max())
    fig_ap.add_trace(go.Scatter(x=[line_min, line_max], y=[line_min, line_max],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
    fig_ap.update_layout(height=500)
    st.plotly_chart(fig_ap, use_container_width=True, config=static_config)

    # Class‑level metrics table
    st.markdown("#### Class‑Level Metrics (per Price Quartile)")
    report = classification_report(yt_bin, yp_bin, output_dict=True, zero_division=0)
    class_rows = []
    for lbl in labels:
        if lbl in report:
            r = report[lbl]
            class_rows.append({
                'Price Range': lbl,
                'Precision': round(r['precision'], 4),
                'Recall': round(r['recall'], 4),
                'F1 Score': round(r['f1-score'], 4),
                'Support': int(r['support']),
            })
    df_class = pd.DataFrame(class_rows)
    st.dataframe(df_class, use_container_width=True, hide_index=True)

    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(yt_bin, yp_bin, labels=labels)
    fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, aspect='auto',
                       color_continuous_scale='Blues',
                       labels={'x': 'Predicted Quartile', 'y': 'Actual Quartile', 'color': 'Count'})
    fig_cm.update_layout(height=500)
    st.plotly_chart(fig_cm, use_container_width=True, config=static_config)


# ═══════════════════════════════════════════════
#  TAB 3 — DATA EXPLORER
# ═══════════════════════════════════════════════
with tab3:
    st.header("📋 Data Explorer")

    # KPI cards
    feature_cols_display = ['bedrooms', 'bathrooms', 'square_feet', 'amenities_count', 'pets_allowed_bin', 'state']
    num_features = len(feature_cols_display)

    # Price quartile labels for display
    q_edges_full = np.quantile(df_explore['price'].dropna(), [0, 0.25, 0.50, 0.75, 1.0])
    q_labels_full = _make_labels(np.unique(q_edges_full))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Records", f"{len(df_explore):,}")
    c2.metric("🧮 Input Features", num_features)
    c3.metric("🏷️ Price Classes (Quartiles)", len(q_labels_full))
    c4.metric("🎯 Target Variable", "price")

    st.divider()

    # Price quartile breakdown
    st.subheader("Price Quartile Breakdown")
    df_q = df_explore.copy()
    df_q['Price Class'] = pd.cut(df_q['price'], bins=np.unique(q_edges_full),
                                  labels=q_labels_full, include_lowest=True)
    class_counts = df_q['Price Class'].value_counts().reset_index()
    class_counts.columns = ['Price Class', 'Count']
    st.dataframe(class_counts, use_container_width=True, hide_index=True)

    st.divider()

    # Dataset preview
    st.subheader("Dataset Preview")
    preview_n = st.slider("Number of rows to display", min_value=5, max_value=100, value=20, step=5)
    st.dataframe(df_explore.head(preview_n), use_container_width=True, hide_index=True)

    # Column info
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df_explore.columns,
        'Data Type': [str(d) for d in df_explore.dtypes],
        'Non‑Null Count': [df_explore[c].notna().sum() for c in df_explore.columns],
        'Null Count': [df_explore[c].isna().sum() for c in df_explore.columns],
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
#  TAB 4 — PRICE PREDICTOR
# ═══════════════════════════════════════════════
with tab4:
    st.header("🏠 Price Predictor")
    st.markdown("Enter apartment features and select a model to predict the estimated monthly rent.")

    # Input form
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

    # Model selector
    pred_model_name = st.selectbox("Choose Prediction Model",
                                    list(models_dict.keys()), key='pred_model')

    if st.button("🔮 Predict Rent", type="primary", use_container_width=True):
        input_data = {
            'bedrooms': pred_beds,
            'bathrooms': pred_baths,
            'pets_allowed_bin': 1 if pred_pets else 0,
            'amenities_count': len(pred_amenities),
            'square_feet': pred_sqft,
            'state': pred_state,
        }

        # Selected model prediction
        sel_info = models_dict[pred_model_name]
        sel_pred = sel_info['predict_fn'](sel_info['model'], sel_info['scaler'], sel_info['features'], input_data)

        st.success(f"### 🏷️ Estimated Monthly Rent ({pred_model_name}): **${sel_pred:,.2f}**")

        st.divider()

        # All‑model comparison
        st.subheader("All Models Comparison")
        all_preds = []
        for name, m_info in models_dict.items():
            p = m_info['predict_fn'](m_info['model'], m_info['scaler'], m_info['features'], input_data)
            all_preds.append({'Model': name, 'Predicted Rent ($)': f"${p:,.2f}", 'Value': p})
        df_all = pd.DataFrame(all_preds)

        st.dataframe(df_all[['Model', 'Predicted Rent ($)']], use_container_width=True, hide_index=True)

        fig_pred = px.bar(df_all, x='Model', y='Value', color='Model',
                          text='Predicted Rent ($)',
                          labels={'Value': 'Estimated Rent ($)'},
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pred.update_traces(textposition='outside')
        fig_pred.update_layout(height=420, yaxis_title="Estimated Rent ($)")
        st.plotly_chart(fig_pred, use_container_width=True, config=static_config)

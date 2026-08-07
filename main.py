import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Apartment Rental Price Prediction & Valuation System", page_icon="🏢", layout="wide")

# Import models
from model.lr import train_model as train_lr, predict_property as predict_lr
from model.knn import train_model as train_knn, predict_property as predict_knn
from model.dt import train_model as train_dt, predict_property as predict_dt
from model.rf import train_model as train_rf, predict_property as predict_rf


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("apartments_for_rent_fully_prepared.csv")
        
        # Clean price column
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_trained_models(df):
    results = {}
    
    # Train Linear Regression (Baseline)
    lr_model, lr_scaler, lr_features, lr_mae, lr_rmse, lr_r2 = train_lr(df)
    results['Linear Regression'] = {
        'model': lr_model, 'scaler': lr_scaler, 'features': lr_features, 'predict_fn': predict_lr,
        'metrics': {'MAE': lr_mae, 'RMSE': lr_rmse, 'R²': lr_r2}
    }
    
    # Train KNN
    knn_model, knn_scaler, knn_features, knn_mae, knn_rmse, knn_r2 = train_knn(df)
    results['KNN Regressor'] = {
        'model': knn_model, 'scaler': knn_scaler, 'features': knn_features, 'predict_fn': predict_knn,
        'metrics': {'MAE': knn_mae, 'RMSE': knn_rmse, 'R²': knn_r2}
    }
    
    # Train Decision Tree
    dt_model, dt_scaler, dt_features, dt_mae, dt_rmse, dt_r2 = train_dt(df)
    results['Decision Tree'] = {
        'model': dt_model, 'scaler': dt_scaler, 'features': dt_features, 'predict_fn': predict_dt,
        'metrics': {'MAE': dt_mae, 'RMSE': dt_rmse, 'R²': dt_r2}
    }
    
    # Train Random Forest
    rf_model, rf_scaler, rf_features, rf_mae, rf_rmse, rf_r2 = train_rf(df)
    results['Random Forest'] = {
        'model': rf_model, 'scaler': rf_scaler, 'features': rf_features, 'predict_fn': predict_rf,
        'metrics': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R²': rf_r2}
    }
    
    return results


# --- Main App ---
st.title("🏢 Apartment Rental Price Prediction & Valuation System")
st.markdown("Enter property features to predict the estimated monthly rental price using 4 regression models.")

df_raw = load_data()

if df_raw.empty:
    st.stop()
    
# Basic preprocessing for exploration
df_explore = df_raw.copy()
median_price = df_explore['price'].median()

if 'amenities_count' not in df_explore.columns:
    df_explore['amenities_count'] = 0
if 'pets_allowed_bin' not in df_explore.columns:
    df_explore['pets_allowed_bin'] = 0

for col in ['bedrooms', 'bathrooms']:
    df_explore[col] = pd.to_numeric(df_explore[col], errors='coerce').fillna(0)

if 'square_feet' in df_explore.columns:
    df_explore['square_feet'] = pd.to_numeric(df_explore['square_feet'], errors='coerce')
    df_explore['square_feet'] = df_explore['square_feet'].fillna(df_explore['square_feet'].median())

# --- Sidebar ---
st.sidebar.header("🏠 Apartment Features Input")
st.sidebar.markdown("Enter property details to predict rent price.")

input_state_options = sorted([str(s) for s in df_explore['state'].dropna().unique()])
input_state = st.sidebar.selectbox("State", input_state_options)
input_beds = st.sidebar.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
input_baths = st.sidebar.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
input_sqft = st.sidebar.number_input("Square Feet", min_value=100, max_value=5000, value=1000, step=50)
input_pets = st.sidebar.checkbox("Pets Allowed", value=True)
amenity_options = ["Gym", "Pool", "Parking", "Washer/Dryer", "AC", "Balcony"]
input_amenities = st.sidebar.multiselect("Amenities", amenity_options, default=["Parking", "AC"])

# Train models
with st.spinner("Training models... This may take a moment."):
    models_dict = get_trained_models(df_raw)

# --- Generate Predictions for Single Property ---
input_data = {
    'bedrooms': input_beds,
    'bathrooms': input_baths,
    'pets_allowed_bin': 1 if input_pets else 0,
    'amenities_count': len(input_amenities),
    'square_feet': input_sqft,
    'state': input_state
}

predictions = []
for name, m_info in models_dict.items():
    pred_val = m_info['predict_fn'](m_info['model'], m_info['scaler'], m_info['features'], input_data)
    predictions.append({
        'Model Name': name,
        'Predicted Price ($)': f"${pred_val:,.2f}",
        'Raw Value': pred_val
    })

df_preds = pd.DataFrame(predictions)
avg_price = df_preds['Raw Value'].mean()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["💰 Rent Valuation & Predictions", "📈 Data Exploration", "📊 Model Evaluation Scorecard"])

# Chart config to disable zoom/pan/toolbar
static_config = {'staticPlot': True, 'displayModeBar': False}

with tab1:
    st.header("Predicted Apartment Rental Valuation")

    # 1. Consensus Estimated Rent KPI
    st.metric(label="🏷️ Consensus Estimated Rent", value=f"${avg_price:,.2f}/month")
    st.caption("Average prediction across all 4 models.")
    
    st.divider()

    # 2. Model Price Comparison Bar Chart
    fig_bar = px.bar(
        df_preds, x='Model Name', y='Raw Value',
        color='Model Name', text='Predicted Price ($)',
        title="Predicted Monthly Rent Across Models",
        labels={'Raw Value': 'Estimated Rent ($)'}
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(height=450, yaxis_title="Estimated Rent ($)")
    st.plotly_chart(fig_bar, use_container_width=True, config=static_config)

    # 3. Breakdown Table
    st.subheader("Model Prediction Breakdown")
    st.dataframe(df_preds[['Model Name', 'Predicted Price ($)']], use_container_width=True)

with tab2:
    st.header("Data Exploration & Understanding")
    
    # Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Apartments", len(df_explore))
    c2.metric("Total Columns", len(df_raw.columns))
    c3.metric("Features Used", 6)  # beds, baths, pets, amenities, state, square_feet
    c4.metric("Median Price (Overall)", f"${median_price:,.2f}")
    
    st.divider()
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        # Scatter Plot: Square Feet vs Rent
        if 'square_feet' in df_explore.columns:
            fig_scatter = px.scatter(
                df_explore.dropna(subset=['square_feet', 'price']),
                x='square_feet', y='price', opacity=0.4,
                title="Square Feet vs Monthly Rent",
                labels={'square_feet': 'Square Feet', 'price': 'Monthly Rent ($)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    with col_e2:
        # Box Plot: State vs Rent (top 10 states)
        top_states = df_explore['state'].value_counts().nlargest(10).index
        df_top_states = df_explore[df_explore['state'].isin(top_states)]
        fig_box = px.box(
            df_top_states, x='state', y='price', color='state',
            title="Price Distribution across Top 10 States",
            labels={'state': 'State', 'price': 'Monthly Rent ($)'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Feature Correlation Matrix
    st.subheader("Feature Correlation Matrix")
    corr_cols = ['price', 'bedrooms', 'bathrooms', 'amenities_count', 'pets_allowed_bin', 'square_feet']
    corr_cols = [c for c in corr_cols if c in df_explore.columns]
    corr_matrix = df_explore[corr_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix of Numeric Features"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.header("Model Evaluation Scorecard")
    
    # 1. Model Scorecard Table
    metrics_data = []
    for name, m_info in models_dict.items():
        m = m_info['metrics']
        metrics_data.append({
            'Model': name,
            'MAE ($)': round(m['MAE'], 2),
            'RMSE ($)': round(m['RMSE'], 2),
            'R² Score': round(m['R²'], 4),
        })
    df_metrics = pd.DataFrame(metrics_data)
    
    st.subheader("Summary Scorecard")
    st.dataframe(df_metrics, use_container_width=True)
    
    st.divider()
    
    # 2. R² Score Comparison
    fig_r2 = px.bar(
        df_metrics, x='Model', y='R² Score', color='Model',
        title="R² Score Comparison (Higher is Better)", text='R² Score'
    )
    fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_r2.update_layout(yaxis_range=[0, max(df_metrics['R² Score'].max() * 1.2, 1.0)])
    st.plotly_chart(fig_r2, use_container_width=True, config=static_config)
    
    # 3. MAE Comparison
    fig_mae = px.bar(
        df_metrics, x='Model', y='MAE ($)', color='Model',
        title="MAE Comparison (Lower is Better)", text='MAE ($)'
    )
    fig_mae.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig_mae, use_container_width=True, config=static_config)

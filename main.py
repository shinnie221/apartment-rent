import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Apartment Rent Prediction", page_icon="🏢", layout="wide")

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
def get_trained_models(df, lr_alpha, knn_k, dt_depth, rf_estimators, rf_depth):
    results = {}
    
    # Train Ridge Regression
    lr_model, lr_scaler, lr_features, lr_mae, lr_rmse, lr_r2, lr_y_test, lr_y_pred = train_lr(df, alpha=lr_alpha)
    results['Ridge Regression'] = {
        'model': lr_model, 'scaler': lr_scaler, 'features': lr_features, 'predict_fn': predict_lr,
        'metrics': {'MAE': lr_mae, 'RMSE': lr_rmse, 'R²': lr_r2},
        'y_test': lr_y_test, 'y_pred': lr_y_pred
    }
    
    # Train KNN
    knn_model, knn_scaler, knn_features, knn_mae, knn_rmse, knn_r2, knn_y_test, knn_y_pred = train_knn(df, n_neighbors=knn_k)
    results['KNN'] = {
        'model': knn_model, 'scaler': knn_scaler, 'features': knn_features, 'predict_fn': predict_knn,
        'metrics': {'MAE': knn_mae, 'RMSE': knn_rmse, 'R²': knn_r2},
        'y_test': knn_y_test, 'y_pred': knn_y_pred
    }
    
    # Train Decision Tree
    dt_model, dt_scaler, dt_features, dt_mae, dt_rmse, dt_r2, dt_y_test, dt_y_pred = train_dt(df, max_depth=dt_depth)
    results['Decision Tree'] = {
        'model': dt_model, 'scaler': dt_scaler, 'features': dt_features, 'predict_fn': predict_dt,
        'metrics': {'MAE': dt_mae, 'RMSE': dt_rmse, 'R²': dt_r2},
        'y_test': dt_y_test, 'y_pred': dt_y_pred
    }
    
    # Train Random Forest
    rf_model, rf_scaler, rf_features, rf_mae, rf_rmse, rf_r2, rf_y_test, rf_y_pred = train_rf(df, n_estimators=rf_estimators, max_depth=rf_depth)
    results['Random Forest'] = {
        'model': rf_model, 'scaler': rf_scaler, 'features': rf_features, 'predict_fn': predict_rf,
        'metrics': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R²': rf_r2},
        'y_test': rf_y_test, 'y_pred': rf_y_pred
    }
    
    return results


# --- Main App ---
st.title("🏢 Apartment Rent Price Predictor")
st.markdown("Enter property features to predict the monthly rent price using multiple ML regression models.")

df_raw = load_data()

if df_raw.empty:
    st.stop()
    
# Basic preprocessing for exploration
df_explore = df_raw.copy()
median_price = df_explore['price'].median()

# is_high_price, amenities_count, pets_allowed_bin already exist in the prepared CSV
if 'is_high_price' not in df_explore.columns:
    df_explore['is_high_price'] = (df_explore['price'] > median_price).astype(int)
df_explore['price_category'] = df_explore['is_high_price'].map({1: 'High Price (Premium)', 0: 'Budget'})

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
st.sidebar.header("⚙️ Model Hyperparameters")
st.sidebar.markdown("Tune models dynamically.")
lr_alpha = st.sidebar.slider("Ridge Regression: Alpha (Regularization)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
knn_k = st.sidebar.slider("KNN: n_neighbors", min_value=1, max_value=25, value=5, step=1)
dt_depth = st.sidebar.slider("Decision Tree: max_depth", min_value=2, max_value=20, value=10, step=1)
rf_estimators = st.sidebar.slider("Random Forest: n_estimators", min_value=10, max_value=200, value=100, step=10)
rf_depth = st.sidebar.slider("Random Forest: max_depth", min_value=2, max_value=20, value=10, step=1)

st.sidebar.divider()

st.sidebar.header("🎯 Property Features (User Input)")
st.sidebar.markdown("Enter property details to predict rent price.")

input_state_options = sorted([str(s) for s in df_explore['state'].dropna().unique()])
input_state = st.sidebar.selectbox("State", input_state_options)
input_beds = st.sidebar.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
input_baths = st.sidebar.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
input_sqft = st.sidebar.number_input("Square Feet", min_value=100, max_value=10000, value=1000, step=100)
input_pets = st.sidebar.checkbox("Pets Allowed", value=True)
amenity_options = ["Gym", "Pool", "Parking", "Washer/Dryer", "AC", "Balcony"]
input_amenities = st.sidebar.multiselect("Amenities", amenity_options, default=["Parking", "AC"])

st.sidebar.divider()
st.sidebar.header("🔎 Search Apartments in Wider Dataset")
st.sidebar.markdown("Filter dataset for Data Exploration (Tab 2).")
filter_state = st.sidebar.selectbox("Filter State", ["All"] + input_state_options)
price_min, price_max = st.sidebar.slider("Price Range", float(df_explore['price'].min(skipna=True)), 10000.0, (500.0, 5000.0))
filter_beds = st.sidebar.slider("Minimum Bedrooms", 0, 10, 0)
filter_baths = st.sidebar.slider("Minimum Bathrooms", 0.0, 10.0, 0.0, step=0.5)

if st.sidebar.button("Reset Filters"):
    st.rerun()

# Apply Filters
df_filtered = df_explore.copy()
if filter_state != "All":
    df_filtered = df_filtered[df_filtered['state'] == filter_state]
df_filtered = df_filtered[(df_filtered['price'] >= price_min) & (df_filtered['price'] <= price_max)]
df_filtered = df_filtered[df_filtered['bedrooms'] >= filter_beds]
df_filtered = df_filtered[df_filtered['bathrooms'] >= filter_baths]

with st.spinner("Training models... This may take a moment."):
    models_dict = get_trained_models(df_raw, lr_alpha, knn_k, dt_depth, rf_estimators, rf_depth)

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
    predicted_price = m_info['predict_fn'](m_info['model'], m_info['scaler'], m_info['features'], input_data)
    
    predictions.append({
        'Model Name': name,
        'Predicted Price ($)': round(predicted_price, 2),
    })
    
df_preds = pd.DataFrame(predictions)
avg_predicted_price = df_preds['Predicted Price ($)'].mean()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Price Prediction", "📈 Data Exploration", "⚙️ Model Performance"])

with tab1:
    st.header("Predicted Rent Price")

    # Chart config to disable zoom/pan/toolbar
    static_config = {'staticPlot': True, 'displayModeBar': False}

    # 1. Predicted Price Metrics
    st.subheader("Model Predictions")
    cols = st.columns(len(predictions))
    for i, pred in enumerate(predictions):
        cols[i].metric(pred['Model Name'], f"${pred['Predicted Price ($)']:,.2f}")
    
    # Average predicted price
    st.metric("📊 Average Predicted Price (All Models)", f"${avg_predicted_price:,.2f}")
    st.caption(f"Market Median Price: ${median_price:,.2f}")

    # 2. Model Price Comparison Bar Chart
    with st.container():
        fig_bar = px.bar(
            df_preds, x='Model Name', y='Predicted Price ($)', 
            color='Model Name', text='Predicted Price ($)',
            title="Model Price Predictions Comparison"
        )
        fig_bar.add_hline(y=median_price, line_dash="dash", line_color="red", annotation_text=f"Median: ${median_price:,.0f}")
        fig_bar.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_bar.update_layout(height=400, yaxis_title="Predicted Price ($)")
        st.plotly_chart(fig_bar, use_container_width=True, config=static_config)

    # 3. Property Feature Profile vs Average Market Profile (Radar Chart)
    with st.container():
        avg_market = df_explore[['bedrooms', 'bathrooms', 'price', 'amenities_count', 'square_feet']].mean()
        
        categories = ['Bedrooms', 'Bathrooms', 'Predicted Price ($)', 'Amenities Count', 'Square Feet']
        fig_radar = go.Figure()
        
        market_vals = [1.0, 1.0, 1.0, 1.0, 1.0]
        prop_vals = [
            input_beds / (avg_market['bedrooms'] or 1),
            input_baths / (avg_market['bathrooms'] or 1),
            avg_predicted_price / (avg_market['price'] or 1),
            len(input_amenities) / (avg_market['amenities_count'] or 1),
            input_sqft / (avg_market['square_feet'] or 1)
        ]
        
        fig_radar.add_trace(go.Scatterpolar(r=market_vals, theta=categories, fill='toself', name='Market Average'))
        fig_radar.add_trace(go.Scatterpolar(r=prop_vals, theta=categories, fill='toself', name='Your Property'))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(prop_vals + [1.5])])),
            showlegend=True,
            title="Your Property vs Market Average (Normalized)"
        )
        st.plotly_chart(fig_radar, use_container_width=True, config=static_config)

    st.subheader("Detailed Model Predictions")
    st.dataframe(df_preds, use_container_width=True)

with tab2:
    st.header("Data Exploration & Understanding")
    st.caption("Using filtered dataset from sidebar.")
    
    # 1. Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Apartments (Filtered)", len(df_filtered))
    c2.metric("Total Columns", len(df_raw.columns))
    c3.metric("Available Features Used", 6) # beds, baths, pets, amenities, state, square_feet
    c4.metric("Median Price (Overall)", f"${median_price:,.2f}")
    
    st.divider()
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        # 2. Price Category Distribution
        dist_counts = df_filtered['price_category'].value_counts().reset_index()
        dist_counts.columns = ['Category', 'Count']
        fig_donut = px.pie(dist_counts, values='Count', names='Category', hole=0.5, title="Price Category Distribution")
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col_e2:
        # 4. Pets Allowed vs Price Category
        pets_dist = df_filtered.groupby(['pets_allowed_bin', 'price_category']).size().reset_index(name='Count')
        pets_dist['Pets Allowed'] = pets_dist['pets_allowed_bin'].map({1: 'Yes', 0: 'No'})
        fig_pets = px.bar(pets_dist, x='Pets Allowed', y='Count', color='price_category', barmode='group', title="Pets Allowed vs Price Category")
        st.plotly_chart(fig_pets, use_container_width=True)
        
    col_e3, col_e4 = st.columns(2)
    with col_e3:
        # Chart 7: Scatter Plot (Square Feet vs Monthly Price colored by Rent Category)
        if 'square_feet' in df_filtered.columns:
            fig_scatter = px.scatter(df_filtered, x='square_feet', y='price', color='price_category', opacity=0.6,
                                     title="Square Feet vs Monthly Price",
                                     labels={'square_feet': 'Square Feet', 'price': 'Monthly Price ($)'})
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    with col_e4:
        # Chart 8: Box Plot / Violin Plot (Price distribution across top States)
        top_states = df_filtered['state'].value_counts().nlargest(10).index
        df_top_states = df_filtered[df_filtered['state'].isin(top_states)]
        fig_box = px.box(df_top_states, x='state', y='price', color='state',
                         title="Price Distribution across Top 10 States",
                         labels={'state': 'State', 'price': 'Monthly Price ($)'})
        st.plotly_chart(fig_box, use_container_width=True)
        
    # Chart 6: Feature Correlation Matrix
    st.subheader("Feature Correlation Matrix")
    corr_cols = ['price', 'bedrooms', 'bathrooms', 'amenities_count', 'pets_allowed_bin', 'square_feet']
    # Filter only columns that exist
    corr_cols = [c for c in corr_cols if c in df_filtered.columns]
    corr_matrix = df_filtered[corr_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix of Numeric Features"
    )
    st.plotly_chart(fig_corr, use_container_width=True)


with tab3:
    st.header("Model Performance & Scorecard")
    
    # 1. Model Scorecard Table
    metrics_data = []
    for name, m_info in models_dict.items():
        m = m_info['metrics']
        metrics_data.append({
            'Model': name,
            'MAE ($)': round(m['MAE'], 2),
            'RMSE ($)': round(m['RMSE'], 2),
            'R²': round(m['R²'], 4),
        })
    df_metrics = pd.DataFrame(metrics_data)
    
    st.subheader("Model Scorecard Table")
    st.dataframe(df_metrics, use_container_width=True)
    
    # 2. Performance Comparison Chart
    st.subheader("Performance Comparison")
    df_melted = df_metrics.melt(id_vars=['Model'], value_vars=['MAE ($)', 'RMSE ($)'], var_name='Metric', value_name='Value ($)')
    
    fig_comp = px.bar(df_melted, x='Metric', y='Value ($)', color='Model', barmode='group', title="Error Metric Comparison across Models")
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # R² comparison
    fig_r2 = px.bar(df_metrics, x='Model', y='R²', color='Model', title="R² Score Comparison (Higher is Better)", text='R²')
    fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_r2.update_layout(yaxis_range=[0, max(df_metrics['R²'].max() * 1.2, 1.0)])
    st.plotly_chart(fig_r2, use_container_width=True)

    c_left, c_right = st.columns(2)
    
    with c_left:
        # 3. Actual vs Predicted Scatter Plot
        st.subheader("Actual vs Predicted")
        selected_model = st.selectbox("Select Model for Scatter Plot", list(models_dict.keys()))
        y_test = models_dict[selected_model]['y_test']
        y_pred = models_dict[selected_model]['y_pred']
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', opacity=0.5, name='Predictions'))
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                                          line=dict(dash='dash', color='red'), name='Perfect Prediction'))
        fig_scatter.update_layout(
            title=f"Actual vs Predicted: {selected_model}",
            xaxis_title="Actual Price ($)",
            yaxis_title="Predicted Price ($)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c_right:
        # 4. Feature Importance
        st.subheader("Feature Importance")
        fi_model_name = st.selectbox("Select Model for Feature Importance", ["Decision Tree", "Random Forest"])
        if fi_model_name in models_dict:
            model_obj = models_dict[fi_model_name]['model']
            if hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
                features = models_dict[fi_model_name]['features']
                df_fi = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
                
                # Keep top 15 features to avoid clutter
                df_fi = df_fi.tail(15)
                
                fig_fi = px.bar(df_fi, x='Importance', y='Feature', orientation='h', title=f"Top Features: {fi_model_name}")
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.warning("Selected model does not support feature importance calculation.")
                
    # 5. Residual Plot
    st.subheader("Residual Plot")
    selected_model_resid = st.selectbox("Select Model for Residual Plot", list(models_dict.keys()), key="resid_model")
    y_test_r = models_dict[selected_model_resid]['y_test']
    y_pred_r = models_dict[selected_model_resid]['y_pred']
    residuals = y_test_r - y_pred_r
    
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(x=y_pred_r, y=residuals, mode='markers', opacity=0.5, name='Residuals'))
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    fig_resid.update_layout(
        title=f"Residual Plot: {selected_model_resid}",
        xaxis_title="Predicted Price ($)",
        yaxis_title="Residual (Actual - Predicted) ($)",
    )
    st.plotly_chart(fig_resid, use_container_width=True)

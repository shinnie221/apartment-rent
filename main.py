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

# --- Helper Functions ---
@st.cache_data
def load_data():
    try:
        # Assuming delimiter is ';' based on the dataset snippet
        df = pd.read_csv("apartments_for_rent_cleaned.csv", sep=";", on_bad_lines='skip')
        
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
    
    # Train Logistic Regression
    lr_model, lr_scaler, lr_features, lr_acc, lr_prec, lr_rec, lr_f1, lr_roc, lr_thresh, lr_cm = train_lr(df)
    results['Logistic Regression'] = {
        'model': lr_model, 'scaler': lr_scaler, 'features': lr_features, 'predict_fn': predict_lr,
        'metrics': {'Accuracy': lr_acc, 'Precision': lr_prec, 'Recall': lr_rec, 'F1-Score': lr_f1, 'ROC-AUC': lr_roc, 'Threshold': lr_thresh}
    }
    
    # Train KNN
    knn_model, knn_scaler, knn_features, knn_acc, knn_prec, knn_rec, knn_f1, knn_roc, knn_thresh, knn_cm = train_knn(df)
    results['KNN'] = {
        'model': knn_model, 'scaler': knn_scaler, 'features': knn_features, 'predict_fn': predict_knn,
        'metrics': {'Accuracy': knn_acc, 'Precision': knn_prec, 'Recall': knn_rec, 'F1-Score': knn_f1, 'ROC-AUC': knn_roc, 'Threshold': knn_thresh}
    }
    
    # Train Decision Tree
    dt_model, dt_scaler, dt_features, dt_acc, dt_prec, dt_rec, dt_f1, dt_roc, dt_thresh, dt_cm = train_dt(df)
    results['Decision Tree'] = {
        'model': dt_model, 'scaler': dt_scaler, 'features': dt_features, 'predict_fn': predict_dt,
        'metrics': {'Accuracy': dt_acc, 'Precision': dt_prec, 'Recall': dt_rec, 'F1-Score': dt_f1, 'ROC-AUC': dt_roc, 'Threshold': dt_thresh}
    }
    
    # Train Random Forest
    rf_model, rf_scaler, rf_features, rf_acc, rf_prec, rf_rec, rf_f1, rf_roc, rf_thresh, rf_cm = train_rf(df)
    results['Random Forest'] = {
        'model': rf_model, 'scaler': rf_scaler, 'features': rf_features, 'predict_fn': predict_rf,
        'metrics': {'Accuracy': rf_acc, 'Precision': rf_prec, 'Recall': rf_rec, 'F1-Score': rf_f1, 'ROC-AUC': rf_roc, 'Threshold': rf_thresh}
    }
    
    return results

def count_amenities(x):
    if pd.isna(x) or str(x).strip().lower() == 'none':
        return 0
    return len(str(x).split(','))

# --- Main App ---
st.title("🏢 Apartment Rent Risk Predictor")
st.markdown("Analyze rental properties, predict high rent risk, and compare ML classification models.")

df_raw = load_data()

if df_raw.empty:
    st.stop()
    
# Basic preprocessing for exploration
df_explore = df_raw.copy()
median_price = df_explore['price'].median()
df_explore['is_high_price'] = (df_explore['price'] > median_price).astype(int)
df_explore['price_category'] = df_explore['is_high_price'].map({1: 'High Price (Premium)', 0: 'Budget'})
df_explore['amenities_count'] = df_explore['amenities'].apply(count_amenities)
df_explore['pets_allowed_bin'] = df_explore['pets_allowed'].apply(lambda x: 0 if pd.isna(x) or str(x).strip().lower() == 'none' else 1)

for col in ['bedrooms', 'bathrooms']:
    df_explore[col] = pd.to_numeric(df_explore[col], errors='coerce').fillna(0)

with st.spinner("Training models... This may take a moment on first run."):
    models_dict = get_trained_models(df_raw)

# --- Sidebar ---
st.sidebar.header("🎯 Single Property Inputs")
st.sidebar.markdown("Enter details to predict rent risk.")

input_state_options = sorted([str(s) for s in df_explore['state'].dropna().unique()])
input_state = st.sidebar.selectbox("State", input_state_options)
input_price = st.sidebar.number_input("Monthly Rent Price ($)", min_value=100, max_value=100000, value=2500, step=100)
input_beds = st.sidebar.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
input_baths = st.sidebar.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
input_pets = st.sidebar.checkbox("Pets Allowed", value=True)
amenity_options = ["Gym", "Pool", "Parking", "Washer/Dryer", "AC", "Balcony"]
input_amenities = st.sidebar.multiselect("Amenities", amenity_options, default=["Parking", "AC"])

st.sidebar.divider()
st.sidebar.header("🔎 Filtering Engine")
st.sidebar.markdown("Filter dataset for Data Exploration (Tab 2).")
filter_state = st.sidebar.selectbox("Filter State", ["All"] + input_state_options)
price_min, price_max = st.sidebar.slider("Price Range", float(df_explore['price'].min(skipna=True)), 10000.0, (500.0, 5000.0))
filter_beds = st.sidebar.slider("Minimum Bedrooms", 0, 10, 0)

if st.sidebar.button("Reset Filters"):
    # This will trigger a rerun and reset widget states naturally if we used session state, 
    # but for simplicity we just instruct the user it reruns
    st.rerun()

# Apply Filters
df_filtered = df_explore.copy()
if filter_state != "All":
    df_filtered = df_filtered[df_filtered['state'] == filter_state]
df_filtered = df_filtered[(df_filtered['price'] >= price_min) & (df_filtered['price'] <= price_max)]
df_filtered = df_filtered[df_filtered['bedrooms'] >= filter_beds]


# --- Generate Predictions for Single Property ---
input_data = {
    'bedrooms': input_beds,
    'bathrooms': input_baths,
    'pets_allowed_bin': 1 if input_pets else 0,
    'amenities_count': len(input_amenities),
    'state': input_state
}

predictions = []
for name, m_info in models_dict.items():
    pred, prob_pct, conf_pct = m_info['predict_fn'](m_info['model'], m_info['scaler'], m_info['features'], input_data)
    thresh = m_info['metrics']['Threshold']
    # Adjust recommendation based on default 0.5 baseline vs optimal
    rec = "High Risk" if prob_pct >= 50 else "Low Risk"
    predictions.append({
        'Model Name': name,
        'Prediction': pred,
        'High Price Probability %': round(prob_pct, 2),
        'Optimal Threshold': round(thresh, 4),
        'Confidence %': round(conf_pct, 2),
        'Recommendation': rec
    })
    
df_preds = pd.DataFrame(predictions)
avg_prob = df_preds['High Price Probability %'].mean()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Property Price / Rent Risk", "📈 Data Exploration", "⚙️ Model Performance"])

with tab1:
    st.header("Property Rent Risk Consensus")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 1. High Rent Risk Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_prob,
            title = {'text': "Average High Rent Risk (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "gold"},
                    {'range': [70, 100], 'color': "tomato"}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col2:
        # 2. Model Consensus Chart
        fig_bar = px.bar(
            df_preds, x='Model Name', y='High Price Probability %', 
            color='Model Name', text='High Price Probability %',
            title="Model Consensus: Probability of High Price"
        )
        fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_bar.update_layout(height=350, yaxis_range=[0, 110])
        st.plotly_chart(fig_bar, use_container_width=True)

    col3, col4 = st.columns([1, 1])
    with col3:
        # 3. Property Feature Profile vs Average Market Profile (Radar Chart)
        avg_market = df_explore[['bedrooms', 'bathrooms', 'price', 'amenities_count']].mean()
        
        categories = ['Bedrooms', 'Bathrooms', 'Price ($)', 'Amenities Count']
        fig_radar = go.Figure()
        
        # We need to scale values for a radar chart to make sense visually if they are on vastly different scales.
        # But we can just use multiple axes or just plot raw with log if needed.
        # Simplest is to normalize against market average (Market Avg = 1.0)
        market_vals = [1.0, 1.0, 1.0, 1.0]
        prop_vals = [
            input_beds / (avg_market['bedrooms'] or 1),
            input_baths / (avg_market['bathrooms'] or 1),
            input_price / (avg_market['price'] or 1),
            len(input_amenities) / (avg_market['amenities_count'] or 1)
        ]
        
        fig_radar.add_trace(go.Scatterpolar(r=market_vals, theta=categories, fill='toself', name='Market Average'))
        fig_radar.add_trace(go.Scatterpolar(r=prop_vals, theta=categories, fill='toself', name='Selected Property'))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(prop_vals + [1.5])])),
            showlegend=True,
            title="Property vs Market Average (Normalized)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col4:
        st.subheader("Detailed Model Predictions")
        st.dataframe(df_preds[['Model Name', 'Prediction', 'High Price Probability %', 'Optimal Threshold', 'Confidence %']], use_container_width=True)
        
        st.subheader("Model Decision Baseline Table")
        st.dataframe(df_preds[['Model Name', 'Optimal Threshold', 'Recommendation']].assign(**{'Default Baseline': 0.5}), use_container_width=True)

with tab2:
    st.header("Data Exploration & Understanding")
    st.caption("Using filtered dataset from sidebar.")
    
    # 1. Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Apartments (Filtered)", len(df_filtered))
    c2.metric("Total Columns", len(df_raw.columns))
    c3.metric("Available Features Used", 5) # beds, baths, pets, amenities, state
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
        
    # 3. Feature Correlation Matrix
    st.subheader("Feature Correlation Matrix")
    corr_cols = ['price', 'bedrooms', 'bathrooms', 'amenities_count', 'pets_allowed_bin', 'is_high_price']
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
            'Accuracy': round(m['Accuracy'], 4),
            'Precision': round(m['Precision'], 4),
            'Recall': round(m['Recall'], 4),
            'F1-Score': round(m['F1-Score'], 4),
            'ROC-AUC': round(m['ROC-AUC'], 4)
        })
    df_metrics = pd.DataFrame(metrics_data)
    
    st.subheader("Model Scorecard Table (Threshold = 0.5)")
    st.dataframe(df_metrics, use_container_width=True)
    
    # 2. Performance Comparison Chart
    st.subheader("Performance Comparison")
    df_melted = df_metrics.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], var_name='Metric', value_name='Score')
    
    fig_comp = px.bar(df_melted, x='Metric', y='Score', color='Model', barmode='group', title="Metric Comparison across Models")
    fig_comp.update_layout(yaxis_range=[0, 1.1])
    st.plotly_chart(fig_comp, use_container_width=True)

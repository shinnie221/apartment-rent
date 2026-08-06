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
def get_trained_models(df, lr_c, knn_k, dt_depth, rf_estimators, rf_depth, threshold):
    results = {}
    
    # Train Logistic Regression
    lr_model, lr_scaler, lr_features, lr_acc, lr_prec, lr_rec, lr_f1, lr_roc, lr_thresh, lr_cm, lr_fpr, lr_tpr = train_lr(df, C=lr_c, threshold=threshold)
    results['Logistic Regression'] = {
        'model': lr_model, 'scaler': lr_scaler, 'features': lr_features, 'predict_fn': predict_lr,
        'metrics': {'Accuracy': lr_acc, 'Precision': lr_prec, 'Recall': lr_rec, 'F1-Score': lr_f1, 'ROC-AUC': lr_roc, 'Threshold': lr_thresh},
        'cm': lr_cm, 'fpr': lr_fpr, 'tpr': lr_tpr
    }
    
    # Train KNN
    knn_model, knn_scaler, knn_features, knn_acc, knn_prec, knn_rec, knn_f1, knn_roc, knn_thresh, knn_cm, knn_fpr, knn_tpr = train_knn(df, n_neighbors=knn_k, threshold=threshold)
    results['KNN'] = {
        'model': knn_model, 'scaler': knn_scaler, 'features': knn_features, 'predict_fn': predict_knn,
        'metrics': {'Accuracy': knn_acc, 'Precision': knn_prec, 'Recall': knn_rec, 'F1-Score': knn_f1, 'ROC-AUC': knn_roc, 'Threshold': knn_thresh},
        'cm': knn_cm, 'fpr': knn_fpr, 'tpr': knn_tpr
    }
    
    # Train Decision Tree
    dt_model, dt_scaler, dt_features, dt_acc, dt_prec, dt_rec, dt_f1, dt_roc, dt_thresh, dt_cm, dt_fpr, dt_tpr = train_dt(df, max_depth=dt_depth, threshold=threshold)
    results['Decision Tree'] = {
        'model': dt_model, 'scaler': dt_scaler, 'features': dt_features, 'predict_fn': predict_dt,
        'metrics': {'Accuracy': dt_acc, 'Precision': dt_prec, 'Recall': dt_rec, 'F1-Score': dt_f1, 'ROC-AUC': dt_roc, 'Threshold': dt_thresh},
        'cm': dt_cm, 'fpr': dt_fpr, 'tpr': dt_tpr
    }
    
    # Train Random Forest
    rf_model, rf_scaler, rf_features, rf_acc, rf_prec, rf_rec, rf_f1, rf_roc, rf_thresh, rf_cm, rf_fpr, rf_tpr = train_rf(df, n_estimators=rf_estimators, max_depth=rf_depth, threshold=threshold)
    results['Random Forest'] = {
        'model': rf_model, 'scaler': rf_scaler, 'features': rf_features, 'predict_fn': predict_rf,
        'metrics': {'Accuracy': rf_acc, 'Precision': rf_prec, 'Recall': rf_rec, 'F1-Score': rf_f1, 'ROC-AUC': rf_roc, 'Threshold': rf_thresh},
        'cm': rf_cm, 'fpr': rf_fpr, 'tpr': rf_tpr
    }
    
    return results


# --- Main App ---
st.title("🏢 Apartment Rent Risk Predictor")
st.markdown("Analyze rental properties, predict high rent risk, and compare ML classification models.")

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
lr_c = st.sidebar.slider("Logistic Regression: C (Regularization)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
knn_k = st.sidebar.slider("KNN: n_neighbors", min_value=1, max_value=25, value=5, step=1)
dt_depth = st.sidebar.slider("Decision Tree: max_depth", min_value=2, max_value=20, value=10, step=1)
rf_estimators = st.sidebar.slider("Random Forest: n_estimators", min_value=10, max_value=200, value=100, step=10)
rf_depth = st.sidebar.slider("Random Forest: max_depth", min_value=2, max_value=20, value=10, step=1)
threshold = st.sidebar.slider("Probability Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.01)

st.sidebar.divider()

st.sidebar.header("🎯 Single Property Inputs")
st.sidebar.markdown("Enter details to predict rent risk.")

input_state_options = sorted([str(s) for s in df_explore['state'].dropna().unique()])
input_state = st.sidebar.selectbox("State", input_state_options)
input_price = st.sidebar.number_input("Monthly Rent Price ($)", min_value=100, max_value=100000, value=2500, step=100)
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
    models_dict = get_trained_models(df_raw, lr_c, knn_k, dt_depth, rf_estimators, rf_depth, threshold)

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
    _, prob_pct, conf_pct = m_info['predict_fn'](m_info['model'], m_info['scaler'], m_info['features'], input_data)
    opt_thresh = m_info['metrics']['Threshold']
    
    # Adjust prediction and recommendation based on user-selected threshold
    pred_val = 1 if (prob_pct / 100.0) >= threshold else 0
    rec = "High Risk" if (prob_pct / 100.0) >= threshold else "Low Risk"
    
    predictions.append({
        'Model Name': name,
        'Prediction': pred_val,
        'High Price Probability %': round(prob_pct, 2),
        'Optimal Threshold': round(opt_thresh, 4),
        'Confidence %': round(conf_pct, 2),
        'Recommendation': rec
    })
    
df_preds = pd.DataFrame(predictions)
avg_prob = df_preds['High Price Probability %'].mean()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Property Price / Rent Risk", "📈 Data Exploration", "⚙️ Model Performance"])

with tab1:
    st.header("Property Rent Risk Consensus")

    # Chart config to disable zoom/pan/toolbar
    static_config = {'staticPlot': True, 'displayModeBar': False}

    # 1. High Rent Risk Gauge
    with st.container():
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
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold * 100}
            }
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True, config=static_config)

    # 2. Model Consensus Chart
    with st.container():
        fig_bar = px.bar(
            df_preds, x='Model Name', y='High Price Probability %', 
            color='Model Name', text='High Price Probability %',
            title="Model Consensus: Probability of High Price"
        )
        fig_bar.add_hline(y=threshold * 100, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold:.2f}")
        fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')

        # Add input_price as a reference line on secondary y-axis
        fig_bar.add_trace(go.Scatter(
            x=df_preds['Model Name'], 
            y=[input_price] * len(df_preds),
            mode='lines', 
            line=dict(color='dodgerblue', width=2, dash='dot'),
            name=f'Input Price (${input_price:,.0f})',
            yaxis='y2'
        ))

        fig_bar.update_layout(
            height=350, 
            yaxis=dict(title='High Price Probability %', range=[0, 110]),
            yaxis2=dict(title='Price ($)', overlaying='y', side='right', showgrid=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config=static_config)

    # 3. Property Feature Profile vs Average Market Profile (Radar Chart)
    with st.container():
        avg_market = df_explore[['bedrooms', 'bathrooms', 'price', 'amenities_count', 'square_feet']].mean()
        
        categories = ['Bedrooms', 'Bathrooms', 'Price ($)', 'Amenities Count', 'Square Feet']
        fig_radar = go.Figure()
        
        market_vals = [1.0, 1.0, 1.0, 1.0, 1.0]
        prop_vals = [
            input_beds / (avg_market['bedrooms'] or 1),
            input_baths / (avg_market['bathrooms'] or 1),
            input_price / (avg_market['price'] or 1),
            len(input_amenities) / (avg_market['amenities_count'] or 1),
            input_sqft / (avg_market['square_feet'] or 1)
        ]
        
        fig_radar.add_trace(go.Scatterpolar(r=market_vals, theta=categories, fill='toself', name='Market Average'))
        fig_radar.add_trace(go.Scatterpolar(r=prop_vals, theta=categories, fill='toself', name='Selected Property'))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(prop_vals + [1.5])])),
            showlegend=True,
            title="Property vs Market Average (Normalized)"
        )
        st.plotly_chart(fig_radar, use_container_width=True, config=static_config)

    st.subheader("Detailed Model Predictions")
    st.dataframe(df_preds[['Model Name', 'Prediction', 'High Price Probability %', 'Optimal Threshold', 'Confidence %']], use_container_width=True)
    
    st.subheader(f"Model Decision Baseline Table (Threshold = {threshold:.2f})")
    st.dataframe(df_preds[['Model Name', 'Optimal Threshold', 'Recommendation']].assign(**{'User Threshold': threshold}), use_container_width=True)

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
    corr_cols = ['price', 'bedrooms', 'bathrooms', 'amenities_count', 'pets_allowed_bin', 'square_feet', 'is_high_price']
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
            'Accuracy': round(m['Accuracy'], 4),
            'Precision': round(m['Precision'], 4),
            'Recall': round(m['Recall'], 4),
            'F1-Score': round(m['F1-Score'], 4),
            'ROC-AUC': round(m['ROC-AUC'], 4)
        })
    df_metrics = pd.DataFrame(metrics_data)
    
    st.subheader(f"Model Scorecard Table (Threshold = {threshold:.2f})")
    st.dataframe(df_metrics, use_container_width=True)
    
    # 2. Performance Comparison Chart
    st.subheader("Performance Comparison")
    df_melted = df_metrics.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], var_name='Metric', value_name='Score')
    
    fig_comp = px.bar(df_melted, x='Metric', y='Score', color='Model', barmode='group', title="Metric Comparison across Models")
    fig_comp.update_layout(yaxis_range=[0, 1.1])
    st.plotly_chart(fig_comp, use_container_width=True)

    c_left, c_right = st.columns(2)
    
    with c_left:
        # 3. Confusion Matrix Heatmap
        st.subheader("Confusion Matrix Heatmap")
        selected_model = st.selectbox("Select Model for Confusion Matrix", list(models_dict.keys()))
        cm = models_dict[selected_model]['cm']
        
        # Ensure cm is correct orientation (True Label on Y, Predicted on X)
        fig_cm = px.imshow(
            cm, 
            text_auto=True, 
            color_continuous_scale="Blues", 
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=['Budget', 'Premium'], 
            y=['Budget', 'Premium'], 
            title=f"Confusion Matrix: {selected_model}"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

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
                
    # 5. ROC Curve for all models
    st.subheader("ROC Curves Comparison")
    fig_roc = go.Figure()
    for name, m_info in models_dict.items():
        fig_roc.add_trace(go.Scatter(x=m_info['fpr'], y=m_info['tpr'], mode='lines', name=f"{name} (AUC={m_info['metrics']['ROC-AUC']:.2f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Guess (AUC=0.50)'))
    fig_roc.update_layout(title="ROC Curve for All Models", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", hovermode="x unified")
    st.plotly_chart(fig_roc, use_container_width=True)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix, roc_auc_score
from .pipeline import preprocess_data

def train_model(df, threshold=0.5):
    # Preprocess
    df_processed = preprocess_data(df)
    
    # Define features
    feature_cols = ['bedrooms', 'bathrooms', 'pets_allowed_bin', 'amenities_count', 'square_feet']
    state_cols = [col for col in df_processed.columns if col.startswith('state_')]
    feature_cols.extend(state_cols)
    
    # Filter out missing target/features
    df_processed = df_processed.dropna(subset=feature_cols + ['is_high_price'])
    
    X = df_processed[feature_cols]
    y = df_processed['is_high_price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predict on test
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # ROC curve to find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    # Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Predictions using user-selected threshold for standard metrics
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return model, scaler, feature_cols, accuracy, precision, recall, f1, roc_auc, optimal_threshold, cm, fpr, tpr

def predict_property(model, scaler, feature_names, input_data):
    """
    input_data should be a dict like:
    {'bedrooms': 2, 'bathrooms': 1, 'pets_allowed_bin': 1, 'amenities_count': 3, 'state': 'CA'}
    """
    # Create a DataFrame for the input
    df_in = pd.DataFrame([input_data])
    
    # Initialize all required features to 0
    X_in = pd.DataFrame(0, index=np.arange(1), columns=feature_names)
    
    # Fill in values
    if 'bedrooms' in df_in.columns: X_in['bedrooms'] = df_in['bedrooms'].values[0]
    if 'bathrooms' in df_in.columns: X_in['bathrooms'] = df_in['bathrooms'].values[0]
    if 'pets_allowed_bin' in df_in.columns: X_in['pets_allowed_bin'] = df_in['pets_allowed_bin'].values[0]
    if 'amenities_count' in df_in.columns: X_in['amenities_count'] = df_in['amenities_count'].values[0]
    if 'square_feet' in df_in.columns: X_in['square_feet'] = df_in['square_feet'].values[0]
    
    # Set state
    if 'state' in df_in.columns:
        state_col = f"state_{df_in['state'].values[0]}"
        if state_col in X_in.columns:
            X_in[state_col] = 1
            
    # Scale
    X_in_scaled = scaler.transform(X_in)
    
    # Predict
    proba = model.predict_proba(X_in_scaled)[0]
    prob_class_1 = proba[1]
    
    # Use 0.5 as default threshold for prediction here (can be overridden by caller)
    prediction = 1 if prob_class_1 >= 0.5 else 0
    
    probability_pct = prob_class_1 * 100
    confidence_pct = max(proba) * 100
    
    return prediction, probability_pct, confidence_pct

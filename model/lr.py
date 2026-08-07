import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .pipeline import preprocess_data

def train_model(df):
    # Preprocess
    df_processed = preprocess_data(df)
    
    # Define features
    feature_cols = ['bedrooms', 'bathrooms', 'pets_allowed_bin', 'amenities_count', 'square_feet']
    state_cols = [col for col in df_processed.columns if col.startswith('state_')]
    feature_cols.extend(state_cols)
    
    # Filter out missing target/features
    df_processed = df_processed.dropna(subset=feature_cols + ['price'])
    
    X = df_processed[feature_cols]
    y = df_processed['price']  # Continuous Target Variable
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predict on test
    y_pred = model.predict(X_test_scaled)
    
    # Regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, feature_cols, mae, rmse, r2

def predict_property(model, scaler, feature_names, input_data):
    """
    input_data should be a dict like:
    {'bedrooms': 2, 'bathrooms': 1, 'pets_allowed_bin': 1, 'amenities_count': 3, 'square_feet': 1000, 'state': 'CA'}
    Returns the predicted monthly rental price in USD.
    """
    # Create a DataFrame for the input
    df_in = pd.DataFrame([input_data])
    
    # Initialize all required features to 0
    X_in = pd.DataFrame(0, index=np.arange(1), columns=feature_names)
    
    # Fill in values
    for col in ['bedrooms', 'bathrooms', 'pets_allowed_bin', 'amenities_count', 'square_feet']:
        if col in df_in.columns:
            X_in[col] = df_in[col].values[0]
    
    # Set state
    if 'state' in df_in.columns:
        state_col = f"state_{df_in['state'].values[0]}"
        if state_col in X_in.columns:
            X_in[state_col] = 1
            
    # Scale
    X_in_scaled = scaler.transform(X_in)
    
    # Predict
    predicted_rent = model.predict(X_in_scaled)[0]
    
    return max(0.0, float(predicted_rent))  # Ensure valid non-negative rent

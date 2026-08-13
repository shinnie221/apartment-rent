import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .pipeline import preprocess_data

def train_model(df, max_depth=16, max_leaf_nodes=256):
    df_processed = preprocess_data(df)
    
    feature_cols = [
        'bedrooms', 'bathrooms', 'pets_allowed_bin', 'amenities_count', 'square_feet',
        'latitude', 'longitude', 'sqft_per_room', 'bath_bed_ratio', 'city_mean_price'
    ]
    state_cols = [col for col in df_processed.columns if col.startswith('state_')]
    feature_cols.extend(state_cols)
    
    df_processed = df_processed.dropna(subset=feature_cols + ['price'])
    
    X = df_processed[feature_cols]
    y = df_processed['price']  # Continuous Target Variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = DecisionTreeRegressor(random_state=42, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    # Regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, feature_cols, mae, rmse, r2, y_test, y_pred

def predict_property(model, scaler, feature_names, input_data):
    """
    input_data should be a dict like:
    {'bedrooms': 2, 'bathrooms': 1, 'pets_allowed_bin': 1, 'amenities_count': 3, 'square_feet': 1000, 'state': 'CA'}
    Returns the predicted monthly rental price in USD.
    """
    df_in = pd.DataFrame([input_data])
    X_in = pd.DataFrame(0.0, index=np.arange(1), columns=feature_names)
    
    beds = float(df_in.get('bedrooms', [2])[0] if 'bedrooms' in df_in.columns else 2)
    baths = float(df_in.get('bathrooms', [1])[0] if 'bathrooms' in df_in.columns else 1)
    sqft = float(df_in.get('square_feet', [1000])[0] if 'square_feet' in df_in.columns else 1000)
    
    df_in['sqft_per_room'] = sqft / (beds + baths + 1)
    df_in['bath_bed_ratio'] = baths / (beds + 1)
    if 'latitude' not in df_in.columns:
        df_in['latitude'] = 37.0
    if 'longitude' not in df_in.columns:
        df_in['longitude'] = -95.0
    if 'city_mean_price' not in df_in.columns:
        df_in['city_mean_price'] = float(input_data.get('city_mean_price', 1500.0))
        
    for col in feature_names:
        if col in df_in.columns:
            X_in[col] = float(df_in[col].values[0])
            
    if 'state' in df_in.columns:
        state_col = f"state_{df_in['state'].values[0]}"
        if state_col in X_in.columns:
            X_in[state_col] = 1.0
            
    X_in_scaled = scaler.transform(X_in)
    predicted_rent = model.predict(X_in_scaled)[0]
    
    return max(0.0, float(predicted_rent))  # Ensure valid non-negative rent

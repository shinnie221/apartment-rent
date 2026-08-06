import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .pipeline import preprocess_data

def train_model(df, max_depth=10):
    df_processed = preprocess_data(df)
    
    feature_cols = ['bedrooms', 'bathrooms', 'pets_allowed_bin', 'amenities_count', 'square_feet']
    state_cols = [col for col in df_processed.columns if col.startswith('state_')]
    feature_cols.extend(state_cols)
    
    df_processed = df_processed.dropna(subset=feature_cols + ['price'])
    
    X = df_processed[feature_cols]
    y = df_processed['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = DecisionTreeRegressor(random_state=42, max_depth=max_depth)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    # Regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, feature_cols, mae, rmse, r2, y_test.values, y_pred

def predict_property(model, scaler, feature_names, input_data):
    df_in = pd.DataFrame([input_data])
    X_in = pd.DataFrame(0, index=np.arange(1), columns=feature_names)
    
    if 'bedrooms' in df_in.columns: X_in['bedrooms'] = df_in['bedrooms'].values[0]
    if 'bathrooms' in df_in.columns: X_in['bathrooms'] = df_in['bathrooms'].values[0]
    if 'pets_allowed_bin' in df_in.columns: X_in['pets_allowed_bin'] = df_in['pets_allowed_bin'].values[0]
    if 'amenities_count' in df_in.columns: X_in['amenities_count'] = df_in['amenities_count'].values[0]
    if 'square_feet' in df_in.columns: X_in['square_feet'] = df_in['square_feet'].values[0]
    
    if 'state' in df_in.columns:
        state_col = f"state_{df_in['state'].values[0]}"
        if state_col in X_in.columns:
            X_in[state_col] = 1
            
    X_in_scaled = scaler.transform(X_in)
    
    predicted_price = model.predict(X_in_scaled)[0]
    
    return predicted_price

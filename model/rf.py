import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .pipeline import preprocess_data

def train_model(df):
    df_processed = preprocess_data(df)
    
    feature_cols = [
        'bedrooms', 'bathrooms', 'pets_allowed_bin', 'amenities_count', 'square_feet',
        'log_sqft', 'total_rooms', 'latitude', 'longitude', 'sqft_per_room', 'bath_bed_ratio', 'city_mean_price'
    ]
    state_cols = [col for col in df_processed.columns if col.startswith('state_')]
    feature_cols.extend(state_cols)
    
    df_processed = df_processed.dropna(subset=feature_cols + ['price'])
    
    X = df_processed[feature_cols]
    y = df_processed['price']
    y_log = np.log1p(y)
    
    X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
        X, y, y_log, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ── RandomizedSearchCV for Random Forest ──
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5],
        'max_samples': [0.6, 0.8, 1.0],
    }
    
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )
    random_search.fit(X_train_scaled, y_train_log)
    
    model = random_search.best_estimator_
    print(f"[Random Forest] Best Hyperparameters: {random_search.best_params_}")
    
    # Store best params on model for reporting
    model.best_params_ = random_search.best_params_
    model.cv_results_summary_ = random_search.cv_results_
    
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.maximum(0.0, np.expm1(y_pred_log))
    
    # Regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, feature_cols, mae, rmse, r2, y_test, y_pred

def predict_property(model, scaler, feature_names, input_data):

    df_in = pd.DataFrame([input_data])
    X_in = pd.DataFrame(0.0, index=np.arange(1), columns=feature_names)
    
    beds = float(df_in.get('bedrooms', [2])[0] if 'bedrooms' in df_in.columns else 2)
    baths = float(df_in.get('bathrooms', [1])[0] if 'bathrooms' in df_in.columns else 1)
    sqft = float(df_in.get('square_feet', [1000])[0] if 'square_feet' in df_in.columns else 1000)
    
    df_in['sqft_per_room'] = sqft / (beds + baths + 1)
    df_in['bath_bed_ratio'] = baths / (beds + 1)
    df_in['log_sqft'] = np.log1p(sqft)
    df_in['total_rooms'] = beds + baths
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
    predicted_rent_log = model.predict(X_in_scaled)[0]
    predicted_rent = np.expm1(predicted_rent_log)
    
    return max(0.0, float(predicted_rent))  # Ensure valid non-negative rent

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .pipeline import preprocess_data

def train_model(df, max_eval_samples=3000):
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
    
    # ── GridSearchCV for KNN ──
    # Subsample training data for faster CV if dataset is large
    if len(X_train_scaled) > 15000:
        np.random.seed(42)
        subsample_idx = np.random.choice(len(X_train_scaled), size=15000, replace=False)
        X_train_cv = X_train_scaled[subsample_idx]
        y_train_cv = y_train_log.iloc[subsample_idx]
    else:
        X_train_cv = X_train_scaled
        y_train_cv = y_train_log
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }
    
    grid_search = GridSearchCV(
        estimator=KNeighborsRegressor(algorithm='auto', n_jobs=-1),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train_cv, y_train_cv)
    
    best_params = grid_search.best_params_
    print(f"[KNN Regressor] Best Hyperparameters: {best_params}")
    
    # Refit best model on full training data
    model = KNeighborsRegressor(
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
        metric=best_params['metric'],
        algorithm='auto',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train_log)
    
    # Store best params on model for reporting
    model.best_params_ = best_params
    model.cv_results_summary_ = grid_search.cv_results_
    
    # Subsample test set for fast metrics calculation if test set is large
    if len(X_test_scaled) > max_eval_samples:
        np.random.seed(42)
        eval_idx = np.random.choice(len(X_test_scaled), size=max_eval_samples, replace=False)
        X_eval = X_test_scaled[eval_idx]
        y_eval = y_test.iloc[eval_idx]
    else:
        X_eval = X_test_scaled
        y_eval = y_test
        
    y_pred_log = model.predict(X_eval)
    y_pred = np.maximum(0.0, np.expm1(y_pred_log))
    
    # Regression metrics
    mae = mean_absolute_error(y_eval, y_pred)
    rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
    r2 = r2_score(y_eval, y_pred)
    
    return model, scaler, feature_cols, mae, rmse, r2, y_eval, y_pred

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

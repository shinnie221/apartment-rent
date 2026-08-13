import pandas as pd
import numpy as np

def preprocess_data(df, clip_outliers=True):
    df = df.copy()
    
    # Clean price and filter extreme outliers (top 0.5% listing anomalies)
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        if clip_outliers and len(df) > 100:
            q995 = df['price'].quantile(0.995)
            df = df[df['price'] <= q995]

    # Clean bedrooms & bathrooms
    for col in ['bedrooms', 'bathrooms']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    # Clean square_feet
    if 'square_feet' in df.columns:
        df['square_feet'] = pd.to_numeric(df['square_feet'], errors='coerce')
        median_sqft = df['square_feet'].median() if len(df['square_feet'].dropna()) > 0 else 1000
        df['square_feet'] = df['square_feet'].fillna(median_sqft)
    else:
        df['square_feet'] = 1000

    # Clean spatial coordinates: latitude & longitude
    for col, default_val in [('latitude', 37.0), ('longitude', -95.0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median() if len(df[col].dropna()) > 0 else default_val
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = default_val

    # amenities_count
    if 'amenities_count' not in df.columns:
        if 'amenities' in df.columns:
            s_amenities = df['amenities'].fillna('')
            df['amenities_count'] = np.where(
                s_amenities.str.strip().str.lower().isin(['', 'none']),
                0,
                s_amenities.str.count(',') + 1
            )
        else:
            df['amenities_count'] = 0
    
    # pets_allowed_bin
    if 'pets_allowed_bin' not in df.columns:
        if 'pets_allowed' in df.columns:
            s_pets = df['pets_allowed'].fillna('')
            df['pets_allowed_bin'] = np.where(
                s_pets.str.strip().str.lower().isin(['', 'none']),
                0,
                1
            )
        else:
            df['pets_allowed_bin'] = 0

    # Engineered Feature Ratios
    df['sqft_per_room'] = df['square_feet'] / (df['bedrooms'] + df['bathrooms'] + 1)
    df['bath_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)

    # City Mean Price Target Encoding
    if 'cityname' in df.columns and 'price' in df.columns:
        city_means = df.groupby('cityname')['price'].transform('mean')
        global_mean = df['price'].mean() if len(df) > 0 else 1500.0
        df['city_mean_price'] = city_means.fillna(global_mean)
    elif 'city_mean_price' not in df.columns:
        df['city_mean_price'] = df['price'].mean() if 'price' in df.columns and len(df) > 0 else 1500.0

    # State one-hot dummy encoding
    if 'state' in df.columns:
        state_dummies = pd.get_dummies(df['state'], prefix='state')
        df = pd.concat([df, state_dummies], axis=1)
        
    return df

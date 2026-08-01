import pandas as pd

def preprocess_data(df):
    """Common preprocessing logic for the dataset."""
    df = df.copy()
    
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        median_price = df['price'].median()
        df['is_high_price'] = (df['price'] > median_price).astype(int)
    
    def count_amenities(x):
        if pd.isna(x) or str(x).strip().lower() == 'none':
            return 0
        return len(str(x).split(','))
    
    if 'amenities' in df.columns:
        df['amenities_count'] = df['amenities'].apply(count_amenities)
    else:
        df['amenities_count'] = 0
        
    def has_pets(x):
        if pd.isna(x) or str(x).strip().lower() == 'none':
            return 0
        return 1
        
    if 'pets_allowed' in df.columns:
        df['pets_allowed_bin'] = df['pets_allowed'].apply(has_pets)
    else:
        df['pets_allowed_bin'] = 0

    for col in ['bedrooms', 'bathrooms']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    if 'square_feet' in df.columns:
        df['square_feet'] = pd.to_numeric(df['square_feet'], errors='coerce')
        median_sqft = df['square_feet'].median()
        df['square_feet'] = df['square_feet'].fillna(median_sqft)
    else:
        df['square_feet'] = 0
            
    if 'state' in df.columns:
        state_dummies = pd.get_dummies(df['state'], prefix='state')
        df = pd.concat([df, state_dummies], axis=1)
        
    return df

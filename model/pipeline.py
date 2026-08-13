import pandas as pd
import numpy as np

def preprocess_data(df):

    df = df.copy()
    
    #Makes sure rent prices are proper numbers and throws away any rows that are missing a price
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
    
    #amenities_count: use existing column if present, otherwise derive from amenities text
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
    
    #pets_allowed_bin: use existing column if present, otherwise derive from pets_allowed text
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

    #turn bedrooms and bathrooms into numbers and get rid of any missing values
    for col in ['bedrooms', 'bathrooms']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    #take the square feet values and turn them into numbers, replacing any missing values with the median
    if 'square_feet' in df.columns:
        df['square_feet'] = pd.to_numeric(df['square_feet'], errors='coerce')
        median_sqft = df['square_feet'].median()
        df['square_feet'] = df['square_feet'].fillna(median_sqft)
    else:
        df['square_feet'] = 0

    #turning state names into numbers
    if 'state' in df.columns:
        state_dummies = pd.get_dummies(df['state'], prefix='state')
        df = pd.concat([df, state_dummies], axis=1)
        
    return df

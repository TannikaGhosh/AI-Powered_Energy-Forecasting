"""
Loads raw data, creates features: hour, day_of_week, lag_1h, rolling_mean_3h.
"""
import pandas as pd
import numpy as np

def load_and_preprocess(csv_path='../data/raw/energy_consumption.csv'):
    df = pd.read_csv(csv_path, index_col='Datetime', parse_dates=True)
    
    # Resample to hourly (already hourly, but ensures consistency)
    df = df.resample('h').mean()
    
    # Forward fill missing values (if any)
    df.ffill(inplace=True)
    
    # Feature engineering
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # 0=Monday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Lag features (previous hour's consumption)
    df['lag_1'] = df['Energy_kW'].shift(1)
    
    # Rolling mean (3h window)
    df['rolling_mean_3'] = df['Energy_kW'].rolling(window=3).mean()
    
    # Drop rows with NaN created by shift/rolling
    df.dropna(inplace=True)
    
    return df

if __name__ == '__main__':
    df = load_and_preprocess()
    print(df.head())
    print(f"✅ Preprocessed shape: {df.shape}")
    df.to_csv('../data/processed/energy_features.csv')

import pandas as pd
import numpy as np

def preprocess_advanced(csv_path='data/raw/advanced_energy_data.csv'):
    df = pd.read_csv(csv_path, index_col='Datetime', parse_dates=True)
    
    # Ensure hourly frequency
    df = df.resample('h').mean()
    df.ffill(inplace=True)
    
    # Time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Lag features (for energy and risks)
    for col in ['Energy_kW', 'PowerCut_Risk', 'ShortCircuit_Risk', 'ExcessElectricity_Risk']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag24'] = df[col].shift(24)  # same hour yesterday
    
    # Rolling means (3h, 6h)
    df['Energy_rolling_mean_3'] = df['Energy_kW'].rolling(3).mean()
    df['PowerCut_Risk_rolling_6'] = df['PowerCut_Risk'].rolling(6).mean()
    
    # Weather interactions
    df['temp_humidity_interaction'] = df['Temperature_C'] * df['Humidity_pct'] / 100
    
    # Drop NaN rows from shifting
    df.dropna(inplace=True)
    
    return df

if __name__ == '__main__':
    df = preprocess_advanced()
    print(f"Preprocessed shape: {df.shape}")
    print(df[['Energy_kW', 'PowerCut_Risk', 'CarbonCredit_USD', 'LowPollution_Flag']].head())
    df.to_csv('data/processed/advanced_features.csv')

"""
Generates synthetic hourly energy consumption data for 90 days.
Patterns: weekday peaks (7 AM-12 PM, 6-8 PM), lower on weekends, random noise.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_energy_data(start_date='2024-01-01', days=90, seed=42):
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, periods=days*24, freq='h')
    energy = []
    
    for dt in date_range:
        hour = dt.hour
        day_of_week = dt.dayofweek  # Mon=0, Sun=6
        is_weekend = day_of_week >= 5
        
        # Base consumption (kW)
        if is_weekend:
            base = 1.2
        else:
            base = 1.5
        
        # Daily pattern: morning peak (7-12), evening peak (18-20)
        if 7 <= hour <= 12:          # CHANGED: 7 AM to 12 PM
            pattern = 2.5
        elif 18 <= hour <= 20:
            pattern = 3.0
        elif 22 <= hour or hour <= 5:
            pattern = 0.8
        else:
            pattern = 1.2
        
        # Add random noise and slight weekly trend
        noise = np.random.normal(0, 0.2)
        trend = 0.005 * (dt - date_range[0]).days  # small increase over time
        
        consumption = (base + pattern) + noise + trend
        consumption = max(0.2, consumption)  # floor value
        energy.append(round(consumption, 2))
    
    df = pd.DataFrame({'Datetime': date_range, 'Energy_kW': energy})
    df.set_index('Datetime', inplace=True)
    return df

if __name__ == '__main__':
    df = generate_energy_data()
    df.to_csv('../data/raw/energy_consumption.csv')
    print("✅ Synthetic energy data saved to data/raw/energy_consumption.csv")

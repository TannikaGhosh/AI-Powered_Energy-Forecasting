"""
Generates synthetic data with:
- Hourly energy consumption (as before)
- Temperature (°C) and humidity (%)
- Risk scores (power cut, short circuit, excess electricity)
- Carbon credit earned (USD) based on avoided emissions
- Low pollution flag (True/False)
"""
import pandas as pd
import numpy as np
from datetime import datetime

def generate_advanced_energy_data(start_date='2024-01-01', days=90, seed=42):
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, periods=days*24, freq='h')
    records = []
    
    # Baseline carbon intensity (g CO2 per kWh) – varies by hour
    # Lower at night (renewable surplus), higher in evening (peak fossil)
    carbon_intensity_profile = {
        0: 150, 1: 140, 2: 130, 3: 120, 4: 110, 5: 120,
        6: 180, 7: 220, 8: 250, 9: 260, 10: 270, 11: 280,
        12: 290, 13: 300, 14: 310, 15: 320, 16: 350,
        17: 400, 18: 450, 19: 480, 20: 460, 21: 400,
        22: 300, 23: 200
    }
    
    for dt in date_range:
        hour = dt.hour
        day_of_week = dt.dayofweek
        is_weekend = day_of_week >= 5
        
        # --- Energy consumption (same logic, but extended morning peak) ---
        if is_weekend:
            base = 1.2
        else:
            base = 1.5
        
        if 7 <= hour <= 12:
            pattern = 2.5
        elif 18 <= hour <= 20:
            pattern = 3.0
        elif 22 <= hour or hour <= 5:
            pattern = 0.8
        else:
            pattern = 1.2
        
        noise = np.random.normal(0, 0.2)
        trend = 0.005 * (dt - date_range[0]).days
        energy = max(0.2, (base + pattern) + noise + trend)
        energy = round(energy, 2)
        
        # --- Weather simulation (temperature & humidity) ---
        # Temperature: hotter in afternoon, colder at night; seasonal trend
        temp_base = 15 + 10 * np.sin(2 * np.pi * (dt.timetuple().tm_yday - 80) / 365)  # seasonal
        temp_daily = 8 * np.sin(2 * np.pi * (hour - 14) / 24)  # peak at 2 PM
        temperature = round(temp_base + temp_daily + np.random.normal(0, 2), 1)
        
        # Humidity: inversely related to temperature (simple)
        humidity = round(60 - 0.3 * temperature + np.random.normal(0, 8), 1)
        humidity = max(20, min(95, humidity))
        
        # --- Risk factors (based on energy load and weather extremes) ---
        # Power cut probability: high when energy > 4.5 kW OR extreme temp > 35°C
        power_cut_risk = 0.0
        if energy > 4.5:
            power_cut_risk += 0.4
        if temperature > 35:
            power_cut_risk += 0.3
        if hour in [18,19,20]:  # evening peak
            power_cut_risk += 0.2
        power_cut_risk = min(0.95, power_cut_risk + np.random.uniform(-0.05, 0.05))
        
        # Short circuit risk: older equipment simulation (increases over time)
        short_circuit_risk = 0.02 + 0.0005 * (dt - date_range[0]).days
        short_circuit_risk += 0.1 if humidity > 80 else 0
        short_circuit_risk = min(0.5, short_circuit_risk)
        
        # Excess electricity risk (when production > demand, only relevant for grids with renewables)
        excess_risk = 0.0
        if energy < 1.0 and hour > 10 and hour < 16:  # low demand, sunny hours
            excess_risk = 0.3
        excess_risk += np.random.uniform(0, 0.1)
        excess_risk = min(0.8, excess_risk)
        
        # --- Carbon credit calculation ---
        # Baseline: assume without forecasting, energy would be 20% higher (wastage)
        baseline_energy = energy * 1.2
        carbon_intensity = carbon_intensity_profile.get(hour, 300)
        co2_saved_kg = (baseline_energy - energy) * carbon_intensity / 1000
        # Carbon credit price: $20 per ton CO2 (typical market price)
        carbon_credit_usd = round(co2_saved_kg * 20 / 1000, 4)
        
        # --- Low pollution check (True if carbon intensity < 200 g/kWh) ---
        low_pollution = carbon_intensity < 200
        
        records.append({
            'Datetime': dt,
            'Energy_kW': energy,
            'Temperature_C': temperature,
            'Humidity_pct': humidity,
            'PowerCut_Risk': round(power_cut_risk, 3),
            'ShortCircuit_Risk': round(short_circuit_risk, 3),
            'ExcessElectricity_Risk': round(excess_risk, 3),
            'CarbonCredit_USD': carbon_credit_usd,
            'LowPollution_Flag': low_pollution
        })
    
    df = pd.DataFrame(records)
    df.set_index('Datetime', inplace=True)
    return df

if __name__ == '__main__':
    df = generate_advanced_energy_data()
    df.to_csv('data/raw/advanced_energy_data.csv')
    print("Advanced dataset saved to data/raw/advanced_energy_data.csv")
    print(df.head())

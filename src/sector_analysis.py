"""
Sector-wise Energy Consumption and Carbon Credit Analysis
Generates hourly data for 6 sectors over 1 year, with seasonal peaks.
Calculates carbon credits in INR based on energy saved.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Sector definitions
SECTORS = {
    'Domestic': {
        'base_load': 1.5,       # kW
        'peak_hours': [(7,9), (18,20)],
        'peak_multiplier': 2.5,
        'off_peak_multiplier': 0.6,
        'seasonal_peak_months': [4,5,6,12],  # summer & winter
        'seasonal_factor': 1.3,
        'weekend_factor': 1.2,  # higher on weekends
        'carbon_intensity': 300,    # gCO2/kWh
        'baseline_waste': 0.20      # 20% waste without AI
    },
    'Food_Industry': {
        'base_load': 10.0,
        'peak_hours': [(8,12), (16,20)],
        'peak_multiplier': 2.2,
        'off_peak_multiplier': 0.7,
        'seasonal_peak_months': [9,10,11,12],  # festive season
        'seasonal_factor': 1.8,
        'weekend_factor': 1.4,  # busy on weekends
        'carbon_intensity': 350,
        'baseline_waste': 0.30
    },
    'Pharma_Industry': {
        'base_load': 8.0,
        'peak_hours': [(0,24)],  # continuous but slightly lower at night
        'peak_multiplier': 1.0,
        'off_peak_multiplier': 0.8,  # night reduction
        'seasonal_peak_months': [],  # no strong seasonality
        'seasonal_factor': 1.0,
        'weekend_factor': 0.9,  # less activity
        'carbon_intensity': 250,
        'baseline_waste': 0.10
    },
    'Car_Industry': {
        'base_load': 20.0,
        'peak_hours': [(6,22)],  # two shifts
        'peak_multiplier': 1.3,
        'off_peak_multiplier': 0.4,
        'seasonal_peak_months': [2,3,4,9,10],  # Feb-April (new models), Sept-Oct (festive demand)
        'seasonal_factor': 1.4,
        'weekend_factor': 1.1,  # slight increase
        'carbon_intensity': 400,
        'baseline_waste': 0.25
    },
    'Government': {
        'base_load': 5.0,
        'peak_hours': [(9,17)],
        'peak_multiplier': 1.5,
        'off_peak_multiplier': 0.3,
        'seasonal_peak_months': [1,2,3,11,12],  # budget & year-end
        'seasonal_factor': 1.2,
        'weekend_factor': 0.8,  # lower on weekends
        'carbon_intensity': 300,
        'baseline_waste': 0.15
    },
    'Private_Sector': {
        'base_load': 7.0,
        'peak_hours': [(9,20)],
        'peak_multiplier': 1.6,
        'off_peak_multiplier': 0.4,
        'seasonal_peak_months': [12,1,2,3],  # financial year end
        'seasonal_factor': 1.25,
        'weekend_factor': 1.0,  # normal
        'carbon_intensity': 320,
        'baseline_waste': 0.18
    }
}

def generate_sector_data(sector_name, start_date='2024-01-01', days=365, seed=42):
    """Generate hourly energy data for a single sector."""
    np.random.seed(seed + hash(sector_name) % 1000)
    date_range = pd.date_range(start=start_date, periods=days*24, freq='h')
    config = SECTORS[sector_name]
    
    energy = []
    for dt in date_range:
        hour = dt.hour
        month = dt.month
        
        # Base load
        base = config['base_load']
        
        # Hourly pattern
        factor = 1.0
        for (start, end) in config['peak_hours']:
            if start <= hour < end:
                factor = config['peak_multiplier']
                break
        else:
            # off-peak
            factor = config['off_peak_multiplier']
        
        # Seasonal factor
        seasonal = 1.0
        if month in config['seasonal_peak_months']:
            seasonal = config['seasonal_factor']
        
        # Weekend factor
        is_weekend = dt.weekday() >= 5  # 5=Sat, 6=Sun
        weekend_mult = config.get('weekend_factor', 1.0) if is_weekend else 1.0
        
        # Random noise
        noise = np.random.normal(0, 0.1 * base)
        
        consumption = base * factor * seasonal * weekend_mult + noise
        consumption = max(0.2, consumption)
        energy.append(round(consumption, 2))
    
    df = pd.DataFrame({'Datetime': date_range, 'Energy_kW': energy})
    df.set_index('Datetime', inplace=True)
    return df

def calculate_carbon_credits(actual_energy_series, baseline_waste_pct, carbon_intensity, price_per_ton_inr=2000):
    """
    actual_energy_series: hourly kWh (assuming kW is same as kWh per hour)
    baseline_waste_pct: e.g., 0.20 means without AI, energy would be 20% higher
    carbon_intensity: gCO2 per kWh
    returns: total carbon credits in INR
    """
    # Baseline energy = actual / (1 - waste_pct)  (because actual = baseline * (1 - waste))
    baseline_energy = actual_energy_series / (1 - baseline_waste_pct)
    energy_saved = baseline_energy - actual_energy_series
    co2_saved_kg = (energy_saved * carbon_intensity) / 1000
    co2_saved_tons = co2_saved_kg.sum() / 1000
    credits_inr = co2_saved_tons * price_per_ton_inr
    return credits_inr, co2_saved_tons

def plot_sector_comparison():
    """Generate comparison graphs for all sectors."""
    results = []
    daily_avg = {}
    
    figures_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures'))
    os.makedirs(figures_dir, exist_ok=True)
    
    for sector in SECTORS.keys():
        print(f"Generating data for {sector}...")
        df = generate_sector_data(sector, days=365)
        daily_avg[sector] = df.resample('D').mean()['Energy_kW']
        
        credits_inr, co2_tons = calculate_carbon_credits(
            df['Energy_kW'],
            SECTORS[sector]['baseline_waste'],
            SECTORS[sector]['carbon_intensity']
        )
        results.append({
            'Sector': sector.replace('_', ' '),
            'Avg Daily Energy (kWh)': df['Energy_kW'].mean(),
            'Total Annual Energy (MWh)': df['Energy_kW'].sum() / 1000,
            'Carbon Credits (INR)': credits_inr,
            'CO2 Saved (tons)': co2_tons
        })
    
    # DataFrame of results
    results_df = pd.DataFrame(results)
    print("\nSector-wise Carbon Credit Summary (1 year, AI forecasting)")
    print(results_df.to_string(index=False))
    
    # Plot 1: Bar chart of carbon credits
    plt.figure(figsize=(12,6))
    bars = plt.bar(results_df['Sector'], results_df['Carbon Credits (INR)'], 
                   color=['#2E86C1', '#28B463', '#E74C3C', '#F39C12', '#8E44AD', '#1ABC9C'])
    plt.title('Annual Carbon Credits Earned by Sector (₹)', fontsize=14)
    plt.ylabel('Indian Rupees (₹)')
    plt.xticks(rotation=45)
    for bar, val in zip(bars, results_df['Carbon Credits (INR)']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, 
                 f'INR {val:,.0f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'sector_carbon_credits.png'), dpi=150)
    plt.close()
    
    # Plot 2: Daily energy patterns for 2 weeks (to show differences)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for idx, sector in enumerate(SECTORS.keys()):
        df = generate_sector_data(sector, days=14)
        ax = axes[idx]
        ax.plot(df.index, df['Energy_kW'], linewidth=1)
        ax.set_title(sector.replace('_', ' '))
        ax.set_xlabel('Date')
        ax.set_ylabel('Power (kW)')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Two-Week Energy Consumption Patterns by Sector', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'sector_weekly_patterns.png'), dpi=150)
    plt.close()
    
    # Plot 3: Seasonal variation (monthly averages) - highlight festive peaks
    monthly_avg = {}
    for sector in SECTORS.keys():
        df = generate_sector_data(sector, days=365)
        df['month'] = df.index.month
        monthly = df.groupby('month')['Energy_kW'].mean()
        monthly_avg[sector] = monthly
    
    monthly_df = pd.DataFrame(monthly_avg)
    plt.figure(figsize=(14,6))
    for sector in monthly_df.columns:
        plt.plot(monthly_df.index, monthly_df[sector], marker='o', label=sector.replace('_', ' '), linewidth=2)
    plt.xlabel('Month')
    plt.ylabel('Average Daily Power (kW)')
    plt.title('Seasonal Variation in Energy Consumption by Sector')
    plt.xticks(range(1,13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Highlight festive months (Sept-Dec)
    plt.axvspan(9, 12, alpha=0.2, color='orange', label='Festive Season (Sep-Dec)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'sector_seasonal_trends.png'), dpi=150)
    plt.close()
    
    print("\nGraphs saved to outputs/figures/")
    return results_df

if __name__ == '__main__':
    results = plot_sector_comparison()

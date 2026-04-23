"""
Simplified Risk Trends Plot
Matches the style requested: X-axis months, Y-axis 0.0–0.15, three risk lines.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Load the advanced dataset
df = pd.read_csv('data/raw/advanced_energy_data.csv', index_col='Datetime', parse_dates=True)

# Filter for year 2024 (if data spans multiple years, adjust as needed)
df = df[df.index.year == 2024]

# Plot setup
fig, ax = plt.subplots(figsize=(14, 6))

# Plot the three risk factors
ax.plot(df.index, df['PowerCut_Risk'], label='PowerCut_Risk', linewidth=1.5, color='red')
ax.plot(df.index, df['ExcessElectricity_Risk'], label='ExcessElectricity_Risk', linewidth=1.5, color='blue')
ax.plot(df.index, df['ShortCircuit_Risk'], label='ShortCircuit_Risk', linewidth=1.5, color='orange')

# Formatting X-axis: show months (Jan, Feb, ...)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator())  # optional week ticks

# Set Y-axis limits to match the described range (0 to 0.15)
ax.set_ylim(0, 0.15)
ax.set_ylabel('Risk Probability')
ax.set_xlabel('Month of 2024')

# Title and legend
ax.set_title('Risk Factors Over Time (Simplified View)', fontsize=14)
ax.legend(loc='upper right')

# Grid for readability
ax.grid(True, linestyle='--', alpha=0.6)

# Rotate x-axis labels for better fit
plt.xticks(rotation=45)

# Tight layout and save
plt.tight_layout()
plt.savefig('outputs/figures/risk_trends_simplified.png', dpi=200)

print("Simplified risk trends plot saved to outputs/figures/risk_trends_simplified.png")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/raw/advanced_energy_data.csv', index_col='Datetime', parse_dates=True)

# Total carbon credits earned over the period
total_credits = df['CarbonCredit_USD'].sum()
print(f"Total carbon credits earned: ${total_credits:.2f}")

# Low pollution hours percentage
low_pollution_pct = df['LowPollution_Flag'].mean() * 100
print(f"Low pollution hours: {low_pollution_pct:.1f}%")

# Hourly average carbon credit
hourly_credit = df.groupby(df.index.hour)['CarbonCredit_USD'].mean()
plt.figure(figsize=(10,4))
hourly_credit.plot(kind='bar', color='green')
plt.title('Average Carbon Credit Earned by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('USD')
plt.savefig('outputs/figures/carbon_credit_by_hour.png')

# Risk trends
risk_cols = ['PowerCut_Risk', 'ShortCircuit_Risk', 'ExcessElectricity_Risk']
df[risk_cols].plot(figsize=(12,4), subplots=True, layout=(3,1), sharex=True)
plt.suptitle('Risk Factors Over Time')
plt.tight_layout()
plt.savefig('outputs/figures/risk_trends.png')

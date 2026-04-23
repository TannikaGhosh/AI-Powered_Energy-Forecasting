"""
Sector Comparison Dashboard - Carbon Credits & Energy Patterns
Run: python sector_dashboard.py
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.sector_analysis import generate_sector_data, SECTORS, calculate_carbon_credits

# Precompute for all sectors
sector_names = [s.replace('_', ' ') for s in SECTORS.keys()]
sector_keys = list(SECTORS.keys())

# Generate one year data for each sector (can be slow, do once and cache)
print("Generating sector data (this may take 10 seconds)...")
all_data = {}
carbon_results = {}
for key in sector_keys:
    df = generate_sector_data(key, days=365)
    all_data[key] = df
    credits, co2 = calculate_carbon_credits(df['Energy_kW'], 
                                            SECTORS[key]['baseline_waste'],
                                            SECTORS[key]['carbon_intensity'])
    carbon_results[key] = {'credits': credits, 'co2_tons': co2}

app = dash.Dash(__name__)
app.title = "Sector-wise Carbon Credit Analyzer"

app.layout = html.Div([
    html.H1("🏭 Energy Sector Analysis: Carbon Credits in ₹", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Select Sector:"),
        dcc.Dropdown(id='sector-dropdown', options=[{'label': s, 'value': k} for s,k in zip(sector_names, sector_keys)],
                     value='Domestic', clearable=False, style={'width': '50%', 'margin': 'auto'}),
    ], style={'textAlign': 'center', 'margin': 20}),
    
    html.Div([
        dcc.Graph(id='carbon-gauge', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='weekly-pattern', style={'width': '48%', 'display': 'inline-block'}),
    ]),
    
    html.Div([
        dcc.Graph(id='seasonal-trend', style={'width': '100%'}),
    ]),
    
    html.Div(id='credit-text', style={'textAlign': 'center', 'marginTop': 20, 'fontSize': 18})
])

@app.callback(
    [Output('carbon-gauge', 'figure'),
     Output('weekly-pattern', 'figure'),
     Output('seasonal-trend', 'figure'),
     Output('credit-text', 'children')],
    Input('sector-dropdown', 'value')
)
def update_sector(sector_key):
    df = all_data[sector_key]
    credits = carbon_results[sector_key]['credits']
    co2 = carbon_results[sector_key]['co2_tons']
    config = SECTORS[sector_key]
    
    # Gauge: Carbon Credits
    gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = credits,
        title = {'text': f"Annual Carbon Credits (₹)"},
        delta = {'reference': 50000, 'increasing': {'color': "green"}},
        gauge = {'axis': {'range': [0, max(credits*1.2, 100000)]},
                 'bar': {'color': "darkgreen"},
                 'steps': [
                     {'range': [0, credits*0.5], 'color': "lightgray"},
                     {'range': [credits*0.5, credits], 'color': "lightgreen"}],
                 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': credits}}
    ))
    gauge.update_layout(height=400)
    
    # Weekly pattern (last 14 days)
    last_14 = df.iloc[-336:]  # 14*24
    weekly = go.Figure()
    weekly.add_trace(go.Scatter(x=last_14.index, y=last_14['Energy_kW'],
                                mode='lines', fill='tozeroy', name='Hourly Power'))
    weekly.update_layout(title=f'Last 14 Days Energy Pattern - {sector_key.replace("_", " ")}',
                         xaxis_title='Date', yaxis_title='Power (kW)')
    
    # Seasonal trend (monthly averages)
    df_month = df.copy()
    df_month['Month'] = df_month.index.month
    monthly_avg = df_month.groupby('Month')['Energy_kW'].mean()
    seasonal = go.Figure()
    seasonal.add_trace(go.Scatter(x=monthly_avg.index, y=monthly_avg.values, mode='lines+markers',
                                  line=dict(width=3), marker=dict(size=8)))
    seasonal.update_layout(title=f'Seasonal Variation - {sector_key.replace("_", " ")}',
                           xaxis=dict(tickmode='array', tickvals=list(range(1,13)), ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
                           yaxis_title='Average Power (kW)')
    # Highlight festive months (Sep-Dec)
    seasonal.add_vrect(x0=9, x1=12, fillcolor="orange", opacity=0.2, layer="below", line_width=0)
    seasonal.add_annotation(x=10.5, y=max(monthly_avg)*0.9, text="Festive Season", showarrow=False, font=dict(size=12))
    
    # Text
    text = html.Div([
        html.H4(f"Sector: {sector_key.replace('_', ' ')}"),
        html.P(f"📉 Annual CO₂ Saved: {co2:.2f} tons"),
        html.P(f"💰 Carbon Credits Earned: ₹{credits:,.2f}"),
        html.P(f"⚡ Baseline Waste Assumption: {config['baseline_waste']*100}% without AI forecasting"),
        html.P(f"🌍 Carbon Intensity: {config['carbon_intensity']} gCO₂/kWh")
    ])
    
    return gauge, weekly, seasonal, text

if __name__ == '__main__':
    # Use port 8051 for the dashboard and match the requested local link
    app.run(debug=True, port=8051)

"""
3D Interactive Dashboard for Energy Forecasting
Displays:
- 2D scatter: Energy vs Temperature 
- Heatmap: Hour vs Day of Week vs Energy
- 2D bar: Carbon credits by hour
- Risk summary & time filter
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load your advanced dataset (or generate on the fly)
try:
    df = pd.read_csv('data/raw/advanced_energy_data.csv', index_col='Datetime', parse_dates=True)
except FileNotFoundError:
    # Fallback: generate minimal dataset if not present
    print("Generating fallback dataset for dashboard...")
    from src.advanced_data_generation import generate_advanced_energy_data
    df = generate_advanced_energy_data(days=30)
    df.to_csv('data/raw/advanced_energy_data.csv')

# Feature engineering for dashboard
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_name'] = df.index.day_name()
df['date'] = df.index.date

# Create a risk composite score for coloring
df['composite_risk'] = (df['PowerCut_Risk'] + df['ShortCircuit_Risk'] + df['ExcessElectricity_Risk']) / 3

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("⚡ Energy Forecasting Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-range',
            start_date=df.index.min().date(),
            end_date=df.index.max().date(),
            display_format='YYYY-MM-DD'
        ),
    ], style={'margin': 20}),
    
    html.Div([
        dcc.Graph(id='scatter-2d', style={'height': '500px'}),
        dcc.Graph(id='heatmap-2d', style={'height': '500px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    
    html.Div([
        dcc.Graph(id='bar-2d', style={'height': '500px', 'width': '49%', 'display': 'inline-block'}),
        html.Div(id='risk-summary', style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    ]),
    
    html.Div([
        html.H3("Live Risk Indicators (Latest Hour)"),
        html.Div(id='live-risk-gauge')
    ], style={'margin': 20}),
])

# Callback for 2D scatter plot
@app.callback(
    Output('scatter-2d', 'figure'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_scatter(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = df[(df.index >= start_date) & (df.index <= end_date)]
    fig = px.scatter(
        filtered, x='Temperature_C', y='Energy_kW',
        color='composite_risk', size='Humidity_pct',
        hover_data={ 'PowerCut_Risk': ':.2f', 'CarbonCredit_USD': ':.3f' },
        title='Energy vs Temperature (color = risk, size = humidity)',
        labels={'Energy_kW': 'Energy (kW)', 'Temperature_C': 'Temp (°C)', 'Humidity_pct': 'Humidity (%)'},
        color_continuous_scale='RdYlGn_r'
    )
    return fig

# Callback for Heatmap (energy over hour and day of week)
@app.callback(
    Output('heatmap-2d', 'figure'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_surface(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = df[(df.index >= start_date) & (df.index <= end_date)]
    # Group by hour and day name
    hourly_day = filtered.groupby(['hour', 'day_name'])['Energy_kW'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig = px.bar(
        hourly_day,
        x='hour',
        y='Energy_kW',
        color='day_name',
        barmode='group',
        title='Average Energy by Hour and Day (Bar Graph)',
        labels={'hour': 'Hour of Day', 'Energy_kW': 'Avg Energy (kW)', 'day_name': 'Day'},
        category_orders={'day_name': day_order},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(height=500)
    return fig

# Callback for 2D bar chart (carbon credits by hour)
@app.callback(
    Output('bar-2d', 'figure'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_bar(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = df[(df.index >= start_date) & (df.index <= end_date)]
    hourly_credit = filtered.groupby('hour')['CarbonCredit_USD'].mean().reset_index()
    
    fig = px.bar(
        hourly_credit, x='hour', y='CarbonCredit_USD',
        color='CarbonCredit_USD', title='Average Carbon Credit by Hour',
        labels={'hour': 'Hour', 'CarbonCredit_USD': 'USD'},
        color_continuous_scale='Greens'
    )
    # Improve layout
    # Removed invalid 3D scene update for 2D plot
    return fig

# Callback for risk summary text
@app.callback(
    Output('risk-summary', 'children'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def risk_summary(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = df[(df.index >= start_date) & (df.index <= end_date)]
    avg_power_cut = filtered['PowerCut_Risk'].mean()
    avg_short = filtered['ShortCircuit_Risk'].mean()
    avg_excess = filtered['ExcessElectricity_Risk'].mean()
    total_credits = filtered['CarbonCredit_USD'].sum()
    low_pollution_pct = filtered['LowPollution_Flag'].mean() * 100
    
    return html.Div([
        html.H3("📊 Summary Metrics"),
        html.P(f"Average Power Cut Risk: {avg_power_cut:.2f}"),
        html.P(f"Average Short Circuit Risk: {avg_short:.2f}"),
        html.P(f"Average Excess Electricity Risk: {avg_excess:.2f}"),
        html.P(f"Total Carbon Credits Earned: ${total_credits:.2f}"),
        html.P(f"Low Pollution Hours: {low_pollution_pct:.1f}%"),
    ])

# Callback for live risk gauge (latest hour)
@app.callback(
    Output('live-risk-gauge', 'children'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def live_risk(_, __):
    latest = df.iloc[-1]
    return html.Div([
        dcc.Graph(
            figure=go.Figure(go.Indicator(
                mode = "gauge+number",
                value = latest['PowerCut_Risk'] * 100,
                title = {'text': "Power Cut Risk (%)"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkred"}}
            )),
            style={'display': 'inline-block', 'width': '30%'}
        ),
        dcc.Graph(
            figure=go.Figure(go.Indicator(
                mode = "gauge+number",
                value = latest['ShortCircuit_Risk'] * 100,
                title = {'text': "Short Circuit Risk (%)"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "orange"}}
            )),
            style={'display': 'inline-block', 'width': '30%'}
        ),
        dcc.Graph(
            figure=go.Figure(go.Indicator(
                mode = "gauge+number",
                value = latest['ExcessElectricity_Risk'] * 100,
                title = {'text': "Excess Electricity Risk (%)"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "blue"}}
            )),
            style={'display': 'inline-block', 'width': '30%'}
        ),
        html.P(f"Latest hour: {latest.name} | Energy: {latest['Energy_kW']} kW | Carbon Credit: ${latest['CarbonCredit_USD']}")
    ])

if __name__ == '__main__':
    app.run(debug=True, port=8050)

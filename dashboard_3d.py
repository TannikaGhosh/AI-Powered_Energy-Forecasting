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
import joblib
from datetime import timedelta
from src.sector_analysis import generate_sector_data, SECTORS, calculate_carbon_credits

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

# Load appliance data
try:
    df_app = pd.read_csv('data/raw/appliance_data.csv', index_col='Datetime', parse_dates=True)
except FileNotFoundError:
    print("Appliance data not found.")

# Load model
model = joblib.load('models/nn_energy_model.save')
scaler = joblib.load('models/scaler.save')
SEQ_LEN = 24
CARBON_CREDIT_PRICE_INR = 2000

PRIORITY_ORDER = ['Robot_W', 'TV_W', 'Lights_W', 'Fridge_W', 'WaterHeater_W', 'HVAC_W']
APPLIANCE_NAMES = {
    'Robot_W': 'Cleaning Robot', 'TV_W': 'Smart TV', 'Lights_W': 'Lights',
    'Fridge_W': 'Refrigerator', 'WaterHeater_W': 'Water Heater', 'HVAC_W': 'HVAC'
}

# Sector data
sector_names = [s.replace('_', ' ') for s in SECTORS.keys()]
sector_keys = list(SECTORS.keys())
all_data = {}
carbon_results = {}
for key in sector_keys:
    df_sector = generate_sector_data(key, days=365)  # Use full year for better seasonal variation
    all_data[key] = df_sector
    credits, co2 = calculate_carbon_credits(df_sector['Energy_kW'], 
                                            SECTORS[key]['baseline_waste'],
                                            SECTORS[key]['carbon_intensity'])
    carbon_results[key] = {'credits': credits, 'co2_tons': co2}

# --- ML Prediction implementation ---
def predict_next_24h(last_24h_actual):
    # MLP expects a flat array of shape (1, 24)
    last_24h_actual = np.array(last_24h_actual).flatten()
    scaled = scaler.transform(last_24h_actual.reshape(-1, 1)).flatten()
    predictions = []
    current_seq = scaled[-SEQ_LEN:].copy()
    for _ in range(24):
        X_input = current_seq.reshape(1, SEQ_LEN) # Reshape for Scikit-Learn 2D
        pred_scaled = model.predict(X_input)[0]
        predictions.append(pred_scaled)
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = pred_scaled
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def get_shedding_plan(current_loads, predicted_load, threshold):
    """Return list of appliances to shed and a message."""
    priority = PRIORITY_ORDER
    if predicted_load <= threshold:
        return [], f"✅ Predicted load ({predicted_load:.0f}W) within threshold. No shedding."
    
    excess = predicted_load - threshold
    to_shed = []
    for app in priority:
        if current_loads.get(app, 0) > 0:
            to_shed.append(app)
            excess -= current_loads[app]
            if excess <= 0:
                break
    if not to_shed:
        return [], f"⚠️ Predicted load ({predicted_load:.0f}W) exceeds threshold, but no appliances to shed."
    return to_shed, f"🔴 Shedding: {', '.join([APPLIANCE_NAMES[a] for a in to_shed])} to reduce by {predicted_load - threshold:.0f}W."

# --- Layouts ---
home_layout = html.Div([
    html.H3("Neural Network Forecast for Next 24 Hours"),
    dcc.Graph(id='forecast-graph'),
    html.Hr(),
    html.H3("Load Shedding Controller"),
    html.Div([
        html.Label("Power Threshold (Watts):"),
        dcc.Slider(id='threshold-slider', min=100, max=7000, step=100, value=4500, updatemode='drag',
                   marks={100:'0.1kW', 500:'0.5kW', 1000:'1kW', 2000:'2kW', 3000:'3kW', 4000:'4kW', 5000:'5kW', 6000:'6kW', 7000:'7kW'}),
        html.Button('Refresh Forecast & Plan', id='refresh-btn', n_clicks=0,
                    style={'margin': '10px', 'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px', 'border': 'none', 'borderRadius': '5px'}),
        html.Button('🔴 Simulate Peak Load (>2000W)', id='peak-btn', n_clicks=0,
                    style={'margin': '10px', 'backgroundColor': '#f44336', 'color': 'white', 'padding': '10px', 'border': 'none', 'borderRadius': '5px'}),
    ]),
    html.Div(id='shedding-output', style={'marginTop': '20px', 'padding': '15px', 'borderRadius': '5px',
                                          'backgroundColor': '#f9f9f9', 'fontSize': '18px'}),
    html.Div(id='appliance-status', style={'marginTop': '20px'})
])

sector_layout = html.Div([
    html.H2("🏭 Sector Carbon Credits Analysis", style={'textAlign': 'center'}),
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

current_3d_layout = html.Div([
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

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("⚡ Energy Forecasting Dashboard", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        dcc.Tab(label='Home Energy Management', children=[home_layout]),
        dcc.Tab(label='Sector Carbon Credit Analysis', children=[sector_layout]),
        dcc.Tab(label='Energy Analytics & 3D Risk Dashboard', children=[current_3d_layout]),
    ])
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

# --- Home Energy Management Callbacks ---
@app.callback(
    [Output('forecast-graph', 'figure'),
     Output('shedding-output', 'children'),
     Output('appliance-status', 'children')],
    [Input('refresh-btn', 'n_clicks'),
     Input('threshold-slider', 'value'),
     Input('peak-btn', 'n_clicks')]
)
def update_forecast_and_shedding(n_clicks, threshold, peak_clicks):
    # Get actual data
    last_24h_actual = df_app['Total_Power_W'].iloc[-SEQ_LEN:].values
    preds = predict_next_24h(last_24h_actual)
    
    last_timestamp = df_app.index[-1]
    forecast_index = [last_timestamp + timedelta(hours=i+1) for i in range(24)]
    actual_48h = df_app['Total_Power_W'].iloc[-48:]
    
    # Determine predicted load for next hour
    normal_predicted = preds[0]
    # Peak mode toggle: if peak_clicks is odd, use high load
    if (peak_clicks or 0) % 2 == 1:
        predicted_load = 3500  # force >2000W
        peak_mode_note = "⚠️ PEAK MODE ACTIVE (simulated high demand) ⚠️"
    else:
        predicted_load = normal_predicted
        peak_mode_note = ""
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_48h.index, y=actual_48h.values,
                             mode='lines', name='Actual (last 48h)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_index, y=preds,
                             mode='lines+markers', name='Neural Net Forecast (next 24h)',
                             line=dict(color='red', dash='dash')))
    fig.add_hline(y=threshold, line_dash="dot", line_color="green",
                  annotation_text=f"Threshold = {threshold}W")
    fig.update_layout(title=f'Energy Forecast vs Actual {peak_mode_note}',
                      xaxis_title='Time', yaxis_title='Power (Watts)')
    
    # Current appliance loads
    latest = df_app.iloc[-1]
    current_loads = {col: latest[col] for col in PRIORITY_ORDER}
    
    # Shedding plan
    plan, msg = get_shedding_plan(current_loads, predicted_load, threshold)
    
    if plan:
        plan_html = html.Div([
            html.H4("Recommended Actions:", style={'color': '#c62828'}),
            html.Ul([html.Li(f"Turn OFF {APPLIANCE_NAMES[app]}") for app in plan])
        ], style={'padding': '10px', 'backgroundColor': '#ffebee'})
    else:
        plan_html = html.Div(msg, style={'color': 'green' if 'within' in msg else 'orange', 'fontWeight': 'bold'})
    
    shedding_display = html.Div([
        html.H4(f"Predicted next hour load: {predicted_load:.0f} W"),
        plan_html
    ])
    
    # Appliance status table with ON/OFF comment and Red Selection formatting
    status_rows = []
    for app in PRIORITY_ORDER:
        power = current_loads[app]
        status = "🔴 ON" if power > 0 else "⚪ OFF"
        
        row_style = {'border': '1px solid gray'}
        if app in plan:
            row_style.update({'backgroundColor': '#ffebee', 'color': '#c62828', 'fontWeight': 'bold'})
            
        status_rows.append(html.Tr([
            html.Td(APPLIANCE_NAMES[app], style={'padding': '8px'}),
            html.Td(f"{power:.0f}", style={'padding': '8px'}),
            html.Td(status, style={'padding': '8px'})
        ], style=row_style))
        
    status_table = html.Table(
        [html.Tr([html.Th("Appliance", style={'padding':'8px'}), html.Th("Power (W)", style={'padding':'8px'}), html.Th("Status", style={'padding':'8px'})], style={'backgroundColor': '#f5f5f5'})] + status_rows,
        style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid gray', 'textAlign': 'left'}
    )
    
    return fig, shedding_display, status_table

# --- Sector Analysis Callbacks ---
@app.callback(
    [Output('carbon-gauge', 'figure'),
     Output('weekly-pattern', 'figure'),
     Output('seasonal-trend', 'figure'),
     Output('credit-text', 'children')],
    [Input('sector-dropdown', 'value')]
)
def update_sector_analysis(sector_key):
    df_sector = all_data[sector_key]
    credits = carbon_results[sector_key]['credits']
    co2 = carbon_results[sector_key]['co2_tons']
    config = SECTORS[sector_key]

    gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = credits,
        title = {'text': f"Carbon Credits Earned (₹{CARBON_CREDIT_PRICE_INR}/ton)"},
        delta = {'reference': 0},
        gauge = {'axis': {'range': [0, max(credits*1.2, 100)]}, 'bar': {'color': "green"}}
    ))
    
    # Weekly pattern
    weekly = df_sector.groupby(df_sector.index.dayofweek)['Energy_kW'].mean()
    weekly_fig = px.bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], y=weekly.values,
                        title=f'Average Energy by Day of Week - {sector_key.replace("_", " ")}', labels={'x': 'Day', 'y': 'Energy (kW)'})
    
    # Seasonal trend
    seasonal = df_sector.groupby(df_sector.index.month)['Energy_kW'].mean()
    seasonal = seasonal.reindex(range(1,13), fill_value=float('nan'))  # Ensure all months are present
    seasonal_fig = px.line(x=range(1,13), y=seasonal.values,
                           title=f'Average Energy by Month - {sector_key.replace("_", " ")}', labels={'x': 'Month', 'y': 'Energy (kW)'})
    
    credit_text = f"🏆 {sector_key.replace('_', ' ')} Sector: Earned ₹{credits:.0f} in carbon credits by saving {co2:.1f} tons of CO₂ through AI forecasting."
    
    return gauge, weekly_fig, seasonal_fig, credit_text

if __name__ == '__main__':
    app.run(debug=True, port=8050)

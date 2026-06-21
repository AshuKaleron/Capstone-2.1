# =============================================================================
# 🌡️ Melbourne Temperature Forecasting — Streamlit Dashboard
# =============================================================================
# Production-ready Streamlit application for next-day minimum temperature
# prediction using a serialized deep-learning champion model.
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

# ── Page Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Melbourne Temperature Forecast",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Styling ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 40%, #415a77 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    font-size: 1rem;
    opacity: 0.85;
    margin: 0;
    font-weight: 300;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    border: 1px solid rgba(255,255,255,0.08);
    transition: transform 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
}
.metric-card .metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    opacity: 0.7;
    margin-bottom: 0.3rem;
}
.metric-card .metric-value {
    font-size: 1.9rem;
    font-weight: 700;
}
.metric-card .metric-unit {
    font-size: 0.75rem;
    opacity: 0.6;
}

.prediction-card {
    background: linear-gradient(135deg, #00b4d8, #0077b6);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(0,119,182,0.35);
    margin: 1rem 0;
}
.prediction-card .pred-label {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    opacity: 0.85;
}
.prediction-card .pred-value {
    font-size: 3.5rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.prediction-card .pred-unit {
    font-size: 1rem;
    opacity: 0.75;
}

.section-divider {
    height: 3px;
    background: linear-gradient(90deg, transparent, #415a77, transparent);
    border: none;
    margin: 2rem 0;
    border-radius: 2px;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(100,100,120,0.15);
    border-radius: 12px;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loading ──────────────────────────────────────────────
@st.cache_data
def load_dataset():
    """Load and clean the raw temperature dataset."""
    df = pd.read_csv('dataset.csv', skipfooter=2, engine='python')
    df.columns = ['Date', 'Temperature']
    df['Temperature'] = df['Temperature'].astype(str).str.replace('?', '', regex=False)
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce').astype('float64')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.asfreq('D')
    df['Temperature'] = df['Temperature'].interpolate(method='linear')
    return df


@st.cache_resource
def load_model():
    """Load the serialized Keras champion model."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('champion_model.keras')
        return model
    except Exception as e:
        st.error(f"❌ Failed to load champion model: {e}")
        return None


@st.cache_resource
def load_scaler():
    """Load the serialized MinMaxScaler."""
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load scaler: {e}")
        return None


@st.cache_data
def load_champion_info():
    """Load champion metadata and metrics."""
    try:
        with open('champion_info.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None


# ── Load Everything ────────────────────────────────────────────────────
df = load_dataset()
model = load_model()
scaler = load_scaler()
champion_info = load_champion_info()

WINDOW_SIZE = 7
MELB_LAT = -37.8136
MELB_LON = 144.9631

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌡️ Melbourne Minimum Temperature Forecast</h1>
    <p>AI-powered next-day temperature prediction • Daily minimums 1981–1990 • Melbourne, Australia</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ Station Location")
    map_df = pd.DataFrame({'lat': [MELB_LAT], 'lon': [MELB_LON]})
    st.map(map_df, zoom=10, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Dataset Summary")
    st.markdown(f"""
    | Property | Value |
    |:---|:---|
    | **Records** | {len(df):,} |
    | **Start** | {df.index.min().strftime('%Y-%m-%d')} |
    | **End** | {df.index.max().strftime('%Y-%m-%d')} |
    | **Mean** | {df['Temperature'].mean():.1f} °C |
    | **Min** | {df['Temperature'].min():.1f} °C |
    | **Max** | {df['Temperature'].max():.1f} °C |
    """)

    if champion_info:
        st.markdown("---")
        st.markdown("### 🏆 Champion Model")
        st.markdown(f"**{champion_info.get('deploy_model_name', 'N/A')}**")
        st.markdown(f"Window Size: **{champion_info.get('window_size', 7)} days**")

# ── Model Confidence KPIs ──────────────────────────────────────────────
if champion_info:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Champion Model</div>
            <div class="metric-value" style="font-size:1.4rem;">{champion_info.get('deploy_model_name', 'N/A')}</div>
            <div class="metric-unit">Deep Learning</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">{champion_info.get('deploy_rmse', 0):.3f}</div>
            <div class="metric-unit">Root Mean Sq Error</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">{champion_info.get('deploy_mae', 0):.3f}</div>
            <div class="metric-unit">Mean Abs Error</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{champion_info.get('deploy_r2', 0):.3f}</div>
            <div class="metric-unit">Coefficient of Determination</div>
        </div>
        """, unsafe_allow_html=True)

    # Full metrics table
    if champion_info.get('metrics'):
        with st.expander("📋 Full Model Comparison Table"):
            full_metrics = pd.DataFrame(champion_info['metrics'])
            st.dataframe(full_metrics, use_container_width=True, hide_index=True)


# ── Next-Day Forecasting Interface ────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🔮 Next-Day Temperature Prediction")
st.markdown("Enter the minimum temperatures for the **past 7 days** to forecast tomorrow's minimum temperature.")

if model is not None and scaler is not None:
    # --- Input sliders for 7 days ---
    cols = st.columns(7)
    input_temps = []
    for i, col in enumerate(cols):
        with col:
            val = st.number_input(
                f"Day t-{7-i}",
                min_value=-5.0,
                max_value=35.0,
                value=float(round(df['Temperature'].iloc[-(7-i)], 1)),
                step=0.1,
                key=f"temp_day_{i}",
                help=f"Temperature {7-i} days ago"
            )
            input_temps.append(val)

    # --- Predict button ---
    if st.button("🚀 Predict Next-Day Temperature", type="primary", use_container_width=True):
        input_array = np.array(input_temps).reshape(-1, 1)

        # Apply log transform if the training pipeline used it
        if champion_info and champion_info.get('use_log_transform', False):
            log_offset = champion_info.get('log_offset', 1.0)
            input_array = np.log(input_array + log_offset)

        input_scaled = scaler.transform(input_array).flatten()
        input_reshaped = input_scaled.reshape(1, WINDOW_SIZE, 1)

        pred_scaled = model.predict(input_reshaped, verbose=0)
        pred_temp = scaler.inverse_transform(pred_scaled)[0][0]

        # Inverse log-transform if the training pipeline used it
        if champion_info and champion_info.get('use_log_transform', False):
            log_offset = champion_info.get('log_offset', 1.0)
            pred_temp = float(np.exp(pred_temp) - log_offset)

        st.markdown(f"""
        <div class="prediction-card">
            <div class="pred-label">Predicted Minimum Temperature</div>
            <div class="pred-value">{pred_temp:.1f}°C</div>
            <div class="pred-unit">for the next day (t+1)</div>
        </div>
        """, unsafe_allow_html=True)

        # --- Context: show input trend ---
        fig_input = go.Figure()
        days = [f"Day t-{7-i}" for i in range(7)] + ["Day t+1"]
        temps_with_pred = input_temps + [pred_temp]
        colors = ['#415a77'] * 7 + ['#00b4d8']

        fig_input.add_trace(go.Bar(
            x=days, y=temps_with_pred,
            marker_color=colors,
            text=[f"{t:.1f}°C" for t in temps_with_pred],
            textposition='outside',
        ))
        fig_input.update_layout(
            title="Input Trend → Prediction",
            yaxis_title="Temperature (°C)",
            template='plotly_white',
            height=350,
            showlegend=False,
            margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_input, use_container_width=True)
else:
    st.warning("⚠️ Model or scaler not loaded. Please run the Colab pipeline first to generate `champion_model.keras` and `scaler.pkl`.")


# ── EDA & Visualization Section ───────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 📊 Exploratory Data Analysis")

# --- 1. Historical Timeline with Moving Averages ---
with st.expander("📈 Historical Temperature Timeline", expanded=True):
    ma7 = df['Temperature'].rolling(7, center=True).mean()
    ma30 = df['Temperature'].rolling(30, center=True).mean()

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=df.index, y=df['Temperature'],
        name='Daily Min Temp', mode='lines',
        line=dict(color='rgba(144,202,249,0.5)', width=0.8),
    ))
    fig_timeline.add_trace(go.Scatter(
        x=df.index, y=ma7,
        name='7-Day MA', mode='lines',
        line=dict(color='#ff7043', width=1.8),
    ))
    fig_timeline.add_trace(go.Scatter(
        x=df.index, y=ma30,
        name='30-Day MA', mode='lines',
        line=dict(color='#1a237e', width=2.5),
    ))
    fig_timeline.update_layout(
        title='Melbourne Daily Minimum Temperature (1981–1990)',
        xaxis_title='Date', yaxis_title='Temperature (°C)',
        template='plotly_white', height=450,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# --- 2. Monthly Aggregated Bar Chart ---
with st.expander("📊 Monthly Average Temperature"):
    df_month = df.copy()
    df_month['Month'] = df_month.index.month
    df_month['Year'] = df_month.index.year
    monthly_agg = df_month.groupby(['Year', 'Month'])['Temperature'].mean().reset_index()
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly_agg['Month_Name'] = monthly_agg['Month'].map(month_names)

    fig_monthly = px.bar(
        monthly_agg, x='Month_Name', y='Temperature', color='Year',
        barmode='group',
        title='Average Monthly Minimum Temperature by Year',
        labels={'Temperature': 'Avg Min Temp (°C)', 'Month_Name': 'Month'},
        color_continuous_scale='Viridis',
        category_orders={'Month_Name': list(month_names.values())},
    )
    fig_monthly.update_layout(template='plotly_white', height=450)
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Overall monthly means
    overall_monthly = df_month.groupby('Month')['Temperature'].mean()
    fig_overall = go.Figure(go.Bar(
        x=[month_names[m] for m in overall_monthly.index],
        y=overall_monthly.values,
        marker_color=px.colors.sequential.RdBu_r,
        text=[f"{v:.1f}°C" for v in overall_monthly.values],
        textposition='outside',
    ))
    fig_overall.update_layout(
        title='Overall Monthly Average (1981–1990)',
        yaxis_title='Temperature (°C)',
        template='plotly_white', height=350,
        showlegend=False,
    )
    st.plotly_chart(fig_overall, use_container_width=True)

# --- 3. Lag Scatter Plots ---
with st.expander("🔗 Autoregressive Lag Scatter Plots"):
    col_lag1, col_lag7 = st.columns(2)

    with col_lag1:
        fig_lag1 = go.Figure(go.Scatter(
            x=df['Temperature'].iloc[:-1].values,
            y=df['Temperature'].iloc[1:].values,
            mode='markers',
            marker=dict(color='#1565c0', size=3, opacity=0.3),
            name='Lag-1',
        ))
        rng = [df['Temperature'].min() - 1, df['Temperature'].max() + 1]
        fig_lag1.add_trace(go.Scatter(
            x=rng, y=rng, mode='lines',
            line=dict(color='#e53935', dash='dash', width=1.5),
            name='y = x',
        ))
        fig_lag1.update_layout(
            title='Lag-1: Y(t) vs Y(t-1)',
            xaxis_title='Y(t-1) °C', yaxis_title='Y(t) °C',
            template='plotly_white', height=400,
        )
        st.plotly_chart(fig_lag1, use_container_width=True)

    with col_lag7:
        fig_lag7 = go.Figure(go.Scatter(
            x=df['Temperature'].iloc[:-7].values,
            y=df['Temperature'].iloc[7:].values,
            mode='markers',
            marker=dict(color='#6a1b9a', size=3, opacity=0.3),
            name='Lag-7',
        ))
        fig_lag7.add_trace(go.Scatter(
            x=rng, y=rng, mode='lines',
            line=dict(color='#e53935', dash='dash', width=1.5),
            name='y = x',
        ))
        fig_lag7.update_layout(
            title='Lag-7: Y(t) vs Y(t-7)',
            xaxis_title='Y(t-7) °C', yaxis_title='Y(t) °C',
            template='plotly_white', height=400,
        )
        st.plotly_chart(fig_lag7, use_container_width=True)

# --- 4. Temperature Distribution ---
with st.expander("📉 Temperature Distribution & Yearly Box Plots"):
    col_hist, col_box = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            df, x='Temperature', nbins=50, marginal='box',
            title='Temperature Distribution with KDE',
            labels={'Temperature': 'Temperature (°C)'},
            color_discrete_sequence=['#42a5f5'],
        )
        fig_hist.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        df_box = df.copy()
        df_box['Year'] = df_box.index.year.astype(str)
        fig_box = px.box(
            df_box, x='Year', y='Temperature',
            title='Temperature Distribution by Year',
            labels={'Temperature': 'Temperature (°C)'},
            color='Year',
            color_discrete_sequence=px.colors.sequential.RdBu_r,
        )
        fig_box.update_layout(template='plotly_white', height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; opacity:0.5; padding: 1rem 0;">
    <p style="font-size:0.8rem;">Melbourne Daily Minimum Temperature Forecasting Dashboard<br>
    Built with Streamlit • TensorFlow • Plotly</p>
</div>
""", unsafe_allow_html=True)

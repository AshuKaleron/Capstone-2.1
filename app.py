# =============================================================================
# 🌡️ Melbourne Temperature Forecasting — Streamlit Dashboard (Premium Edition)
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
    page_title="Melbourne Temp Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Dark Theme & Glassmorphism ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Main font setup */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #e2e8f0;
}

/* Page Background */
.stApp {
    background-color: #090d16;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Premium Header Banner */
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 60%, #020617 100%);
    padding: 2.5rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.main-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(14, 165, 233, 0.15) 0%, transparent 70%);
    z-index: 1;
}

.main-header h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-header p {
    font-size: 1.1rem;
    opacity: 0.8;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.2px;
}

/* Dashboard Cards (Glassmorphism) */
.glass-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 1.8rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    margin-bottom: 1.5rem;
}

/* Prediction Output Card (Glowing Cyan) */
.prediction-card {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(3, 105, 161, 0.2) 100%);
    border: 1.5px solid rgba(14, 165, 233, 0.4);
    border-radius: 24px;
    padding: 2.2rem;
    text-align: center;
    color: white;
    box-shadow: 0 15px 35px rgba(14, 165, 233, 0.15);
    margin: 1.5rem 0;
    position: relative;
}

.prediction-card .pred-label {
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #38bdf8;
    font-weight: 600;
}

.prediction-card .pred-value {
    font-family: 'Outfit', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    margin: 0.5rem 0;
    color: #ffffff;
    text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
}

.prediction-card .pred-unit {
    font-size: 0.95rem;
    opacity: 0.8;
    font-weight: 300;
}

/* Custom KPI Metrics */
.kpi-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.kpi-card .kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94a3b8;
    margin-bottom: 0.3rem;
}

.kpi-card .kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
}

.kpi-card .kpi-sub {
    font-size: 0.7rem;
    color: #64748b;
}

/* Interactive Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px 12px 0px 0px;
    color: #94a3b8;
    padding: 10px 20px;
    font-weight: 500;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: rgba(14, 165, 233, 0.15) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(14, 165, 233, 0.3) !important;
}

/* Sidebar preset pills */
.preset-button {
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.preset-button:hover {
    background-color: rgba(14, 165, 233, 0.1);
    border-color: rgba(14, 165, 233, 0.3);
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
    """Load the Keras model using tf_keras for robust backward compatibility."""
    try:
        # Crucial: Load using tf_keras to bypass Keras 3 schema translation error
        import tf_keras as keras
        model = keras.models.load_model('champion_model.keras')
        return model
    except Exception as e:
        # Fallback to standard tf.keras if tf_keras package is missing
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model('champion_model.keras')
            return model
        except Exception as e_inner:
            st.error(f"❌ Failed to load champion model: {e_inner}")
            return None

@st.cache_resource
def load_scaler():
    """Load the MinMaxScaler."""
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load scaler: {e}")
        return None

@st.cache_data
def load_champion_info():
    """Load champion metadata."""
    try:
        with open('champion_info.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

# ── Global Setup ──────────────────────────────────────────────────────
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
    <h1>🌡️ Melbourne Climate Analytics & AI Forecaster</h1>
    <p>Predictive dashboard powered by Deep Learning (LSTM/GRU) and Classical Time-Series Models</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Setup ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ Station Location")
    # Clean Mapbox dark themed layout
    map_df = pd.DataFrame({'lat': [MELB_LAT], 'lon': [MELB_LON]})
    st.map(map_df, zoom=10, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧬 Metadata")
    st.markdown(f"""
    - **Records:** {len(df):,} days
    - **Period:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
    - **Mean Temp:** {df['Temperature'].mean():.1f} °C
    - **Latitude:** {MELB_LAT}
    - **Longitude:** {MELB_LON}
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Quick System Check")
    if model is not None:
        st.success("🤖 Core AI Model: Active")
    else:
        st.error("🤖 Core AI Model: Offline")
        
    if scaler is not None:
        st.success("📏 Data Normalizer: Ready")
    else:
        st.error("📏 Data Normalizer: Missing")

# ── Dashboard Tabs ─────────────────────────────────────────────────────
tab_forecast, tab_analytics, tab_brain = st.tabs([
    "🔮 Forecast Center", 
    "📊 Climate Insights (EDA)", 
    "🧠 Brain (Diagnostics)"
])

# ───────────────────────────────────────────────────────────────────────
# TAB 1: Forecast Center
# ───────────────────────────────────────────────────────────────────────
with tab_forecast:
    st.subheader("Predict Future Minimum Temperature")
    
    col_input, col_result = st.columns([2, 3])
    
    with col_input:
        st.markdown("""
        <div class="glass-card">
            <h4>1. Define Lookback Scenario</h4>
            <p style="font-size:0.85rem; color:#94a3b8;">Select a preset meteorological scenario or manually configure values below.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Scenario Selector
        preset = st.selectbox(
            "Load Historical Preset Scenario",
            ["Custom Configurations", "1982 Winter Freeze", "1982 Midsummer Heatwave", "1989 Cool Spring", "1990 Autumn Freeze"]
        )
        
        # Populate values based on preset selection
        if preset == "1982 Winter Freeze":
            # Real freeze period around June 1982
            defaults = [2.2, 1.8, 0.2, 0.8, 2.5, 3.1, 1.4]
        elif preset == "1982 Midsummer Heatwave":
            # Hot period around Feb 1982
            defaults = [22.4, 21.0, 24.3, 26.3, 23.5, 20.8, 19.5]
        elif preset == "1989 Cool Spring":
            defaults = [11.2, 10.4, 9.8, 8.5, 9.2, 10.1, 11.5]
        elif preset == "1990 Autumn Freeze":
            defaults = [8.4, 7.2, 6.5, 5.1, 4.2, 4.8, 5.5]
        else:
            # Default values (last 7 days of the dataset)
            defaults = [float(round(df['Temperature'].iloc[-(7-i)], 1)) for i in range(7)]

        # Custom inputs grid
        st.markdown("##### Manual Fine-Tuning")
        input_temps = []
        cols_grid = st.columns(4)
        for i in range(7):
            col_target = cols_grid[i % 4]
            with col_target:
                val = st.number_input(
                    f"Day t-{7-i}",
                    min_value=-5.0,
                    max_value=35.0,
                    value=defaults[i],
                    step=0.1,
                    key=f"slider_day_{i}"
                )
                input_temps.append(val)
                
    with col_result:
        if model is not None and scaler is not None:
            # Perform calculations
            input_array = np.array(input_temps).reshape(-1, 1)

            # Apply log transform if configured during training
            use_log = champion_info.get('use_log_transform', False) if champion_info else False
            if use_log:
                log_offset = champion_info.get('log_offset', 1.0)
                input_array = np.log(input_array + log_offset)

            input_scaled = scaler.transform(input_array).flatten()
            input_reshaped = input_scaled.reshape(1, WINDOW_SIZE, 1)

            pred_scaled = model.predict(input_reshaped, verbose=0)
            pred_temp = scaler.inverse_transform(pred_scaled)[0][0]

            # Inverse log-transform if applicable
            if use_log:
                log_offset = champion_info.get('log_offset', 1.0)
                pred_temp = float(np.exp(pred_temp) - log_offset)

            # Predict display card
            st.markdown(f"""
            <div class="prediction-card">
                <div class="pred-label">Tomorrow's Temperature Forecast (t+1)</div>
                <div class="pred-value">{pred_temp:.1f}°C</div>
                <div class="pred-unit">Based on historical lookback sequence</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Plotly Trend visualization
            fig_input = go.Figure()
            days = [f"t-{7-i}" for i in range(7)] + ["t+1 (Pred)"]
            temps_with_pred = input_temps + [round(pred_temp, 2)]
            
            # Glowing bar styling
            fig_input.add_trace(go.Bar(
                x=days[:-1], y=temps_with_pred[:-1],
                marker=dict(color='#475569', line=dict(color='#64748b', width=1)),
                name='Input Sequence',
                text=[f"{t}°C" for t in temps_with_pred[:-1]],
                textposition='auto',
            ))
            
            fig_input.add_trace(go.Bar(
                x=[days[-1]], y=[temps_with_pred[-1]],
                marker=dict(color='#0ea5e9', line=dict(color='#38bdf8', width=1.5)),
                name='AI Forecast',
                text=[f"{temps_with_pred[-1]}°C"],
                textposition='auto',
            ))
            
            fig_input.update_layout(
                title=dict(text="Lookback Sequence Trend & Predicted Shift", font=dict(color="#f8fafc", size=15)),
                yaxis_title="Minimum Temperature (°C)",
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=320,
                margin=dict(t=50, b=20, l=10, r=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_input, use_container_width=True)
        else:
            st.warning("⚠️ Critical system modules are missing. Model failed to execute.")

# ───────────────────────────────────────────────────────────────────────
# TAB 2: Climate Insights (EDA)
# ───────────────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader("Historical Exploratory Climate Analysis")
    
    # Timeline
    with st.expander("📈 Decadal Historical Timeline & Rolling Cycles", expanded=True):
        ma7 = df['Temperature'].rolling(7, center=True).mean()
        ma30 = df['Temperature'].rolling(30, center=True).mean()

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=df.index, y=df['Temperature'],
            name='Daily Observations', mode='lines',
            line=dict(color='rgba(148, 163, 184, 0.25)', width=0.8),
        ))
        fig_timeline.add_trace(go.Scatter(
            x=df.index, y=ma7,
            name='7-Day Cycle (Weekly)', mode='lines',
            line=dict(color='#ff7043', width=1.3),
        ))
        fig_timeline.add_trace(go.Scatter(
            x=df.index, y=ma30,
            name='30-Day Cycle (Monthly)', mode='lines',
            line=dict(color='#38bdf8', width=2.2),
        ))
        
        fig_timeline.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(15,23,42,0.8)'),
            hovermode='x unified',
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # Monthly Averages
    col_bar1, col_bar2 = st.columns(2)
    with col_bar1:
        with st.expander("📊 Monthly Inter-Annual Aggregates", expanded=True):
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
                labels={'Temperature': 'Avg Temp (°C)', 'Month_Name': 'Month'},
                color_continuous_scale='Turbo',
                category_orders={'Month_Name': list(month_names.values())},
            )
            fig_monthly.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=380,
                margin=dict(t=20, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
    with col_bar2:
        with st.expander("📊 Global Climatology Mean", expanded=True):
            overall_monthly = df_month.groupby('Month')['Temperature'].mean()
            fig_overall = go.Figure(go.Bar(
                x=[month_names[m] for m in overall_monthly.index],
                y=overall_monthly.values,
                marker=dict(color=overall_monthly.values, colorscale='RdBu_r'),
                text=[f"{v:.1f}°C" for v in overall_monthly.values],
                textposition='outside',
            ))
            fig_overall.update_layout(
                yaxis_title='Avg Min Temp (°C)',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=380,
                showlegend=False,
                margin=dict(t=20, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_overall, use_container_width=True)

    # Distributions & Autocorrelation
    col_dist, col_lag = st.columns(2)
    with col_dist:
        with st.expander("📉 Temperature Density Profile", expanded=True):
            fig_hist = px.histogram(
                df, x='Temperature', nbins=50, marginal='box',
                color_discrete_sequence=['#38bdf8'],
            )
            fig_hist.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=380,
                margin=dict(t=20, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
    with col_lag:
        with st.expander("🔗 Temporal Lag Correlation Patterns", expanded=True):
            fig_lag1 = go.Figure(go.Scatter(
                x=df['Temperature'].iloc[:-1].values,
                y=df['Temperature'].iloc[1:].values,
                mode='markers',
                marker=dict(color='#818cf8', size=3, opacity=0.3),
                name='Observations',
            ))
            rng = [df['Temperature'].min() - 1, df['Temperature'].max() + 1]
            fig_lag1.add_trace(go.Scatter(
                x=rng, y=rng, mode='lines',
                line=dict(color='#ef4444', dash='dash', width=1.5),
                name='Identity Line (y=x)',
            ))
            fig_lag1.update_layout(
                xaxis_title='Temperature (t-1) °C', yaxis_title='Temperature (t) °C',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=380,
                margin=dict(t=20, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_lag1, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────
# TAB 3: Brain (Diagnostics)
# ───────────────────────────────────────────────────────────────────────
with tab_brain:
    st.subheader("Model Validation & Neural Architecture Diagnostics")
    
    if champion_info:
        # Main Metrics Row
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Champion Framework</div>
                <div class="kpi-value">{champion_info.get('overall_champion', 'N/A')}</div>
                <div class="kpi-sub">Optimal Benchmark Model</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Deployment RMSE</div>
                <div class="kpi-value">{champion_info.get('deploy_rmse', 0):.3f}</div>
                <div class="kpi-sub">Standard Deviation of Error</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Deployment MAE</div>
                <div class="kpi-value">{champion_info.get('deploy_mae', 0):.3f}</div>
                <div class="kpi-sub">Average Absolute Error Deviation</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Coefficient of Determination (R²)</div>
                <div class="kpi-value">{champion_info.get('deploy_r2', 0):.3f}</div>
                <div class="kpi-sub">Variance Explained by AI</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Grid layout for deep diagnostics
        col_m_tbl, col_m_params = st.columns([3, 2])
        
        with col_m_tbl:
            st.markdown("##### 📋 Global Validation Comparison Table")
            if champion_info.get('metrics'):
                full_metrics = pd.DataFrame(champion_info['metrics'])
                st.dataframe(full_metrics, use_container_width=True, hide_index=True)
                
        with col_m_params:
            st.markdown("##### ⚙️ Active Training Hyperparameters")
            st.markdown(f"""
            - **Lookback Window Size:** `{champion_info.get('window_size', 7)} days`
            - **ADF Optimal Differencing Order (d):** `{champion_info.get('best_sarima_order', [0,0,0])[1]}`
            - **Selected Deep Network type:** `{champion_info.get('deploy_model_name', 'LSTM')}`
            - **ADF Stationarity Log-Shift:** `{"Applied" if champion_info.get('use_log_transform') else "Bypassed"}`
            - **Variance Offset constant:** `{champion_info.get('log_offset', 0.0):.2f}`
            """)
            
        st.markdown("---")
        
        # Residual Analysis / Training Curves Visualizations
        st.markdown("##### 🧠 Neural Model Training Optimization Curves")
        col_hist_ch1, col_hist_ch2 = st.columns(2)
        
        with col_hist_ch1:
            st.image('lstm_training.png', caption='LSTM Training Convergence Curve (MSE/MAE vs Epochs)', use_container_width=True)
            
        with col_hist_ch2:
            st.image('gru_training.png', caption='GRU Training Convergence Curve (MSE/MAE vs Epochs)', use_container_width=True)
            
    else:
        st.info("💡 Run the ML training pipeline locally or in Colab to populate this diagnostic screen with production telemetry.")

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; opacity:0.4; padding: 2rem 0; border-top: 1px solid rgba(255,255,255,0.05); margin-top:3rem;">
    <p style="font-size:0.75rem;">Melbourne Temperature Predictor System v2.0.0<br>
    Engineered using Python 3.11 • Streamlit Server Interface • TensorFlow Backends</p>
</div>
""", unsafe_allow_html=True)

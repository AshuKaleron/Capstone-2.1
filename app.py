# =============================================================================
# 🌡️ Melbourne Climate Intelligence Platform — Capstone Dashboard (v5 ULTRA)
# =============================================================================
# AI-powered climate analytics with predictive forecasting, exhaustive EDA,
# model intelligence diagnostics, statistical decomposition, climate risk
# scoring, extreme event analysis, what-if simulator, and data laboratory.
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta
import math

# ── Page Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Melbourne Climate Intelligence Platform",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS Design System v5 ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ──── Global Reset ──── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #e2e8f0;
}

.stApp {
    background: linear-gradient(180deg, #060a13 0%, #0a0f1a 50%, #090d16 100%);
}

/* ──── Sidebar ──── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1220 0%, #0f172a 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.08);
}

/* ──── Hero Banner ──── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 24px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.05);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.1) 0%, transparent 70%);
    pointer-events: none;
    animation: pulse-glow 6s ease-in-out infinite;
}

.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(129, 140, 248, 0.08) 0%, transparent 70%);
    pointer-events: none;
    animation: pulse-glow 8s ease-in-out infinite reverse;
}

@keyframes pulse-glow {
    0%, 100% { opacity: 0.5; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.15); }
}

.hero-banner h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1.2px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
    z-index: 2;
}

.hero-banner .hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.3px;
    position: relative;
    z-index: 2;
}

.hero-banner .hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 1rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 2;
}

.hero-banner .badge {
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #38bdf8;
    font-weight: 500;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
}

.hero-banner .badge:hover {
    background: rgba(56, 189, 248, 0.2);
    transform: translateY(-1px);
}

/* ──── Glass Cards ──── */
.glass-card {
    background: rgba(15, 23, 42, 0.5);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 1.8rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
}

/* ──── KPI Metric Cards ──── */
.kpi-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.7), rgba(30, 41, 59, 0.4));
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.3);
    border-color: rgba(56, 189, 248, 0.15);
}

.kpi-card .kpi-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
.kpi-card .kpi-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.8px; color: #64748b; margin-bottom: 0.4rem; }
.kpi-card .kpi-value { font-family: 'Outfit', sans-serif; font-size: 1.9rem; font-weight: 700; color: #f1f5f9; line-height: 1.1; }
.kpi-card .kpi-sub { font-size: 0.7rem; color: #475569; margin-top: 0.2rem; }

/* Colored KPI variants */
.kpi-cyan .kpi-value { color: #38bdf8; }
.kpi-violet .kpi-value { color: #a78bfa; }
.kpi-amber .kpi-value { color: #fbbf24; }
.kpi-emerald .kpi-value { color: #34d399; }
.kpi-rose .kpi-value { color: #fb7185; }
.kpi-blue .kpi-value { color: #60a5fa; }
.kpi-orange .kpi-value { color: #f97316; }
.kpi-pink .kpi-value { color: #e879f9; }

/* ──── Prediction Card ──── */
.prediction-card {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(99, 102, 241, 0.1) 100%);
    border: 1.5px solid rgba(56, 189, 248, 0.3);
    border-radius: 24px;
    padding: 2.5rem 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 20px 40px rgba(14, 165, 233, 0.12), inset 0 1px 0 rgba(255,255,255,0.05);
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.05) 0%, transparent 50%);
    animation: shimmer 8s ease-in-out infinite;
    pointer-events: none;
}

@keyframes shimmer {
    0%, 100% { transform: rotate(0deg); }
    50% { transform: rotate(180deg); }
}

.prediction-card .pred-label { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2.5px; color: #38bdf8; font-weight: 600; position: relative; z-index: 2; }
.prediction-card .pred-value { font-family: 'Outfit', sans-serif; font-size: 4.5rem; font-weight: 800; margin: 0.4rem 0; color: #ffffff; text-shadow: 0 0 30px rgba(56, 189, 248, 0.35); position: relative; z-index: 2; }
.prediction-card .pred-sub { font-size: 0.85rem; color: #94a3b8; font-weight: 300; position: relative; z-index: 2; }

/* ──── Risk Score Badge ──── */
.risk-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 30px;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1px;
}
.risk-low { background: rgba(52,211,153,0.15); border: 1px solid rgba(52,211,153,0.3); color: #34d399; }
.risk-moderate { background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.3); color: #fbbf24; }
.risk-high { background: rgba(251,113,133,0.15); border: 1px solid rgba(251,113,133,0.3); color: #fb7185; }
.risk-extreme { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); color: #ef4444; }

/* ──── Tabs ──── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: transparent;
    padding: 0;
    overflow-x: auto;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 14px 14px 0px 0px;
    color: #64748b;
    padding: 10px 18px;
    font-weight: 500;
    font-size: 0.82rem;
    transition: all 0.2s ease;
    white-space: nowrap;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #94a3b8;
    background-color: rgba(30, 41, 59, 0.4);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: rgba(56, 189, 248, 0.1) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    font-weight: 600;
}

/* ──── Section Headers ──── */
.section-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid rgba(56, 189, 248, 0.15);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ──── Expander Styling ──── */
div[data-testid="stExpander"] {
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    margin-bottom: 1rem;
    background: rgba(15, 23, 42, 0.3);
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

/* ──── Dataframe Styling ──── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ──── Download Button ──── */
.stDownloadButton button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.stDownloadButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(14, 165, 233, 0.3) !important;
}

/* ──── Insight Box ──── */
.insight-box {
    background: linear-gradient(135deg, rgba(129, 140, 248, 0.08) 0%, rgba(56, 189, 248, 0.05) 100%);
    border: 1px solid rgba(129, 140, 248, 0.15);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.6;
}

.insight-box .insight-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
}

/* ──── Stat Row ──── */
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem;
}
.stat-row .stat-label { color: #64748b; }
.stat-row .stat-value { color: #e2e8f0; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

/* ──── Footer ──── */
.app-footer {
    text-align: center;
    opacity: 0.35;
    padding: 2.5rem 0 1rem 0;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    margin-top: 3rem;
    font-size: 0.72rem;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_dataset():
    """Load, clean, and return the Melbourne temperature dataset."""
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
    """Load the Keras champion model with backward compatibility."""
    if not os.path.exists('champion_model.keras'):
        return None
    try:
        import tf_keras as keras
        return keras.models.load_model('champion_model.keras')
    except Exception:
        try:
            import tensorflow as tf
            return tf.keras.models.load_model('champion_model.keras')
        except Exception as e:
            st.error(f"❌ Model load error: {e}")
            return None


@st.cache_resource
def load_scaler():
    """Load the MinMaxScaler fitted during training."""
    if not os.path.exists('scaler.pkl'):
        return None
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Scaler load error: {e}")
        return None


@st.cache_data
def load_champion_info():
    """Load champion metadata and evaluation metrics."""
    if not os.path.exists('champion_info.pkl'):
        return None
    try:
        with open('champion_info.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


# ── Helper Functions ───────────────────────────────────────────────────

def predict_single(model, scaler, input_temps, use_log, log_offset, window_size):
    """Run a single next-day prediction through the full pipeline."""
    inp = np.array(input_temps, dtype=np.float64).reshape(-1, 1)
    if use_log:
        inp = np.log(inp + log_offset)
    inp_scaled = scaler.transform(inp).flatten()
    inp_reshaped = inp_scaled.reshape(1, window_size, 1)
    pred_scaled = model.predict(inp_reshaped, verbose=0)
    pred_val = scaler.inverse_transform(pred_scaled)[0][0]
    if use_log:
        pred_val = float(np.exp(pred_val) - log_offset)
    return float(pred_val)


def multi_step_forecast(model, scaler, initial_temps, n_steps, use_log, log_offset, window_size):
    """Iterative multi-step rolling forecast."""
    window = np.array(initial_temps, dtype=np.float64).copy()
    predictions = []
    for _ in range(n_steps):
        pred = predict_single(model, scaler, window, use_log, log_offset, window_size)
        predictions.append(pred)
        window = np.append(window[1:], pred)
    return predictions


def compute_acf(series, nlags=30):
    """Compute the autocorrelation function without statsmodels."""
    vals = series.dropna().values
    n = len(vals)
    mean_val = np.mean(vals)
    variance = np.sum((vals - mean_val) ** 2) / n
    if variance == 0:
        return [0.0] * (nlags + 1)
    acf_vals = []
    for lag in range(nlags + 1):
        cov = np.sum((vals[:n - lag] - mean_val) * (vals[lag:] - mean_val)) / n
        acf_vals.append(cov / variance)
    return acf_vals


def compute_pacf(series, nlags=30):
    """Compute partial autocorrelation using Durbin-Levinson recursion."""
    acf_vals = compute_acf(series, nlags)
    pacf_vals = [1.0]  # lag 0

    if nlags == 0:
        return pacf_vals

    # Durbin-Levinson algorithm
    phi = [[0.0] * (nlags + 1) for _ in range(nlags + 1)]

    phi[1][1] = acf_vals[1]
    pacf_vals.append(phi[1][1])

    for k in range(2, nlags + 1):
        num = acf_vals[k]
        for j in range(1, k):
            num -= phi[k - 1][j] * acf_vals[k - j]
        denom = 1.0
        for j in range(1, k):
            denom -= phi[k - 1][j] * acf_vals[j]
        if abs(denom) < 1e-12:
            pacf_vals.append(0.0)
            continue
        phi[k][k] = num / denom
        for j in range(1, k):
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j]
        pacf_vals.append(phi[k][k])

    return pacf_vals


def compute_test_predictions(model, scaler, df, champion_info):
    """Reconstruct test-set predictions live from the deployed model."""
    window_size = champion_info.get('window_size', 7)
    split_idx = champion_info.get('split_index', int(len(df) * 0.8))
    use_log = champion_info.get('use_log_transform', False)
    log_offset = champion_info.get('log_offset', 1.0)

    temp_values = df['Temperature'].values.astype(np.float64)
    if use_log:
        modeling_values = np.log(temp_values + log_offset)
    else:
        modeling_values = temp_values.copy()

    scaled = scaler.transform(modeling_values.reshape(-1, 1)).flatten()

    train_window_count = split_idx - window_size
    X_test = []
    for i in range(train_window_count, len(scaled) - window_size):
        X_test.append(scaled[i:i + window_size])

    if len(X_test) == 0:
        return None, None, None

    X_test = np.array(X_test).reshape(-1, window_size, 1)
    pred_scaled = model.predict(X_test, verbose=0, batch_size=64)
    pred = scaler.inverse_transform(pred_scaled).flatten()

    if use_log:
        pred = np.exp(pred) - log_offset

    n_pred = len(pred)
    actual = temp_values[split_idx:split_idx + n_pred]
    dates = df.index[split_idx:split_idx + n_pred]
    return dates, actual, pred


def seasonal_decompose_manual(series, period=365):
    """Manual additive seasonal decomposition without statsmodels."""
    values = series.values.astype(float)
    n = len(values)

    # Trend via centered moving average
    trend = pd.Series(values, index=series.index).rolling(
        window=period, center=True, min_periods=1
    ).mean().values

    # Detrended
    detrended = values - trend

    # Seasonal component: average by position in cycle
    seasonal = np.zeros(n)
    for i in range(period):
        indices = list(range(i, n, period))
        seasonal_val = np.nanmean(detrended[indices])
        for idx in indices:
            seasonal[idx] = seasonal_val

    # Residual
    residual = values - trend - seasonal

    return trend, seasonal, residual


def compute_climate_risk_score(df_ext):
    """Compute a composite climate risk score (0-100)."""
    temp = df_ext['Temperature']

    # Factor 1: Extreme cold frequency (days below 2°C)
    extreme_cold_pct = (temp < 2.0).sum() / len(temp) * 100

    # Factor 2: Temperature volatility (std of daily changes)
    daily_change = temp.diff().dropna()
    volatility = daily_change.std()

    # Factor 3: Anomaly intensity (mean absolute anomaly)
    if 'Anomaly' in df_ext.columns:
        anomaly_intensity = df_ext['Anomaly'].abs().mean()
    else:
        monthly_means = df_ext.groupby(df_ext.index.month)['Temperature'].transform('mean')
        anomaly_intensity = (temp - monthly_means).abs().mean()

    # Factor 4: Range expansion
    range_val = temp.max() - temp.min()

    # Normalize and combine (each factor 0-25 points)
    score = 0
    score += min(extreme_cold_pct * 2, 25)
    score += min(volatility * 7, 25)
    score += min(anomaly_intensity * 5, 25)
    score += min(range_val * 0.8, 25)

    return min(round(score, 1), 100)


# ── Plotly Theme Helper ────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans, sans-serif', color='#94a3b8'),
    margin=dict(t=40, b=30, l=50, r=20),
)

MONTH_MAP = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
             7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

SEASON_MAP = {12: 'Summer', 1: 'Summer', 2: 'Summer',
              3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
              6: 'Winter', 7: 'Winter', 8: 'Winter',
              9: 'Spring', 10: 'Spring', 11: 'Spring'}

SEASON_COLORS = {'Summer': '#ef4444', 'Autumn': '#f97316', 'Winter': '#38bdf8', 'Spring': '#34d399'}


# ══════════════════════════════════════════════════════════════════════
#  LOAD EVERYTHING
# ══════════════════════════════════════════════════════════════════════

df = load_dataset()
model = load_model()
scaler = load_scaler()
champion_info = load_champion_info()

WINDOW_SIZE = champion_info.get('window_size', 7) if champion_info else 7
USE_LOG = champion_info.get('use_log_transform', False) if champion_info else False
LOG_OFFSET = champion_info.get('log_offset', 1.0) if champion_info else 1.0
MELB_LAT, MELB_LON = -37.8136, 144.9631

# Precompute commonly-used derived columns
df_ext = df.copy()
df_ext['Month'] = df_ext.index.month
df_ext['Year'] = df_ext.index.year
df_ext['DayOfYear'] = df_ext.index.dayofyear
df_ext['MonthName'] = df_ext['Month'].map(MONTH_MAP)
df_ext['Season'] = df_ext['Month'].map(SEASON_MAP)
df_ext['DayOfWeek'] = df_ext.index.dayofweek
df_ext['WeekOfYear'] = df_ext.index.isocalendar().week.astype(int)
df_ext['Quarter'] = df_ext.index.quarter
monthly_means_global = df_ext.groupby('Month')['Temperature'].mean()
df_ext['Anomaly'] = df_ext['Temperature'] - df_ext['Month'].map(monthly_means_global)
df_ext['TempChange'] = df_ext['Temperature'].diff()
df_ext['RollingStd7'] = df_ext['Temperature'].rolling(7).std()
df_ext['RollingStd30'] = df_ext['Temperature'].rolling(30).std()


# ══════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════════════

years_span = df.index.max().year - df.index.min().year + 1

st.markdown(f"""
<div class="hero-banner">
    <h1>🌡️ Melbourne Climate Intelligence Platform</h1>
    <div class="hero-sub">
        AI-powered minimum temperature forecasting system with deep-learning inference,
        exhaustive exploratory analysis, statistical decomposition, climate risk assessment,
        and real-time model diagnostics.
    </div>
    <div class="hero-badges">
        <span class="badge">LSTM / GRU</span>
        <span class="badge">SARIMA</span>
        <span class="badge">{years_span} Years of Data</span>
        <span class="badge">TensorFlow</span>
        <span class="badge">{len(df):,} Records</span>
        <span class="badge">8 Analytics Modules</span>
        <span class="badge">Capstone Project</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🗺️ Weather Station")
    map_df = pd.DataFrame({'lat': [MELB_LAT], 'lon': [MELB_LON]})
    st.map(map_df, zoom=10, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Dataset Card")
    st.markdown(f"""
    | Property | Value |
    |:---|:---|
    | **Records** | {len(df):,} days |
    | **Start** | {df.index.min().strftime('%Y-%m-%d')} |
    | **End** | {df.index.max().strftime('%Y-%m-%d')} |
    | **Mean** | {df['Temperature'].mean():.2f} °C |
    | **Std Dev** | {df['Temperature'].std():.2f} °C |
    | **Min** | {df['Temperature'].min():.1f} °C |
    | **Max** | {df['Temperature'].max():.1f} °C |
    | **Median** | {df['Temperature'].median():.1f} °C |
    | **Skewness** | {df['Temperature'].skew():.3f} |
    | **Kurtosis** | {df['Temperature'].kurtosis():.3f} |
    """)

    st.markdown("---")
    st.markdown("### 🔌 System Status")
    if df is not None:
        st.success("✅ Dataset Loaded")
    else:
        st.error("❌ Dataset Missing")
    if model is not None:
        st.success("✅ AI Model Active")
    else:
        st.error("❌ AI Model Offline")
    if scaler is not None:
        st.success("✅ Scaler Ready")
    else:
        st.error("❌ Scaler Missing")
    if champion_info is not None:
        st.success("✅ Metrics Loaded")
    else:
        st.warning("⚠️ No Metrics")

    if champion_info:
        st.markdown("---")
        st.markdown("### 🏆 Deployed Model")
        deploy_name = champion_info.get('deploy_model_name', 'N/A')
        overall_champ = champion_info.get('overall_champion', '')
        deploy_rmse = champion_info.get('deploy_rmse', 0)
        deploy_r2 = champion_info.get('deploy_r2', 0)
        deploy_mae = champion_info.get('deploy_mae', 0)
        st.markdown(f"**{deploy_name}** ({overall_champ})")
        st.markdown(f"RMSE: `{deploy_rmse:.4f}` · MAE: `{deploy_mae:.4f}` · R²: `{deploy_r2:.4f}`")

    st.markdown("---")
    st.markdown("### 🌡️ Climate Risk Score")
    risk_score = compute_climate_risk_score(df_ext)
    if risk_score < 30:
        risk_class, risk_label = "risk-low", "LOW"
    elif risk_score < 55:
        risk_class, risk_label = "risk-moderate", "MODERATE"
    elif risk_score < 80:
        risk_class, risk_label = "risk-high", "HIGH"
    else:
        risk_class, risk_label = "risk-extreme", "EXTREME"
    st.markdown(f'<div style="text-align:center;"><span class="risk-badge {risk_class}">{risk_label} — {risk_score}/100</span></div>', unsafe_allow_html=True)
    st.caption("Composite score based on extreme cold frequency, temperature volatility, anomaly intensity, and range expansion.")


# ══════════════════════════════════════════════════════════════════════
#  MAIN TABS (8 Tabs)
# ══════════════════════════════════════════════════════════════════════

tab_overview, tab_forecast, tab_analytics, tab_decomp, tab_extreme, tab_brain, tab_whatif, tab_lab = st.tabs([
    "🏠 Overview",
    "🔮 Forecast Engine",
    "📊 Climate Deep Dive",
    "🔬 Decomposition",
    "⚡ Extreme Events",
    "🧠 Model Intelligence",
    "🧪 What-If Simulator",
    "📥 Data Laboratory",
])


# ──────────────────────────────────────────────────────────────────────
#  TAB 1 — EXECUTIVE OVERVIEW
# ──────────────────────────────────────────────────────────────────────

with tab_overview:

    # ── KPI Row 1 ──
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, "📅", "Total Records", f"{len(df):,}", "daily observations", "kpi-cyan"),
        (k2, "📆", "Date Span", f"{years_span} Years", f"{df.index.min().year}–{df.index.max().year}", "kpi-violet"),
        (k3, "🌡️", "Mean Temp", f"{df['Temperature'].mean():.1f}°C", "all-time average", "kpi-amber"),
        (k4, "📊", "Std Deviation", f"{df['Temperature'].std():.2f}°C", "volatility measure", "kpi-emerald"),
        (k5, "❄️", "Record Low", f"{df['Temperature'].min():.1f}°C", f"{df['Temperature'].idxmin().strftime('%Y-%m-%d')}", "kpi-rose"),
        (k6, "☀️", "Record High", f"{df['Temperature'].max():.1f}°C", f"{df['Temperature'].idxmax().strftime('%Y-%m-%d')}", "kpi-blue"),
    ]
    for col, icon, label, value, sub, cls in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── KPI Row 2 ──
    st.markdown("<br>", unsafe_allow_html=True)
    k7, k8, k9, k10 = st.columns(4)
    iqr_val = df['Temperature'].quantile(0.75) - df['Temperature'].quantile(0.25)
    cv_val = (df['Temperature'].std() / df['Temperature'].mean()) * 100
    kpis2 = [
        (k7, "📐", "Median", f"{df['Temperature'].median():.1f}°C", "50th percentile", "kpi-orange"),
        (k8, "📏", "IQR", f"{iqr_val:.2f}°C", "Q3 − Q1 spread", "kpi-pink"),
        (k9, "🔄", "CoV", f"{cv_val:.1f}%", "coefficient of variation", "kpi-cyan"),
        (k10, "📈", "Daily Δ Std", f"{df_ext['TempChange'].std():.2f}°C", "day-to-day volatility", "kpi-violet"),
    ]
    for col, icon, label, value, sub, cls in kpis2:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Year × Month Heatmap ──
    col_heat, col_trend = st.columns([3, 2])

    with col_heat:
        st.markdown('<div class="section-header">🗓️ Year × Month Temperature Heatmap</div>', unsafe_allow_html=True)
        pivot = df_ext.pivot_table(values='Temperature', index='Year', columns='Month', aggfunc='mean')
        pivot.columns = [MONTH_MAP[m] for m in pivot.columns]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index.tolist()],
            colorscale='RdBu_r',
            reversescale=True,
            text=np.round(pivot.values, 1),
            texttemplate='%{text}',
            textfont=dict(size=10),
            colorbar=dict(title='°C', thickness=15),
        ))
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=380, yaxis=dict(dtick=1))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_trend:
        st.markdown('<div class="section-header">📈 Annual Mean Temperature Trend</div>', unsafe_allow_html=True)
        annual = df_ext.groupby('Year')['Temperature'].agg(['mean', 'std']).reset_index()

        fig_annual = go.Figure()
        fig_annual.add_trace(go.Scatter(
            x=annual['Year'], y=annual['mean'] + annual['std'],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))
        fig_annual.add_trace(go.Scatter(
            x=annual['Year'], y=annual['mean'] - annual['std'],
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(56,189,248,0.1)', showlegend=False, hoverinfo='skip',
        ))
        fig_annual.add_trace(go.Scatter(
            x=annual['Year'], y=annual['mean'],
            mode='lines+markers',
            line=dict(color='#38bdf8', width=2.5),
            marker=dict(size=8, color='#0ea5e9', line=dict(color='#38bdf8', width=1.5)),
            name='Annual Mean',
            text=[f"{v:.2f}°C" for v in annual['mean']],
            textposition='top center',
        ))
        # Add linear trendline
        years_num = annual['Year'].values.astype(float)
        z = np.polyfit(years_num, annual['mean'].values, 1)
        p = np.poly1d(z)
        fig_annual.add_trace(go.Scatter(
            x=annual['Year'], y=p(years_num),
            mode='lines', line=dict(color='#fb7185', width=1.5, dash='dash'),
            name=f'Trend ({z[0]:+.3f}°C/yr)',
        ))
        fig_annual.update_layout(**PLOTLY_LAYOUT, height=380,
                                  xaxis=dict(dtick=1), yaxis_title='Temperature (°C)',
                                  legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'))
        st.plotly_chart(fig_annual, use_container_width=True)

    # ── Temperature Anomaly Timeline ──
    st.markdown('<div class="section-header">🌊 Daily Temperature Anomaly (deviation from monthly mean)</div>', unsafe_allow_html=True)
    fig_anom = go.Figure()
    pos_mask = df_ext['Anomaly'] >= 0
    fig_anom.add_trace(go.Bar(
        x=df_ext.index[pos_mask], y=df_ext['Anomaly'][pos_mask],
        marker_color='rgba(251, 113, 133, 0.5)', name='Above Average',
    ))
    fig_anom.add_trace(go.Bar(
        x=df_ext.index[~pos_mask], y=df_ext['Anomaly'][~pos_mask],
        marker_color='rgba(56, 189, 248, 0.5)', name='Below Average',
    ))
    fig_anom.update_layout(**PLOTLY_LAYOUT, height=300, barmode='relative',
                            yaxis_title='Anomaly (°C)',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'))
    st.plotly_chart(fig_anom, use_container_width=True)

    # ── Seasonal Summary Cards ──
    st.markdown('<div class="section-header">🍃 Seasonal Summary</div>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, season in zip([sc1, sc2, sc3, sc4], ['Summer', 'Autumn', 'Winter', 'Spring']):
        s_data = df_ext[df_ext['Season'] == season]['Temperature']
        color = SEASON_COLORS[season]
        with col:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 3px solid {color};">
                <div style="font-family:'Outfit'; font-weight:600; color:{color}; font-size:1rem; margin-bottom:0.5rem;">
                    {'☀️' if season=='Summer' else '🍂' if season=='Autumn' else '❄️' if season=='Winter' else '🌸'} {season}
                </div>
                <div class="stat-row"><span class="stat-label">Mean</span><span class="stat-value">{s_data.mean():.1f}°C</span></div>
                <div class="stat-row"><span class="stat-label">Min</span><span class="stat-value">{s_data.min():.1f}°C</span></div>
                <div class="stat-row"><span class="stat-label">Max</span><span class="stat-value">{s_data.max():.1f}°C</span></div>
                <div class="stat-row"><span class="stat-label">Std</span><span class="stat-value">{s_data.std():.2f}°C</span></div>
                <div class="stat-row"><span class="stat-label">Days</span><span class="stat-value">{len(s_data):,}</span></div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
#  TAB 2 — AI FORECAST ENGINE
# ──────────────────────────────────────────────────────────────────────

with tab_forecast:

    if model is None or scaler is None:
        st.warning("⚠️ The AI model or scaler is not available. Upload `champion_model.keras` and `scaler.pkl` to enable forecasting.")
    else:
        col_config, col_output = st.columns([2, 3])

        with col_config:
            st.markdown('<div class="section-header">⚙️ Configure Input Sequence</div>', unsafe_allow_html=True)

            preset = st.selectbox("Load Preset Scenario", [
                "Manual Input (Last 7 Days)",
                "❄️ 1982 Winter Freeze",
                "☀️ 1982 Midsummer Heatwave",
                "🍂 1990 Autumn Chill",
                "🌸 1989 Cool Spring",
                "📊 Dataset Average Week",
                "🌡️ Record Cold Week",
                "🔥 Record Warm Week",
            ])

            if preset == "❄️ 1982 Winter Freeze":
                defaults = [2.2, 1.8, 0.2, 0.8, 2.5, 3.1, 1.4]
            elif preset == "☀️ 1982 Midsummer Heatwave":
                defaults = [22.4, 21.0, 24.3, 26.3, 23.5, 20.8, 19.5]
            elif preset == "🍂 1990 Autumn Chill":
                defaults = [8.4, 7.2, 6.5, 5.1, 4.2, 4.8, 5.5]
            elif preset == "🌸 1989 Cool Spring":
                defaults = [11.2, 10.4, 9.8, 8.5, 9.2, 10.1, 11.5]
            elif preset == "📊 Dataset Average Week":
                avg = df['Temperature'].mean()
                defaults = [round(avg, 1)] * 7
            elif preset == "🌡️ Record Cold Week":
                cold_idx = df['Temperature'].idxmin()
                cold_pos = df.index.get_loc(cold_idx)
                start = max(0, cold_pos - 6)
                defaults = [round(float(v), 1) for v in df['Temperature'].iloc[start:start+7].values]
                if len(defaults) < 7:
                    defaults = defaults + [defaults[-1]] * (7 - len(defaults))
            elif preset == "🔥 Record Warm Week":
                warm_idx = df['Temperature'].idxmax()
                warm_pos = df.index.get_loc(warm_idx)
                start = max(0, warm_pos - 6)
                defaults = [round(float(v), 1) for v in df['Temperature'].iloc[start:start+7].values]
                if len(defaults) < 7:
                    defaults = defaults + [defaults[-1]] * (7 - len(defaults))
            else:
                defaults = [float(round(df['Temperature'].iloc[-(7 - i)], 1)) for i in range(7)]

            st.markdown("##### Fine-Tune Each Day")
            input_temps = []
            row1 = st.columns(4)
            row2 = st.columns(4)
            all_cols = row1 + row2[:3]
            for i in range(7):
                with all_cols[i]:
                    val = st.number_input(
                        f"Day t−{7 - i}", min_value=-5.0, max_value=35.0,
                        value=defaults[i], step=0.1, key=f"fc_day_{i}",
                    )
                    input_temps.append(val)

            st.markdown("---")
            st.markdown("##### Multi-Step Rolling Forecast")
            n_forecast = st.slider("Forecast horizon (days ahead)", 1, 21, 7, key="fc_horizon")

            # Input Statistics
            st.markdown("---")
            st.markdown("##### 📊 Input Window Statistics")
            inp_arr = np.array(input_temps)
            st.markdown(f"""
            <div class="glass-card">
                <div class="stat-row"><span class="stat-label">Mean</span><span class="stat-value">{inp_arr.mean():.2f}°C</span></div>
                <div class="stat-row"><span class="stat-label">Std Dev</span><span class="stat-value">{inp_arr.std():.2f}°C</span></div>
                <div class="stat-row"><span class="stat-label">Range</span><span class="stat-value">{inp_arr.min():.1f} – {inp_arr.max():.1f}°C</span></div>
                <div class="stat-row"><span class="stat-label">Trend</span><span class="stat-value">{"↗️ Rising" if inp_arr[-1] > inp_arr[0] else "↘️ Falling" if inp_arr[-1] < inp_arr[0] else "➡️ Flat"}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col_output:
            # Single-day prediction
            pred_temp = predict_single(model, scaler, input_temps, USE_LOG, LOG_OFFSET, WINDOW_SIZE)

            st.markdown(f"""
            <div class="prediction-card">
                <div class="pred-label">Next-Day Forecast (t+1)</div>
                <div class="pred-value">{pred_temp:.1f}°C</div>
                <div class="pred-sub">Predicted minimum temperature for tomorrow · Δ from input mean: {pred_temp - np.mean(input_temps):+.2f}°C</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence assessment
            input_std = np.std(input_temps)
            if input_std < 1.5:
                conf_label, conf_color = "HIGH CONFIDENCE", "#34d399"
            elif input_std < 3.0:
                conf_label, conf_color = "MODERATE CONFIDENCE", "#fbbf24"
            else:
                conf_label, conf_color = "LOW CONFIDENCE", "#fb7185"
            st.markdown(f'<div style="text-align:center; margin-bottom:1rem;"><span style="color:{conf_color}; font-weight:600; font-size:0.85rem;">🎯 {conf_label} — Input σ = {input_std:.2f}°C</span></div>', unsafe_allow_html=True)

            # Input trend chart
            fig_trend = go.Figure()
            days_labels = [f"t−{7 - i}" for i in range(7)] + ["t+1"]
            all_vals = list(input_temps) + [round(pred_temp, 2)]

            fig_trend.add_trace(go.Bar(
                x=days_labels[:7], y=all_vals[:7],
                marker=dict(color='#334155', line=dict(color='#475569', width=1)),
                name='Input Window', text=[f"{v:.1f}°" for v in all_vals[:7]], textposition='auto',
            ))
            fig_trend.add_trace(go.Bar(
                x=[days_labels[7]], y=[all_vals[7]],
                marker=dict(color='#0ea5e9', line=dict(color='#38bdf8', width=1.5)),
                name='AI Prediction', text=[f"{all_vals[7]:.1f}°"], textposition='auto',
            ))
            fig_trend.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=True,
                                     legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'),
                                     yaxis_title='Temperature (°C)')
            st.plotly_chart(fig_trend, use_container_width=True)

            # Multi-step forecast
            st.markdown(f'<div class="section-header">📈 {n_forecast}-Day Rolling Forecast</div>', unsafe_allow_html=True)
            multi_preds = multi_step_forecast(model, scaler, input_temps, n_forecast, USE_LOG, LOG_OFFSET, WINDOW_SIZE)

            fig_multi = go.Figure()
            # Input context
            fig_multi.add_trace(go.Scatter(
                x=[f"t−{7 - i}" for i in range(7)], y=input_temps,
                mode='lines+markers', name='Input (observed)',
                line=dict(color='#64748b', width=2),
                marker=dict(size=7, color='#475569', line=dict(color='#94a3b8', width=1)),
            ))
            # Forecast
            fc_labels = [f"t+{i + 1}" for i in range(n_forecast)]
            fig_multi.add_trace(go.Scatter(
                x=fc_labels, y=multi_preds,
                mode='lines+markers', name='AI Forecast',
                line=dict(color='#38bdf8', width=2.5, dash='dot'),
                marker=dict(size=8, color='#0ea5e9', line=dict(color='#38bdf8', width=1.5)),
                text=[f"{v:.1f}°C" for v in multi_preds],
            ))
            # Uncertainty band (heuristic: widens with horizon)
            upper = [multi_preds[i] + (i + 1) * 0.3 * input_std for i in range(n_forecast)]
            lower = [multi_preds[i] - (i + 1) * 0.3 * input_std for i in range(n_forecast)]
            fig_multi.add_trace(go.Scatter(
                x=fc_labels, y=upper, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip',
            ))
            fig_multi.add_trace(go.Scatter(
                x=fc_labels, y=lower, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(56,189,248,0.08)',
                name='Uncertainty Band', hoverinfo='skip',
            ))
            # Connection line
            fig_multi.add_trace(go.Scatter(
                x=["t−1", "t+1"], y=[input_temps[-1], multi_preds[0]],
                mode='lines', line=dict(color='#38bdf8', width=1.5, dash='dash'),
                showlegend=False, hoverinfo='skip',
            ))
            fig_multi.update_layout(**PLOTLY_LAYOUT, height=340,
                                     yaxis_title='Temperature (°C)',
                                     legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'))
            st.plotly_chart(fig_multi, use_container_width=True)

            # Forecast summary table
            fc_summary = pd.DataFrame({
                'Day': fc_labels,
                'Predicted °C': [round(v, 2) for v in multi_preds],
                'Δ from Input Mean': [round(v - np.mean(input_temps), 2) for v in multi_preds],
                'Cumulative Δ': [round(v - multi_preds[0], 2) for v in multi_preds],
            })
            st.dataframe(fc_summary, use_container_width=True, hide_index=True)

            # Forecast insights
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">🔍 Forecast Insights</div>
                • The model predicts a <strong>{"warming" if multi_preds[-1] > multi_preds[0] else "cooling"} trend</strong> over the next {n_forecast} days.<br>
                • Forecasted range: <strong>{min(multi_preds):.1f}°C – {max(multi_preds):.1f}°C</strong><br>
                • Average forecast: <strong>{np.mean(multi_preds):.1f}°C</strong> vs input average: <strong>{np.mean(input_temps):.1f}°C</strong><br>
                • Maximum day-to-day change: <strong>{max(abs(np.diff(multi_preds))):.2f}°C</strong>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
#  TAB 3 — CLIMATE DEEP DIVE (EDA)
# ──────────────────────────────────────────────────────────────────────

with tab_analytics:

    # 1. Full Timeline
    with st.expander("📈 Decadal Temperature Timeline with Rolling Averages", expanded=True):
        ma7 = df['Temperature'].rolling(7, center=True).mean()
        ma30 = df['Temperature'].rolling(30, center=True).mean()
        ma90 = df['Temperature'].rolling(90, center=True).mean()

        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(x=df.index, y=df['Temperature'], name='Daily',
                                     mode='lines', line=dict(color='rgba(148,163,184,0.2)', width=0.7)))
        fig_tl.add_trace(go.Scatter(x=df.index, y=ma7, name='7-Day MA',
                                     mode='lines', line=dict(color='#ff7043', width=1.2)))
        fig_tl.add_trace(go.Scatter(x=df.index, y=ma30, name='30-Day MA',
                                     mode='lines', line=dict(color='#38bdf8', width=1.8)))
        fig_tl.add_trace(go.Scatter(x=df.index, y=ma90, name='90-Day MA',
                                     mode='lines', line=dict(color='#a78bfa', width=2.2)))
        fig_tl.update_layout(**PLOTLY_LAYOUT, height=420, hovermode='x unified',
                              legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'),
                              yaxis_title='Temperature (°C)')
        st.plotly_chart(fig_tl, use_container_width=True)

    # 2. Monthly Analysis Row
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        with st.expander("📊 Monthly Averages by Year (Grouped)", expanded=True):
            monthly_agg = df_ext.groupby(['Year', 'Month'])['Temperature'].mean().reset_index()
            monthly_agg['MonthName'] = monthly_agg['Month'].map(MONTH_MAP)
            fig_mbar = px.bar(monthly_agg, x='MonthName', y='Temperature', color='Year',
                               barmode='group',
                               labels={'Temperature': 'Avg Temp (°C)', 'MonthName': 'Month'},
                               color_continuous_scale='Turbo',
                               category_orders={'MonthName': list(MONTH_MAP.values())})
            fig_mbar.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig_mbar, use_container_width=True)

    with col_m2:
        with st.expander("📊 Overall Monthly Climatology (10-Year Mean)", expanded=True):
            monthly_vals = monthly_means_global.values
            monthly_labels = [MONTH_MAP[m] for m in monthly_means_global.index]
            fig_clim = go.Figure(go.Bar(
                x=monthly_labels, y=monthly_vals,
                marker=dict(
                    color=monthly_vals,
                    colorscale=[[0, '#38bdf8'], [0.5, '#fbbf24'], [1, '#ef4444']],
                    line=dict(color='rgba(255,255,255,0.1)', width=1),
                ),
                text=[f"{v:.1f}°" for v in monthly_vals], textposition='outside',
            ))
            fig_clim.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False,
                                    yaxis_title='Avg Minimum Temp (°C)')
            st.plotly_chart(fig_clim, use_container_width=True)

    # 3. Seasonal Profile + Box Plots
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        with st.expander("🌿 Seasonal Day-of-Year Profile", expanded=True):
            doy_profile = df_ext.groupby('DayOfYear')['Temperature'].agg(['mean', 'min', 'max']).reset_index()
            fig_doy = go.Figure()
            fig_doy.add_trace(go.Scatter(x=doy_profile['DayOfYear'], y=doy_profile['max'],
                                          mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_doy.add_trace(go.Scatter(x=doy_profile['DayOfYear'], y=doy_profile['min'],
                                          mode='lines', line=dict(width=0), fill='tonexty',
                                          fillcolor='rgba(56,189,248,0.12)', name='Min–Max Range'))
            fig_doy.add_trace(go.Scatter(x=doy_profile['DayOfYear'], y=doy_profile['mean'],
                                          mode='lines', line=dict(color='#38bdf8', width=2), name='Mean'))
            fig_doy.update_layout(**PLOTLY_LAYOUT, height=380,
                                   xaxis_title='Day of Year', yaxis_title='Temperature (°C)',
                                   legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'))
            st.plotly_chart(fig_doy, use_container_width=True)

    with col_s2:
        with st.expander("📦 Yearly Distribution Box Plots", expanded=True):
            fig_box = px.box(df_ext, x='Year', y='Temperature',
                              color_discrete_sequence=['#818cf8'],
                              labels={'Temperature': 'Temp (°C)'})
            fig_box.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                                   xaxis=dict(dtick=1))
            st.plotly_chart(fig_box, use_container_width=True)

    # 4. Distribution + Lag Correlation
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        with st.expander("📉 Temperature Distribution & Percentiles", expanded=True):
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df['Temperature'], nbinsx=60,
                marker=dict(color='rgba(56,189,248,0.5)', line=dict(color='rgba(56,189,248,0.8)', width=0.5)),
                name='Distribution',
            ))
            mean_t = df['Temperature'].mean()
            median_t = df['Temperature'].median()
            fig_hist.add_vline(x=mean_t, line=dict(color='#ef4444', dash='dash', width=1.5),
                                annotation_text=f"Mean: {mean_t:.1f}°C", annotation_position="top right")
            fig_hist.add_vline(x=median_t, line=dict(color='#fbbf24', dash='dot', width=1.5),
                                annotation_text=f"Median: {median_t:.1f}°C", annotation_position="top left")
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=380,
                                    xaxis_title='Temperature (°C)', yaxis_title='Frequency',
                                    showlegend=False)
            skew_val = df['Temperature'].skew()
            kurt_val = df['Temperature'].kurtosis()
            fig_hist.add_annotation(x=0.98, y=0.95, xref='paper', yref='paper',
                                     text=f"Skewness: {skew_val:.3f}<br>Kurtosis: {kurt_val:.3f}",
                                     showarrow=False, font=dict(size=11, color='#94a3b8'),
                                     align='right', bgcolor='rgba(15,23,42,0.8)',
                                     bordercolor='rgba(255,255,255,0.1)', borderwidth=1, borderpad=6)
            st.plotly_chart(fig_hist, use_container_width=True)

    with col_d2:
        with st.expander("🔗 Lag-1 & Lag-7 Autocorrelation Scatter", expanded=True):
            lag_choice = st.radio("Select lag:", [1, 7], horizontal=True, key="lag_radio")
            temp_vals = df['Temperature'].values
            fig_lag = go.Figure(go.Scatter(
                x=temp_vals[:-lag_choice], y=temp_vals[lag_choice:],
                mode='markers',
                marker=dict(color='#818cf8' if lag_choice == 1 else '#a78bfa', size=3, opacity=0.3),
                name=f'Lag-{lag_choice}',
            ))
            rng = [float(df['Temperature'].min() - 1), float(df['Temperature'].max() + 1)]
            fig_lag.add_trace(go.Scatter(x=rng, y=rng, mode='lines',
                                          line=dict(color='#ef4444', dash='dash', width=1.5), name='y = x'))
            corr_val = np.corrcoef(temp_vals[:-lag_choice], temp_vals[lag_choice:])[0, 1]
            fig_lag.update_layout(**PLOTLY_LAYOUT, height=380,
                                   xaxis_title=f'Temperature (t−{lag_choice}) °C',
                                   yaxis_title='Temperature (t) °C',
                                   title=dict(text=f'Lag-{lag_choice} Scatter · r = {corr_val:.4f}',
                                              font=dict(size=13, color='#94a3b8')))
            st.plotly_chart(fig_lag, use_container_width=True)

    # 5. ACF & PACF Plot
    col_acf, col_pacf = st.columns(2)

    with col_acf:
        with st.expander("📐 Autocorrelation Function (ACF)", expanded=True):
            acf_vals = compute_acf(df['Temperature'], nlags=40)
            n_obs = len(df['Temperature'].dropna())
            ci = 1.96 / np.sqrt(n_obs)

            fig_acf = go.Figure()
            for i, val in enumerate(acf_vals):
                fig_acf.add_trace(go.Scatter(
                    x=[i, i], y=[0, val], mode='lines',
                    line=dict(color='#38bdf8', width=2), showlegend=False, hoverinfo='skip',
                ))
            fig_acf.add_trace(go.Scatter(
                x=list(range(len(acf_vals))), y=acf_vals,
                mode='markers', marker=dict(color='#0ea5e9', size=6), name='ACF',
            ))
            fig_acf.add_hline(y=ci, line=dict(color='#ef4444', dash='dash', width=1), annotation_text='95% CI')
            fig_acf.add_hline(y=-ci, line=dict(color='#ef4444', dash='dash', width=1))
            fig_acf.add_hline(y=0, line=dict(color='#475569', width=0.5))
            fig_acf.update_layout(**PLOTLY_LAYOUT, height=320,
                                   xaxis_title='Lag', yaxis_title='Autocorrelation')
            st.plotly_chart(fig_acf, use_container_width=True)

    with col_pacf:
        with st.expander("📐 Partial Autocorrelation (PACF)", expanded=True):
            pacf_vals = compute_pacf(df['Temperature'], nlags=40)

            fig_pacf = go.Figure()
            for i, val in enumerate(pacf_vals):
                fig_pacf.add_trace(go.Scatter(
                    x=[i, i], y=[0, val], mode='lines',
                    line=dict(color='#a78bfa', width=2), showlegend=False, hoverinfo='skip',
                ))
            fig_pacf.add_trace(go.Scatter(
                x=list(range(len(pacf_vals))), y=pacf_vals,
                mode='markers', marker=dict(color='#818cf8', size=6), name='PACF',
            ))
            fig_pacf.add_hline(y=ci, line=dict(color='#ef4444', dash='dash', width=1), annotation_text='95% CI')
            fig_pacf.add_hline(y=-ci, line=dict(color='#ef4444', dash='dash', width=1))
            fig_pacf.add_hline(y=0, line=dict(color='#475569', width=0.5))
            fig_pacf.update_layout(**PLOTLY_LAYOUT, height=320,
                                    xaxis_title='Lag', yaxis_title='Partial Autocorrelation')
            st.plotly_chart(fig_pacf, use_container_width=True)

    # 6. Rolling Volatility
    with st.expander("📈 Rolling Temperature Volatility", expanded=False):
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=df_ext.index, y=df_ext['RollingStd7'],
            mode='lines', name='7-Day σ', line=dict(color='#38bdf8', width=1.5),
        ))
        fig_vol.add_trace(go.Scatter(
            x=df_ext.index, y=df_ext['RollingStd30'],
            mode='lines', name='30-Day σ', line=dict(color='#a78bfa', width=2),
        ))
        fig_vol.update_layout(**PLOTLY_LAYOUT, height=320,
                               yaxis_title='Std Dev (°C)', hovermode='x unified',
                               legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'))
        st.plotly_chart(fig_vol, use_container_width=True)

    # 7. Quarterly Violin Plots
    with st.expander("🎻 Quarterly Temperature Distributions", expanded=False):
        quarter_names = {1: 'Q1 (Jan-Mar)', 2: 'Q2 (Apr-Jun)', 3: 'Q3 (Jul-Sep)', 4: 'Q4 (Oct-Dec)'}
        df_ext['QuarterName'] = df_ext['Quarter'].map(quarter_names)
        fig_violin = px.violin(df_ext, x='QuarterName', y='Temperature', box=True, points='outliers',
                                color_discrete_sequence=['#818cf8'],
                                labels={'Temperature': 'Temp (°C)', 'QuarterName': 'Quarter'},
                                category_orders={'QuarterName': list(quarter_names.values())})
        fig_violin.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
#  TAB 4 — DECOMPOSITION & STATIONARITY
# ──────────────────────────────────────────────────────────────────────

with tab_decomp:

    st.markdown('<div class="section-header">🔬 Seasonal Decomposition (Additive)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">📖 What is Seasonal Decomposition?</div>
        Time series decomposition separates a signal into three components:
        <strong>Trend</strong> (long-term direction), <strong>Seasonal</strong> (repeating patterns),
        and <strong>Residual</strong> (random noise). This helps understand the underlying structure
        of Melbourne's temperature patterns.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Computing seasonal decomposition (365-day period)..."):
        trend, seasonal, residual = seasonal_decompose_manual(df['Temperature'], period=365)

    # Plot decomposition
    fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'],
                                vertical_spacing=0.06)

    fig_decomp.add_trace(go.Scatter(x=df.index, y=df['Temperature'], name='Observed',
                                     line=dict(color='#38bdf8', width=0.8)), row=1, col=1)
    fig_decomp.add_trace(go.Scatter(x=df.index, y=trend, name='Trend',
                                     line=dict(color='#f97316', width=2)), row=2, col=1)
    fig_decomp.add_trace(go.Scatter(x=df.index, y=seasonal, name='Seasonal',
                                     line=dict(color='#34d399', width=0.8)), row=3, col=1)
    fig_decomp.add_trace(go.Scatter(x=df.index, y=residual, name='Residual',
                                     line=dict(color='#a78bfa', width=0.5)), row=4, col=1)

    fig_decomp.update_layout(**PLOTLY_LAYOUT, height=700, showlegend=False)
    fig_decomp.update_annotations(font=dict(size=12, color='#94a3b8'))
    for i in range(1, 5):
        fig_decomp.update_yaxes(title_text='°C', row=i, col=1)
    st.plotly_chart(fig_decomp, use_container_width=True)

    # Decomposition stats
    st.markdown("---")
    col_ds1, col_ds2, col_ds3 = st.columns(3)
    with col_ds1:
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-family:'Outfit'; font-weight:600; color:#f97316; margin-bottom:0.5rem;">📈 Trend Component</div>
            <div class="stat-row"><span class="stat-label">Range</span><span class="stat-value">{np.nanmin(trend):.1f} – {np.nanmax(trend):.1f}°C</span></div>
            <div class="stat-row"><span class="stat-label">Mean</span><span class="stat-value">{np.nanmean(trend):.2f}°C</span></div>
            <div class="stat-row"><span class="stat-label">% Variance</span><span class="stat-value">{np.nanvar(trend)/np.nanvar(df['Temperature'].values)*100:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_ds2:
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-family:'Outfit'; font-weight:600; color:#34d399; margin-bottom:0.5rem;">🔄 Seasonal Component</div>
            <div class="stat-row"><span class="stat-label">Amplitude</span><span class="stat-value">{np.nanmax(seasonal) - np.nanmin(seasonal):.2f}°C</span></div>
            <div class="stat-row"><span class="stat-label">Mean</span><span class="stat-value">{np.nanmean(seasonal):.4f}°C</span></div>
            <div class="stat-row"><span class="stat-label">Period</span><span class="stat-value">365 days</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_ds3:
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-family:'Outfit'; font-weight:600; color:#a78bfa; margin-bottom:0.5rem;">🎲 Residual Component</div>
            <div class="stat-row"><span class="stat-label">Std Dev</span><span class="stat-value">{np.nanstd(residual):.2f}°C</span></div>
            <div class="stat-row"><span class="stat-label">Mean</span><span class="stat-value">{np.nanmean(residual):.4f}°C</span></div>
            <div class="stat-row"><span class="stat-label">% Variance</span><span class="stat-value">{np.nanvar(residual)/np.nanvar(df['Temperature'].values)*100:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Residual distribution
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Residual Analysis</div>', unsafe_allow_html=True)
    col_rd1, col_rd2 = st.columns(2)
    with col_rd1:
        valid_resid = residual[~np.isnan(residual)]
        fig_resid_hist = go.Figure(go.Histogram(
            x=valid_resid, nbinsx=60,
            marker=dict(color='rgba(167,139,250,0.5)', line=dict(color='rgba(167,139,250,0.8)', width=0.5)),
        ))
        fig_resid_hist.add_vline(x=0, line=dict(color='#ef4444', dash='dash', width=1.5))
        fig_resid_hist.update_layout(**PLOTLY_LAYOUT, height=320,
                                      xaxis_title='Residual (°C)', yaxis_title='Frequency',
                                      title=dict(text='Residual Distribution', font=dict(size=13, color='#94a3b8')))
        st.plotly_chart(fig_resid_hist, use_container_width=True)

    with col_rd2:
        resid_acf = compute_acf(pd.Series(valid_resid), nlags=30)
        fig_resid_acf = go.Figure()
        for i, val in enumerate(resid_acf):
            fig_resid_acf.add_trace(go.Scatter(
                x=[i, i], y=[0, val], mode='lines',
                line=dict(color='#a78bfa', width=2), showlegend=False, hoverinfo='skip',
            ))
        fig_resid_acf.add_trace(go.Scatter(
            x=list(range(len(resid_acf))), y=resid_acf,
            mode='markers', marker=dict(color='#818cf8', size=6), name='ACF',
        ))
        ci_resid = 1.96 / np.sqrt(len(valid_resid))
        fig_resid_acf.add_hline(y=ci_resid, line=dict(color='#ef4444', dash='dash', width=1))
        fig_resid_acf.add_hline(y=-ci_resid, line=dict(color='#ef4444', dash='dash', width=1))
        fig_resid_acf.add_hline(y=0, line=dict(color='#475569', width=0.5))
        fig_resid_acf.update_layout(**PLOTLY_LAYOUT, height=320,
                                     xaxis_title='Lag', yaxis_title='ACF',
                                     title=dict(text='Residual Autocorrelation', font=dict(size=13, color='#94a3b8')))
        st.plotly_chart(fig_resid_acf, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
#  TAB 5 — EXTREME EVENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────

with tab_extreme:

    st.markdown('<div class="section-header">⚡ Extreme Temperature Event Analysis</div>', unsafe_allow_html=True)

    # Define thresholds
    p5 = df['Temperature'].quantile(0.05)
    p95 = df['Temperature'].quantile(0.95)
    p1 = df['Temperature'].quantile(0.01)
    p99 = df['Temperature'].quantile(0.99)

    # KPIs
    ek1, ek2, ek3, ek4 = st.columns(4)
    cold_extreme = (df['Temperature'] < p5).sum()
    hot_extreme = (df['Temperature'] > p95).sum()
    freeze_days = (df['Temperature'] < 0).sum()
    warm_days = (df['Temperature'] > 20).sum()

    for col, icon, label, value, sub, cls in [
        (ek1, "🥶", "Extreme Cold", str(cold_extreme), f"below {p5:.1f}°C (P5)", "kpi-cyan"),
        (ek2, "🔥", "Extreme Warm", str(hot_extreme), f"above {p95:.1f}°C (P95)", "kpi-rose"),
        (ek3, "❄️", "Sub-Zero Days", str(freeze_days), "below 0°C", "kpi-blue"),
        (ek4, "☀️", "Warm Days (>20°C)", str(warm_days), "above 20°C", "kpi-amber"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Extreme events timeline
    with st.expander("📈 Extreme Events Timeline", expanded=True):
        fig_ext = go.Figure()
        fig_ext.add_trace(go.Scatter(
            x=df.index, y=df['Temperature'], mode='lines',
            line=dict(color='rgba(148,163,184,0.3)', width=0.7), name='Daily Temp',
        ))
        # Mark extreme cold
        cold_mask = df['Temperature'] < p5
        fig_ext.add_trace(go.Scatter(
            x=df.index[cold_mask], y=df['Temperature'][cold_mask],
            mode='markers', marker=dict(color='#38bdf8', size=4, symbol='circle'),
            name=f'Extreme Cold (<{p5:.1f}°C)',
        ))
        # Mark extreme warm
        hot_mask = df['Temperature'] > p95
        fig_ext.add_trace(go.Scatter(
            x=df.index[hot_mask], y=df['Temperature'][hot_mask],
            mode='markers', marker=dict(color='#ef4444', size=4, symbol='circle'),
            name=f'Extreme Warm (>{p95:.1f}°C)',
        ))
        fig_ext.add_hline(y=p5, line=dict(color='#38bdf8', dash='dash', width=1), annotation_text='P5')
        fig_ext.add_hline(y=p95, line=dict(color='#ef4444', dash='dash', width=1), annotation_text='P95')
        fig_ext.update_layout(**PLOTLY_LAYOUT, height=380,
                               yaxis_title='Temperature (°C)',
                               legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'))
        st.plotly_chart(fig_ext, use_container_width=True)

    # Monthly extreme events distribution
    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        with st.expander("📊 Monthly Cold Extremes Count", expanded=True):
            cold_monthly = df_ext[df_ext['Temperature'] < p5].groupby('MonthName').size()
            all_months = list(MONTH_MAP.values())
            cold_counts = [int(cold_monthly.get(m, 0)) for m in all_months]
            fig_cold_bar = go.Figure(go.Bar(
                x=all_months, y=cold_counts,
                marker=dict(color=cold_counts, colorscale=[[0, '#1e3a5f'], [1, '#38bdf8']],
                            line=dict(color='rgba(255,255,255,0.1)', width=1)),
                text=cold_counts, textposition='outside',
            ))
            fig_cold_bar.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False,
                                        yaxis_title='Count of Extreme Cold Days')
            st.plotly_chart(fig_cold_bar, use_container_width=True)

    with col_ex2:
        with st.expander("📊 Monthly Warm Extremes Count", expanded=True):
            hot_monthly = df_ext[df_ext['Temperature'] > p95].groupby('MonthName').size()
            hot_counts = [int(hot_monthly.get(m, 0)) for m in all_months]
            fig_hot_bar = go.Figure(go.Bar(
                x=all_months, y=hot_counts,
                marker=dict(color=hot_counts, colorscale=[[0, '#5f1e1e'], [1, '#ef4444']],
                            line=dict(color='rgba(255,255,255,0.1)', width=1)),
                text=hot_counts, textposition='outside',
            ))
            fig_hot_bar.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False,
                                       yaxis_title='Count of Extreme Warm Days')
            st.plotly_chart(fig_hot_bar, use_container_width=True)

    # Extreme day-to-day changes
    with st.expander("📈 Largest Day-to-Day Temperature Swings", expanded=True):
        temp_change = df_ext['TempChange'].dropna().copy()
        top_increases = temp_change.nlargest(10)
        top_decreases = temp_change.nsmallest(10)

        col_inc, col_dec = st.columns(2)
        with col_inc:
            st.markdown("##### 🔥 Top 10 Warmest Jumps")
            inc_df = pd.DataFrame({
                'Date': top_increases.index.strftime('%Y-%m-%d'),
                'Change (°C)': [f"+{v:.1f}" for v in top_increases.values],
                'From': [f"{df['Temperature'].loc[d - timedelta(days=1)]:.1f}°C" if d - timedelta(days=1) in df.index else "N/A" for d in top_increases.index],
                'To': [f"{df['Temperature'].loc[d]:.1f}°C" for d in top_increases.index],
            })
            st.dataframe(inc_df, use_container_width=True, hide_index=True)

        with col_dec:
            st.markdown("##### ❄️ Top 10 Coldest Drops")
            dec_df = pd.DataFrame({
                'Date': top_decreases.index.strftime('%Y-%m-%d'),
                'Change (°C)': [f"{v:.1f}" for v in top_decreases.values],
                'From': [f"{df['Temperature'].loc[d - timedelta(days=1)]:.1f}°C" if d - timedelta(days=1) in df.index else "N/A" for d in top_decreases.index],
                'To': [f"{df['Temperature'].loc[d]:.1f}°C" for d in top_decreases.index],
            })
            st.dataframe(dec_df, use_container_width=True, hide_index=True)

    # Consecutive extreme days analysis
    with st.expander("🔗 Consecutive Cold Streak Analysis", expanded=False):
        cold_streak = (df['Temperature'] < p5).astype(int)
        streaks = []
        current_streak = 0
        streak_start = None
        for i, (date, val) in enumerate(cold_streak.items()):
            if val == 1:
                if current_streak == 0:
                    streak_start = date
                current_streak += 1
            else:
                if current_streak > 1:
                    streaks.append({'Start': streak_start, 'End': date - timedelta(days=1),
                                    'Duration (days)': current_streak,
                                    'Min Temp': f"{df['Temperature'].loc[streak_start:date - timedelta(days=1)].min():.1f}°C"})
                current_streak = 0
        if current_streak > 1:
            streaks.append({'Start': streak_start, 'End': cold_streak.index[-1],
                            'Duration (days)': current_streak,
                            'Min Temp': f"{df['Temperature'].loc[streak_start:].min():.1f}°C"})

        if streaks:
            streaks_df = pd.DataFrame(streaks).sort_values('Duration (days)', ascending=False).head(15)
            streaks_df['Start'] = streaks_df['Start'].dt.strftime('%Y-%m-%d')
            streaks_df['End'] = streaks_df['End'].dt.strftime('%Y-%m-%d')
            st.dataframe(streaks_df, use_container_width=True, hide_index=True)
        else:
            st.info("No consecutive extreme cold streaks found.")


# ──────────────────────────────────────────────────────────────────────
#  TAB 6 — MODEL INTELLIGENCE CENTER
# ──────────────────────────────────────────────────────────────────────

with tab_brain:

    if not champion_info:
        st.info("💡 Run the Colab training pipeline (18 cells) to populate model intelligence. Upload `champion_info.pkl` to this repository.")
    else:
        # ── KPI Cards ──
        c1, c2, c3, c4 = st.columns(4)
        deploy_name = champion_info.get('deploy_model_name', 'N/A')
        deploy_rmse = champion_info.get('deploy_rmse', 0)
        deploy_mae = champion_info.get('deploy_mae', 0)
        deploy_r2 = champion_info.get('deploy_r2', 0)

        with c1:
            st.markdown(f"""
            <div class="kpi-card kpi-cyan">
                <div class="kpi-icon">🏆</div>
                <div class="kpi-label">Champion</div>
                <div class="kpi-value">{deploy_name}</div>
                <div class="kpi-sub">Overall: {champion_info.get('overall_champion', '')}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card kpi-rose">
                <div class="kpi-icon">📉</div>
                <div class="kpi-label">RMSE</div>
                <div class="kpi-value">{deploy_rmse:.3f}</div>
                <div class="kpi-sub">Root Mean Squared Error</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card kpi-amber">
                <div class="kpi-icon">📏</div>
                <div class="kpi-label">MAE</div>
                <div class="kpi-value">{deploy_mae:.3f}</div>
                <div class="kpi-sub">Mean Absolute Error</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-card kpi-emerald">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-label">R² Score</div>
                <div class="kpi-value">{deploy_r2:.3f}</div>
                <div class="kpi-sub">Variance Explained</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metrics Table + Config ──
        col_tbl, col_cfg = st.columns([3, 2])

        with col_tbl:
            st.markdown('<div class="section-header">📋 All Models — Validation Metrics</div>', unsafe_allow_html=True)
            if champion_info.get('metrics'):
                metrics_df = pd.DataFrame(champion_info['metrics'])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        with col_cfg:
            st.markdown('<div class="section-header">⚙️ Training Configuration</div>', unsafe_allow_html=True)
            sarima_order = champion_info.get('best_sarima_order', (0, 0, 0))
            sarima_seasonal = champion_info.get('best_sarima_seasonal', (0, 0, 0, 7))
            st.markdown(f"""
            | Parameter | Value |
            |:---|:---|
            | **Deploy Model** | `{deploy_name}` |
            | **Lookback Window** | `{champion_info.get('window_size', 7)} days` |
            | **SARIMA Order** | `{sarima_order}` |
            | **SARIMA Seasonal** | `{sarima_seasonal}` |
            | **Log Transform** | `{'Applied' if champion_info.get('use_log_transform') else 'Not Required'}` |
            | **Log Offset** | `{champion_info.get('log_offset', 0.0):.2f}` |
            | **Train/Test Split** | `{champion_info.get('split_index', 'N/A')}` idx |
            """)

        st.markdown("---")

        # ── Comparative Bar Chart ──
        if champion_info.get('metrics'):
            st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

            metrics_df = pd.DataFrame(champion_info['metrics'])
            model_names = metrics_df['Model'].tolist()
            colors = ['#38bdf8', '#818cf8', '#a78bfa', '#fb7185', '#34d399']
            bar_colors = (colors * 3)[:len(model_names)]

            r2_col = 'R²' if 'R²' in metrics_df.columns else 'R2'
            has_r2 = r2_col in metrics_df.columns
            n_cols = 3 if has_r2 else 2
            subtitles = ['RMSE (lower = better)', 'MAE (lower = better)']
            if has_r2:
                subtitles.append('R² (higher = better)')

            fig_comp = make_subplots(rows=1, cols=n_cols, subplot_titles=subtitles)

            rmse_vals = [float(v) for v in metrics_df['RMSE'].tolist()]
            mae_vals = [float(v) for v in metrics_df['MAE'].tolist()]

            fig_comp.add_trace(go.Bar(x=model_names, y=rmse_vals, marker_color=bar_colors,
                                       text=[f"{v:.3f}" for v in rmse_vals], textposition='auto',
                                       showlegend=False), row=1, col=1)
            fig_comp.add_trace(go.Bar(x=model_names, y=mae_vals, marker_color=bar_colors,
                                       text=[f"{v:.3f}" for v in mae_vals], textposition='auto',
                                       showlegend=False), row=1, col=2)

            if has_r2:
                r2_vals = [float(v) for v in metrics_df[r2_col].tolist()]
                fig_comp.add_trace(go.Bar(x=model_names, y=r2_vals, marker_color=bar_colors,
                                           text=[f"{v:.3f}" for v in r2_vals], textposition='auto',
                                           showlegend=False), row=1, col=3)

            fig_comp.update_layout(**PLOTLY_LAYOUT, height=360)
            fig_comp.update_annotations(font=dict(size=12, color='#94a3b8'))
            st.plotly_chart(fig_comp, use_container_width=True)

            # Radar chart for model comparison
            if has_r2 and len(model_names) >= 2:
                st.markdown('<div class="section-header">🕸️ Model Comparison Radar</div>', unsafe_allow_html=True)
                fig_radar = go.Figure()
                categories = ['RMSE', 'MAE', 'R²']
                for idx, name in enumerate(model_names):
                    # Normalize: invert RMSE and MAE so higher is better
                    max_rmse = max(rmse_vals) if max(rmse_vals) > 0 else 1
                    max_mae = max(mae_vals) if max(mae_vals) > 0 else 1
                    norm_rmse = 1 - (rmse_vals[idx] / max_rmse)
                    norm_mae = 1 - (mae_vals[idx] / max_mae)
                    norm_r2 = max(0, r2_vals[idx])

                    fig_radar.add_trace(go.Scatterpolar(
                        r=[norm_rmse, norm_mae, norm_r2, norm_rmse],
                        theta=categories + [categories[0]],
                        name=name,
                        line=dict(color=bar_colors[idx], width=2),
                        fill='toself',
                        fillcolor=bar_colors[idx].replace(')', ',0.1)').replace('rgb', 'rgba') if 'rgb' in bar_colors[idx] else f"rgba(56,189,248,0.1)",
                    ))
                fig_radar.update_layout(**PLOTLY_LAYOUT, height=400,
                                         polar=dict(bgcolor='rgba(0,0,0,0)',
                                                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor='rgba(255,255,255,0.05)'),
                                                    angularaxis=dict(gridcolor='rgba(255,255,255,0.05)')),
                                         legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'))
                st.plotly_chart(fig_radar, use_container_width=True)

        # ── Live Actual vs Predicted ──
        if model is not None and scaler is not None:
            st.markdown("---")
            st.markdown('<div class="section-header">🎯 Live Test-Set: Actual vs Predicted</div>', unsafe_allow_html=True)

            with st.spinner("Computing test predictions from deployed model..."):
                dates, actual, pred = compute_test_predictions(model, scaler, df, champion_info)

            if dates is not None and len(dates) > 0:
                fig_avp = go.Figure()
                fig_avp.add_trace(go.Scatter(
                    x=dates, y=actual, name='Actual (Ground Truth)',
                    mode='lines', line=dict(color='#f8fafc', width=1.2),
                ))
                fig_avp.add_trace(go.Scatter(
                    x=dates, y=pred, name=f'{deploy_name} Prediction',
                    mode='lines', line=dict(color='#38bdf8', width=1.2, dash='dot'),
                ))
                fig_avp.update_layout(**PLOTLY_LAYOUT, height=400, hovermode='x unified',
                                       yaxis_title='Temperature (°C)',
                                       legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'))
                st.plotly_chart(fig_avp, use_container_width=True)

                # Error analysis
                errors = actual - pred
                col_err1, col_err2 = st.columns(2)
                with col_err1:
                    st.markdown('<div class="section-header">📊 Prediction Error Distribution</div>', unsafe_allow_html=True)
                    fig_err = go.Figure(go.Histogram(
                        x=errors, nbinsx=50,
                        marker=dict(color='rgba(129,140,248,0.5)', line=dict(color='rgba(129,140,248,0.8)', width=0.5)),
                    ))
                    fig_err.add_vline(x=0, line=dict(color='#ef4444', dash='dash', width=1.5))
                    fig_err.add_annotation(x=0.95, y=0.92, xref='paper', yref='paper',
                                            text=f"Mean Error: {np.mean(errors):.3f}°C<br>Std Error: {np.std(errors):.3f}°C",
                                            showarrow=False, font=dict(size=11, color='#94a3b8'),
                                            bgcolor='rgba(15,23,42,0.8)', bordercolor='rgba(255,255,255,0.1)',
                                            borderwidth=1, borderpad=6, align='right')
                    fig_err.update_layout(**PLOTLY_LAYOUT, height=320,
                                           xaxis_title='Prediction Error (°C)', yaxis_title='Frequency')
                    st.plotly_chart(fig_err, use_container_width=True)

                with col_err2:
                    st.markdown('<div class="section-header">🎯 Actual vs Predicted Scatter</div>', unsafe_allow_html=True)
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=actual, y=pred, mode='markers',
                        marker=dict(color='#818cf8', size=4, opacity=0.4), name='Predictions',
                    ))
                    perfect_range = [float(min(actual.min(), pred.min()) - 1), float(max(actual.max(), pred.max()) + 1)]
                    fig_scatter.add_trace(go.Scatter(
                        x=perfect_range, y=perfect_range, mode='lines',
                        line=dict(color='#ef4444', dash='dash', width=1.5), name='Perfect Prediction',
                    ))
                    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=320,
                                               xaxis_title='Actual (°C)', yaxis_title='Predicted (°C)')
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # Error by month
                st.markdown('<div class="section-header">📅 Prediction Error by Month</div>', unsafe_allow_html=True)
                error_df = pd.DataFrame({'Date': dates, 'Error': errors})
                error_df['Month'] = error_df['Date'].dt.month
                error_df['MonthName'] = error_df['Month'].map(MONTH_MAP)
                monthly_error = error_df.groupby('MonthName')['Error'].agg(['mean', 'std']).reindex(list(MONTH_MAP.values()))

                fig_merr = go.Figure()
                fig_merr.add_trace(go.Bar(
                    x=monthly_error.index.tolist(),
                    y=monthly_error['mean'].values.tolist(),
                    marker=dict(
                        color=['#38bdf8' if v < 0 else '#fb7185' for v in monthly_error['mean'].values],
                        line=dict(color='rgba(255,255,255,0.1)', width=1),
                    ),
                    text=[f"{v:.2f}" for v in monthly_error['mean'].values],
                    textposition='outside',
                    error_y=dict(type='data', array=monthly_error['std'].fillna(0).values.tolist(), visible=True,
                                  color='rgba(148,163,184,0.5)'),
                ))
                fig_merr.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                                        yaxis_title='Mean Error (°C)')
                st.plotly_chart(fig_merr, use_container_width=True)

        # ── Training Curve Images (if uploaded) ──
        training_images = [
            ('lstm_training.png', 'LSTM Training Convergence (Loss & MAE vs Epochs)'),
            ('gru_training.png', 'GRU Training Convergence (Loss & MAE vs Epochs)'),
            ('forecast_comparison.png', 'Test-Set Forecast Comparison (SARIMA vs LSTM vs GRU)'),
        ]
        available_images = [(f, c) for f, c in training_images if os.path.exists(f)]
        if available_images:
            st.markdown("---")
            st.markdown('<div class="section-header">📷 Training Artifacts</div>', unsafe_allow_html=True)
            img_cols = st.columns(len(available_images))
            for col, (fname, caption) in zip(img_cols, available_images):
                with col:
                    st.image(fname, caption=caption, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
#  TAB 7 — WHAT-IF SIMULATOR
# ──────────────────────────────────────────────────────────────────────

with tab_whatif:

    st.markdown('<div class="section-header">🧪 What-If Climate Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">🔬 About This Simulator</div>
        Explore hypothetical climate scenarios by adjusting temperature parameters.
        See how different baseline temperatures, volatility levels, and seasonal
        amplitudes affect the temperature distribution and anomaly patterns.
        This tool helps build intuition about Melbourne's climate dynamics.
    </div>
    """, unsafe_allow_html=True)

    col_wi_config, col_wi_output = st.columns([1, 2])

    with col_wi_config:
        st.markdown("##### 🎛️ Scenario Parameters")
        temp_shift = st.slider("Global Temperature Shift (°C)", -5.0, 5.0, 0.0, 0.1, key="wi_shift",
                                help="Simulate uniform warming or cooling")
        vol_mult = st.slider("Volatility Multiplier", 0.3, 3.0, 1.0, 0.1, key="wi_vol",
                              help="Scale temperature variability")
        seasonal_mult = st.slider("Seasonal Amplitude", 0.3, 3.0, 1.0, 0.1, key="wi_season",
                                   help="Amplify or dampen seasonal cycles")

        st.markdown("---")
        # Show current vs modified stats
        modified_temps = (df['Temperature'] - df['Temperature'].mean()) * vol_mult + df['Temperature'].mean() + temp_shift
        # Apply seasonal amplitude
        if seasonal_mult != 1.0:
            mean_temp = modified_temps.mean()
            modified_temps = (modified_temps - mean_temp) * seasonal_mult + mean_temp

        st.markdown("##### 📊 Scenario Statistics")
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-row"><span class="stat-label">Original Mean</span><span class="stat-value">{df['Temperature'].mean():.2f}°C</span></div>
            <div class="stat-row"><span class="stat-label">Modified Mean</span><span class="stat-value">{modified_temps.mean():.2f}°C</span></div>
            <div class="stat-row"><span class="stat-label">Original Std</span><span class="stat-value">{df['Temperature'].std():.2f}°C</span></div>
            <div class="stat-row"><span class="stat-label">Modified Std</span><span class="stat-value">{modified_temps.std():.2f}°C</span></div>
            <div class="stat-row"><span class="stat-label">ΔMean</span><span class="stat-value">{modified_temps.mean() - df['Temperature'].mean():+.2f}°C</span></div>
            <div class="stat-row"><span class="stat-label">Freeze Days</span><span class="stat-value">{(modified_temps < 0).sum()} (was {(df['Temperature'] < 0).sum()})</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_wi_output:
        # Comparison chart
        fig_wi = go.Figure()
        fig_wi.add_trace(go.Scatter(
            x=df.index, y=df['Temperature'].rolling(30, center=True).mean(),
            mode='lines', name='Original (30-Day MA)',
            line=dict(color='#475569', width=1.5),
        ))
        fig_wi.add_trace(go.Scatter(
            x=df.index, y=modified_temps.rolling(30, center=True).mean(),
            mode='lines', name='Modified Scenario (30-Day MA)',
            line=dict(color='#38bdf8', width=2),
        ))
        fig_wi.update_layout(**PLOTLY_LAYOUT, height=380,
                              yaxis_title='Temperature (°C)', hovermode='x unified',
                              legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'))
        st.plotly_chart(fig_wi, use_container_width=True)

        # Distribution comparison
        col_wi_d1, col_wi_d2 = st.columns(2)
        with col_wi_d1:
            fig_wi_hist = go.Figure()
            fig_wi_hist.add_trace(go.Histogram(
                x=df['Temperature'], nbinsx=50, name='Original',
                marker=dict(color='rgba(71,85,105,0.4)', line=dict(color='rgba(71,85,105,0.6)', width=0.5)),
            ))
            fig_wi_hist.add_trace(go.Histogram(
                x=modified_temps, nbinsx=50, name='Modified',
                marker=dict(color='rgba(56,189,248,0.4)', line=dict(color='rgba(56,189,248,0.6)', width=0.5)),
            ))
            fig_wi_hist.update_layout(**PLOTLY_LAYOUT, height=300, barmode='overlay',
                                       xaxis_title='Temperature (°C)', yaxis_title='Count',
                                       legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'))
            st.plotly_chart(fig_wi_hist, use_container_width=True)

        with col_wi_d2:
            # Monthly comparison
            orig_monthly = df_ext.groupby('Month')['Temperature'].mean()
            mod_monthly_vals = pd.Series(modified_temps.values, index=df.index)
            mod_df = pd.DataFrame({'Temperature': mod_monthly_vals, 'Month': mod_monthly_vals.index.month})
            mod_monthly = mod_df.groupby('Month')['Temperature'].mean()

            month_labels = [MONTH_MAP[m] for m in range(1, 13)]
            fig_wi_monthly = go.Figure()
            fig_wi_monthly.add_trace(go.Scatter(
                x=month_labels, y=[float(orig_monthly.get(m, 0)) for m in range(1, 13)],
                mode='lines+markers', name='Original',
                line=dict(color='#475569', width=2), marker=dict(size=6),
            ))
            fig_wi_monthly.add_trace(go.Scatter(
                x=month_labels, y=[float(mod_monthly.get(m, 0)) for m in range(1, 13)],
                mode='lines+markers', name='Modified',
                line=dict(color='#38bdf8', width=2), marker=dict(size=6),
            ))
            fig_wi_monthly.update_layout(**PLOTLY_LAYOUT, height=300,
                                          yaxis_title='Avg Temp (°C)',
                                          legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'))
            st.plotly_chart(fig_wi_monthly, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
#  TAB 8 — DATA LABORATORY
# ──────────────────────────────────────────────────────────────────────

with tab_lab:
    st.markdown('<div class="section-header">🔬 Interactive Data Explorer</div>', unsafe_allow_html=True)

    # Date range filter
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        start_date = st.date_input("Start Date", value=df.index.min().date(),
                                    min_value=df.index.min().date(), max_value=df.index.max().date(), key="lab_start")
    with col_f2:
        end_date = st.date_input("End Date", value=df.index.max().date(),
                                  min_value=df.index.min().date(), max_value=df.index.max().date(), key="lab_end")
    with col_f3:
        resample_freq = st.selectbox("Resample", ["Daily", "Weekly", "Monthly"], key="lab_resample")

    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    df_filtered = df.loc[mask].copy()

    if len(df_filtered) == 0:
        st.warning("No data in the selected date range.")
    else:
        col_stats, col_dl = st.columns([3, 1])
        with col_stats:
            st.markdown(f"**Selected:** {len(df_filtered):,} records · "
                         f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        with col_dl:
            csv_data = df_filtered.to_csv()
            st.download_button("⬇️ Download CSV", csv_data, "melbourne_temp_filtered.csv",
                                "text/csv", use_container_width=True)

        # Summary statistics
        st.markdown("##### 📊 Summary Statistics for Selected Range")
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        stats = [
            (s1, "Mean", f"{df_filtered['Temperature'].mean():.2f}°C"),
            (s2, "Median", f"{df_filtered['Temperature'].median():.2f}°C"),
            (s3, "Std Dev", f"{df_filtered['Temperature'].std():.2f}°C"),
            (s4, "Min", f"{df_filtered['Temperature'].min():.1f}°C"),
            (s5, "Max", f"{df_filtered['Temperature'].max():.1f}°C"),
            (s6, "Range", f"{df_filtered['Temperature'].max() - df_filtered['Temperature'].min():.1f}°C"),
        ]
        for col, label, value in stats:
            with col:
                st.metric(label, value)

        # Percentile table
        with st.expander("📐 Percentile Analysis", expanded=False):
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            pct_values = [round(df_filtered['Temperature'].quantile(p/100), 2) for p in percentiles]
            pct_df = pd.DataFrame({
                'Percentile': [f'P{p}' for p in percentiles],
                'Temperature (°C)': pct_values,
            })
            st.dataframe(pct_df, use_container_width=True, hide_index=True)

        # Data table
        with st.expander("📋 Raw Data Table", expanded=False):
            st.dataframe(df_filtered, use_container_width=True, height=400)

        # Filtered timeline chart
        with st.expander("📈 Selected Range Timeline", expanded=True):
            if resample_freq == "Weekly":
                plot_data = df_filtered['Temperature'].resample('W').mean()
            elif resample_freq == "Monthly":
                plot_data = df_filtered['Temperature'].resample('ME').mean()
            else:
                plot_data = df_filtered['Temperature']

            fig_filt = go.Figure()
            fig_filt.add_trace(go.Scatter(x=plot_data.index, y=plot_data.values,
                                           mode='lines', line=dict(color='#38bdf8', width=1.2), name='Temperature'))
            if resample_freq == "Daily":
                ma_filt = df_filtered['Temperature'].rolling(7, center=True).mean()
                fig_filt.add_trace(go.Scatter(x=df_filtered.index, y=ma_filt,
                                               mode='lines', line=dict(color='#fbbf24', width=2), name='7-Day MA'))
            fig_filt.update_layout(**PLOTLY_LAYOUT, height=350, yaxis_title='Temperature (°C)',
                                    hovermode='x unified',
                                    legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'))
            st.plotly_chart(fig_filt, use_container_width=True)

        # Year-over-Year Comparison
        with st.expander("📅 Year-over-Year Comparison", expanded=False):
            available_years = sorted(df_ext['Year'].unique())
            selected_years = st.multiselect("Select years to compare:",
                                             available_years, default=available_years[:3], key="yoy_years")

            if selected_years:
                fig_yoy = go.Figure()
                yoy_colors = ['#38bdf8', '#a78bfa', '#fb7185', '#34d399', '#fbbf24',
                              '#f97316', '#06b6d4', '#e879f9', '#60a5fa', '#f43f5e']
                for idx, year in enumerate(selected_years):
                    year_data = df_ext[df_ext['Year'] == year]
                    fig_yoy.add_trace(go.Scatter(
                        x=year_data['DayOfYear'], y=year_data['Temperature'],
                        mode='lines', name=str(year),
                        line=dict(color=yoy_colors[idx % len(yoy_colors)], width=1.3),
                    ))
                fig_yoy.update_layout(**PLOTLY_LAYOUT, height=380,
                                       xaxis_title='Day of Year', yaxis_title='Temperature (°C)',
                                       legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'))
                st.plotly_chart(fig_yoy, use_container_width=True)
            else:
                st.info("Select at least one year to display.")

        # Correlation matrix
        with st.expander("🔗 Feature Correlation Heatmap", expanded=False):
            corr_cols = ['Temperature', 'Month', 'DayOfYear', 'Year']
            corr_data = df_ext[corr_cols].copy()
            if 'TempChange' in df_ext.columns:
                corr_data['DailyChange'] = df_ext['TempChange']
            if 'Anomaly' in df_ext.columns:
                corr_data['Anomaly'] = df_ext['Anomaly']
            corr_matrix = corr_data.corr()

            fig_corr = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale='RdBu_r',
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont=dict(size=11),
                colorbar=dict(title='r', thickness=15),
            ))
            fig_corr.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Day-of-Week analysis
        with st.expander("📅 Day-of-Week Temperature Analysis", expanded=False):
            dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            dow_means = df_ext.groupby('DayOfWeek')['Temperature'].mean()
            dow_labels = [dow_map[i] for i in range(7)]
            dow_vals = [float(dow_means.get(i, 0)) for i in range(7)]

            fig_dow = go.Figure(go.Bar(
                x=dow_labels, y=dow_vals,
                marker=dict(color=dow_vals,
                            colorscale=[[0, '#38bdf8'], [1, '#a78bfa']],
                            line=dict(color='rgba(255,255,255,0.1)', width=1)),
                text=[f"{v:.2f}°" for v in dow_vals], textposition='outside',
            ))
            fig_dow.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                                   yaxis_title='Avg Temperature (°C)')
            st.plotly_chart(fig_dow, use_container_width=True)
            st.caption("Note: Temperature is a natural phenomenon and should not vary by day of week. "
                       "Any small differences are due to random sampling variation.")


# ══════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-footer">
    Melbourne Climate Intelligence Platform v5.0 · ULTRA Edition<br>
    Deep Learning (LSTM / GRU) · SARIMA · AR · ARMA · TensorFlow · Streamlit<br>
    8 Analytics Modules · 30+ Interactive Visualizations · Climate Risk Scoring<br>
    Capstone Project — IOT & Data Science
</div>
""", unsafe_allow_html=True)

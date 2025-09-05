import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import math
from datetime import datetime

# --- Configuración general ---
st.set_page_config(page_title="BTC Spot vs Futuros", layout="wide")
st.title("📊 BTC Futuros vs Spot — Análisis con CoinGecko")

# --- Sidebar ---
risk_free_rate = st.sidebar.number_input("Tasa libre de riesgo anual (r)", value=0.05, step=0.005, format="%.4f")
future_expiry_days = st.sidebar.number_input("Días hasta vencimiento del futuro (T)", value=30, min_value=1, max_value=365)
alert_threshold_pct = st.sidebar.number_input("Umbral de alerta (%)", value=0.5, step=0.1)
show_debug = st.sidebar.checkbox("Mostrar datos raw")

# --- Headers para evitar bloqueos ---
headers = {
    "User-Agent": "Mozilla/5.0 (compatible; MyApp/1.0; +https://example.com)"
}

# --- Funciones para obtener precios ---

@st.cache_data(ttl=60)
def get_spot_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        price = float(data['bitcoin']['usd'])
        if show_debug:
            st.write("🔍 Datos CoinGecko (spot):", data)
        return price
    except Exception as e:
        st.error(f"Error al obtener precio spot: {e}")
        return None

@st.cache_data(ttl=60)
def get_future_price():
    # CoinGecko no tiene futuros, simulamos un precio futuro ligeramente superior
    spot = get_spot_price()
    if spot:
        simulated_future = spot * 1.01  # +1% estimado
        return simulated_future
    else:
        return None

# --- Cargar datos históricos de BTC ---
@st.cache_data(ttl=3600)
def download_btc_data():
    btc = yf.download("BTC-USD", period="1y", interval="1d", progress=False, auto_adjust=True)
    btc['Returns'] = btc['Close'].pct_change()
    btc.dropna(inplace=True)
    return btc

btc_data = download_btc_data()
mu = btc_data['Returns'].mean() * 365
sigma = btc_data['Returns'].std() * np.sqrt(365)

st.sidebar.write(f"📈 Rendimiento anual estimado (mu): {mu:.4f}")
st.sidebar.write(f"📉 Volatilidad anual estimada (sigma): {sigma:.4f}")

# --- Obtener precios actuales ---
spot_price = get_spot_price()
future_price = get_future_price()

if spot_price and future_price:
    st.write(f"📍 Precio Spot (CoinGecko): **${spot_price:,.2f}**")
    st.write(f"📍 Precio Futuro (simulado): **${future_price:,.2f}**")

    # --- Precio futuro teórico ---
    T = future_expiry_days / 365
    r = risk_free_rate
    theoretical_future = spot_price * math.exp(r * T)
    diff = future_price - theoretical_future
    diff_pct = (diff / theoretical_future) * 100

    st.write(f"💡 Precio futuro teórico: **${theoretical_future:,.2f}**")
    st.write(f"🧮 Diferencia: ${diff:,.2f} ({diff_pct:.2f}%)")

    # --- Señal de arbitraje ---
    if diff_pct > alert_threshold_pct:
        st.success("🚀 El futuro está sobrevalorado. Posible oportunidad de venta.")
    elif diff_pct < -alert_threshold_pct:
        st.warning("📉 El futuro está subvaluado. Posible oportunidad de compra.")
    else:
        st.info("ℹ️ Diferencia dentro del umbral normal.")

    # --- Simulación Monte Carlo ---
    st.subheader("🔮 Simulación Monte Carlo para precio spot BTC")
    days_sim = st.slider("Días a simular", 10, 180, 60, step=10)
    num_simulations = st.slider("Número de simulaciones", 50, 500, 200, step=50)

    dt = 1/365
    spot_init = spot_price
    np.random.seed(42)

    simulations = np.zeros((days_sim +1, num_simulations))
    simulations[0] = spot_init

    for t in range(1, days_sim + 1):
        z = np.random.standard_normal(num_simulations)
        simulations[t] = simulations[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    sim_df = pd.DataFrame(simulations)

    # Graficar simulaciones
    st.subheader("📉 Trajectorias simuladas")
    fig, ax = plt.subplots(figsize=(12,6))
    for i in range(min(num_simulations, 20)):
        ax.plot(sim_df.index, sim_df[i], lw=0.8, alpha=0.6)
    ax.set_title(f"Simulaciones Monte Carlo spot BTC ({days_sim} días)")
    ax.set_xlabel("Día")
    ax.set_ylabel("Precio BTC (USD)")
    ax.grid(True)
    st.pyplot(fig)

    # Percentiles
    st.subheader("📊 Percentiles de simulación")
    p10 = sim_df.quantile(0.10, axis=1)
    p50 = sim_df.quantile(0.50, axis=1)
    p90 = sim_df.quantile(0.90, axis=1)

    st.line_chart(pd.DataFrame({"P10": p10, "Mediana (P50)": p50, "P90": p90}))
else:
    st.error("❌ No se pudieron obtener los precios actuales. Verifica tu conexión o espera unos minutos.")

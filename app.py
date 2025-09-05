import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import math
import time
from datetime import datetime
import yfinance as yf

# --- Configuración de página ---
st.set_page_config(page_title="BTC Futuros vs Spot - Avanzado", layout="wide")
st.title("📊 BTC Futuros vs Spot — Análisis avanzado")

# --- Sidebar ---
risk_free_rate = st.sidebar.number_input("Tasa libre de riesgo anual (r)", value=0.05, step=0.005, format="%.4f")
future_expiry_days = st.sidebar.number_input("Días para vencimiento del futuro (T)", value=30, min_value=1, max_value=365)
alert_threshold_pct = st.sidebar.number_input("Umbral de alerta en %", value=0.5, step=0.1)

# --- Descargar datos históricos de BTC ---
@st.cache_data(ttl=3600)
def download_btc_data():
    btc = yf.download("BTC-USD", period="1y", interval="1d", progress=False, auto_adjust=True)
    btc['Returns'] = btc['Close'].pct_change()
    btc.dropna(inplace=True)
    return btc

btc_data = download_btc_data()

# Calcular rendimiento esperado y volatilidad
mu = btc_data['Returns'].mean() * 365
sigma = btc_data['Returns'].std() * np.sqrt(365)
st.sidebar.write(f"📈 Mu (rendimiento anual): {mu:.4f}")
st.sidebar.write(f"📉 Sigma (volatilidad anual): {sigma:.4f}")

# --- Función para obtener precio spot de CoinGecko ---
@st.cache_data(ttl=300)
def get_spot_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(5):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 429:
                st.warning(f"⏳ Esperando por límite 429 (spot)... intento {attempt + 1}/5")
                time.sleep(5 + attempt * 2)
                continue
            response.raise_for_status()
            data = response.json()
            return float(data['bitcoin']['usd'])
        except Exception as e:
            if attempt == 4:
                st.error(f"❌ No se pudo obtener precio spot: {e}")
                return None

# --- Función simulada para precio futuro ---
@st.cache_data(ttl=300)
def get_future_price():
    # En este ejemplo usamos el mismo precio spot para futuro
    # Puedes cambiarlo si conectas con una fuente real
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(5):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 429:
                st.warning(f"⏳ Esperando por límite 429 (futuro)... intento {attempt + 1}/5")
                time.sleep(5 + attempt * 2)
                continue
            response.raise_for_status()
            data = response.json()
            return float(data['bitcoin']['usd']) * (1 + np.random.normal(0, 0.002))  # pequeña desviación
        except Exception as e:
            if attempt == 4:
                st.error(f"❌ No se pudo obtener precio futuro: {e}")
                return None

# --- Obtener precios ---
spot_price = get_spot_price()
future_price = get_future_price()

if spot_price and future_price:
    st.write(f"Precio Spot BTC/USDT: **${spot_price:,.2f}**")
    st.write(f"Precio Futuro BTC/USDT (simulado): **${future_price:,.2f}**")

    # Calcular precio futuro teórico
    T = future_expiry_days / 365
    r = risk_free_rate
    theoretical_future = spot_price * math.exp(r * T)
    st.write(f"💡 Precio futuro teórico (modelo): **${theoretical_future:,.2f}**")

    diff = future_price - theoretical_future
    diff_pct = (diff / theoretical_future) * 100
    st.write(f"📉 Diferencia: ${diff:,.2f} ({diff_pct:.2f}%)")

    if diff_pct > alert_threshold_pct:
        st.success("🚀 El futuro está sobrevalorado. Posible venta de futuro y compra de spot.")
    elif diff_pct < -alert_threshold_pct:
        st.warning("📉 El futuro está subvaluado. Posible compra de futuro y venta de spot.")
    else:
        st.info("ℹ️ Diferencia en rango normal. Sin oportunidad clara de arbitraje.")

    # --- Simulación Monte Carlo ---
    st.subheader("🔮 Simulación Monte Carlo para precio spot BTC")

    days_sim = st.slider("Días a simular", 10, 180, 60, step=10)
    num_simulations = st.slider("Número de simulaciones", 50, 500, 200, step=50)

    dt = 1 / 365
    np.random.seed(42)
    simulations = np.zeros((days_sim + 1, num_simulations))
    simulations[0] = spot_price

    for t in range(1, days_sim + 1):
        z = np.random.standard_normal(num_simulations)
        simulations[t] = simulations[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    sim_df = pd.DataFrame(simulations)

    # Graficar simulaciones
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(min(num_simulations, 20)):
        ax.plot(sim_df.index, sim_df[i], lw=0.8, alpha=0.6)
    ax.set_title(f"Simulaciones Monte Carlo spot BTC ({days_sim} días)")
    ax.set_xlabel("Día")
    ax.set_ylabel("Precio BTC (USD)")
    ax.grid(True)
    st.pyplot(fig)

    # Percentiles
    p10 = sim_df.quantile(0.10, axis=1)
    p50 = sim_df.quantile(0.50, axis=1)
    p90 = sim_df.quantile(0.90, axis=1)
    st.line_chart(pd.DataFrame({"P10": p10, "Mediana": p50, "P90": p90}))

    # --- Backtesting simple ---
    st.subheader("📊 Backtesting estrategia arbitraje")

    lookback_days = st.number_input("Días para backtesting histórico", value=90, min_value=30, max_value=365, step=10)

    if st.button("Ejecutar backtesting"):
        spot_hist = btc_data['Close'].tail(lookback_days)
        fut_theoretical_hist = spot_hist * np.exp(risk_free_rate * (future_expiry_days / 365))

        ruido = np.random.normal(0, 0.005, size=lookback_days)
        fut_real_hist = fut_theoretical_hist * (1 + ruido)

        diff_hist = (fut_real_hist - fut_theoretical_hist) / fut_theoretical_hist * 100

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(spot_hist.index, diff_hist, label="Diferencia % Futuro Real vs Teórico")
        ax2.axhline(alert_threshold_pct, color='green', linestyle='--', label="Umbral superior")
        ax2.axhline(-alert_threshold_pct, color='red', linestyle='--', label="Umbral inferior")
        ax2.set_ylabel("% Diferencia")
        ax2.set_title("Backtesting diferencia futuro vs teórico")
        ax2.legend()
        st.pyplot(fig2)

        aciertos = np.sum((diff_hist > alert_threshold_pct) | (diff_hist < -alert_threshold_pct))
        total = len(diff_hist)
        st.write(f"Días con señal clara de arbitraje: {aciertos} de {total} ({aciertos / total * 100:.2f}%)")

else:
    st.error("❌ No se pudo obtener precios. Revisa conexión o espera unos segundos.")

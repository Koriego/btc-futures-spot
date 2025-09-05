import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import math
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

st.set_page_config(page_title="BTC Futuros vs Spot - Avanzado", layout="wide")
st.title("📊 BTC Futuros vs Spot — Análisis avanzado y alertas")

# --- Sidebar ---
risk_free_rate = st.sidebar.number_input("Tasa libre de riesgo anual (r)", value=0.05, step=0.005, format="%.4f")
future_expiry_days = st.sidebar.number_input("Días para vencimiento del futuro (T)", value=30, min_value=1, max_value=365)
alert_threshold_pct = st.sidebar.number_input("Umbral de alerta en %", value=0.5, step=0.1)
email_alerts = st.sidebar.checkbox("Enviar alertas por email")

# --- Funciones ---

@st.cache_data(ttl=3600)
def download_btc_data():
    btc = yf.download("BTC-USD", period="1y", interval="1d", progress=False, auto_adjust=True)
    btc['Returns'] = btc['Close'].pct_change()
    btc.dropna(inplace=True)
    return btc

btc_data = download_btc_data()

# Cálculo mu y sigma
mu = btc_data['Returns'].mean() * 365
sigma = btc_data['Returns'].std() * np.sqrt(365)

st.sidebar.write(f"📈 Rendimiento anual estimado (mu): {mu:.4f}")
st.sidebar.write(f"📉 Volatilidad anual estimada (sigma): {sigma:.4f}")

# Obtener precios spot y futuro (Binance ejemplo)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

@st.cache_data(ttl=60)
def get_spot_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        st.write("DEBUG spot price data:", data)
        price = float(data.get('price', 0))
        if price == 0:
            st.error("No se encontró el precio en la respuesta de la API Spot.")
        return price
    except Exception as e:
        st.error(f"Error al obtener precio spot: {e}")
        return 0

@st.cache_data(ttl=60)
def get_future_price():
    url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        st.write("DEBUG future price data:", data)
        price = float(data.get('price', 0))
        if price == 0:
            st.error("No se encontró el precio en la respuesta de la API Futuro.")
        return price
    except Exception as e:
        st.error(f"Error al obtener precio futuro: {e}")
        return 0


spot_price = get_spot_price()
future_price = get_future_price()

st.write(f"Precio Spot BTC/USDT: **${spot_price:,.2f}**")
st.write(f"Precio Futuro BTC/USDT: **${future_price:,.2f}**")

# Precio teórico del futuro
T = future_expiry_days / 365
r = risk_free_rate
theoretical_future = spot_price * math.exp(r * T)
st.write(f"Precio futuro teórico (modelo): **${theoretical_future:,.2f}**")

diff = future_price - theoretical_future
diff_pct = (diff / theoretical_future) * 100 if theoretical_future != 0 else 0
st.write(f"Diferencia: ${diff:,.2f} ({diff_pct:.2f}%)")

if diff_pct > alert_threshold_pct:
    st.success("🚀 El futuro está sobrevalorado. Considera vender el futuro y comprar spot.")
    if email_alerts:
        send_email_alert("Futuro BTC sobrevalorado", f"Diferencia de {diff_pct:.2f}% positiva")
elif diff_pct < -alert_threshold_pct:
    st.warning("⚠️ El futuro está subvaluado. Considera comprar el futuro y vender spot.")
    if email_alerts:
        send_email_alert("Futuro BTC subvaluado", f"Diferencia de {diff_pct:.2f}% negativa")
else:
    st.info("ℹ️ La diferencia está dentro del rango normal. Sin señales claras.")

# --- Simulación Monte Carlo del spot ---
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
fig, ax = plt.subplots(figsize=(12,6))
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

# --- Backtesting simple: arbitraje basado en diferencia porcentual ---

st.subheader("📊 Backtesting estrategia arbitraje")

lookback_days = st.number_input("Días para backtesting histórico", value=90, min_value=30, max_value=365, step=10)

if st.button("Ejecutar backtesting"):
    # Precios históricos spot
    spot_hist = btc_data['Close'].tail(lookback_days)
    # Simulación simplificada: usar fórmula para futuros teóricos con tasa fija
    fut_theoretical_hist = spot_hist * np.exp(risk_free_rate * (future_expiry_days / 365))

    # Supongamos futuros reales = teóricos * (1 + ruido)
    ruido = np.random.normal(0, 0.005, size=lookback_days)  # ±0.5% ruido
    fut_real_hist = fut_theoretical_hist * (1 + ruido)

    diff_hist = (fut_real_hist - fut_theoretical_hist) / fut_theoretical_hist * 100

    fig2, ax2 = plt.subplots(figsize=(12,5))
    ax2.plot(spot_hist.index, diff_hist, label="Diferencia % Futuro Real vs Teórico")
    ax2.axhline(alert_threshold_pct, color='green', linestyle='--', label="Umbral superior")
    ax2.axhline(-alert_threshold_pct, color='red', linestyle='--', label="Umbral inferior")
    ax2.set_ylabel("% Diferencia")
    ax2.set_title("Backtesting diferencia futura vs teórica")
    ax2.legend()
    st.pyplot(fig2)

    # Métrica simple de aciertos
    aciertos = np.sum((diff_hist > alert_threshold_pct) | (diff_hist < -alert_threshold_pct))
    total = len(diff_hist)
    st.write(f"Días con señal clara de arbitraje: {aciertos} de {total} ({aciertos/total*100:.2f}%)")

# --- Función para enviar email ---

def send_email_alert(subject, message):
    # CONFIGURA ESTOS DATOS CON TU CUENTA Y SERVIDOR SMTP
    sender_email = "tuemail@gmail.com"
    receiver_email = "destino@email.com"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    password = "tu_contraseña_de_app"

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        st.success("✅ Alerta enviada por email!")
    except Exception as e:
        st.error(f"❌ Error enviando email: {e}")


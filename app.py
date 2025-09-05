import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Simulación GBM de Bitcoin", layout="wide")

st.sidebar.title("⚙️ Configuración de simulación")

num_simulations = st.sidebar.slider("Número de simulaciones", 10, 1000, 100, 10)
days_ahead = st.sidebar.slider("Días a simular", 30, 730, 365, 30)

price_targets_input = st.sidebar.text_input("🎯 Precio(s) objetivo (USD, separados por coma)", "100000,150000,200000")
try:
    price_targets = [float(p.strip()) for p in price_targets_input.split(",") if p.strip()]
except:
    price_targets = [100000, 150000, 200000]

@st.cache_data
def load_btc_data():
    today = datetime.today().strftime('%Y-%m-%d')
    btc = yf.download("BTC-USD", start="2021-01-01", end=today, interval="1d", auto_adjust=True)
    btc = btc.dropna(subset=['Close'])
    return btc['Close']

prices = load_btc_data()
log_returns = np.log(prices / prices.shift(1)).dropna()
mu = log_returns.mean()
sigma = log_returns.std()
S0 = prices[-1]
dt = 1

st.write(f"### Último precio de BTC: ${S0:,.2f}")
st.write(f"Media diaria estimada (μ): {mu:.6f}")
st.write(f"Volatilidad diaria estimada (σ): {sigma:.6f}")

def simulate_gbm(S0, mu, sigma, dt, days, simulations):
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = S0
    for t in range(1, days + 1):
        Z = np.random.standard_normal(simulations)
        price_paths[t] = price_paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return price_paths

simulated_prices = simulate_gbm(S0, mu, sigma, dt, days_ahead, num_simulations)

st.subheader("📈 Simulación Monte Carlo de precios futuros de BTC")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(simulated_prices, color='grey', alpha=0.3, lw=1)
ax.set_xlabel("Días")
ax.set_ylabel("Precio (USD)")
ax.grid(True)

p10 = np.percentile(simulated_prices, 10, axis=1)
p25 = np.percentile(simulated_prices, 25, axis=1)
p50 = np.percentile(simulated_prices, 50, axis=1)
p75 = np.percentile(simulated_prices, 75, axis=1)
p90 = np.percentile(simulated_prices, 90, axis=1)

ax.plot(p10, label='P10', linestyle='--', color='red')
ax.plot(p25, label='P25', linestyle='--', color='orange')
ax.plot(p50, label='Mediana (P50)', color='blue', linewidth=2)
ax.plot(p75, label='P75', linestyle='--', color='green')
ax.plot(p90, label='P90', linestyle='--', color='purple')
ax.fill_between(range(days_ahead + 1), p10, p90, color='gray', alpha=0.1, label='Rango P10-P90')

ax.legend()
st.pyplot(fig)

st.subheader("🎯 Probabilidades de superar precios objetivo")
final_prices = simulated_prices[-1]
for pt in price_targets:
    prob = np.mean(final_prices > pt) * 100
    st.write(f"Probabilidad de superar **${pt:,.0f}**: **{prob:.2f}%**")

st.subheader("📊 Precio simulado al final del período")
percentile_values = {
    'P10': p10[-1],
    'P25': p25[-1],
    'P50': p50[-1],
    'P75': p75[-1],
    'P90': p90[-1]
}
df_percentiles = pd.DataFrame.from_dict(percentile_values, orient='index', columns=['Precio simulado (USD)'])
df_percentiles.index.name = 'Percentil'
st.dataframe(df_percentiles.style.format('${:,.0f}'))

csv_data = pd.DataFrame(simulated_prices).to_csv().encode('utf-8')
st.download_button(
    label="⬇️ Descargar resultados CSV",
    data=csv_data,
    file_name=f"simulacion_gbm_btc_{num_simulations}_sim_{days_ahead}_dias.csv",
    mime='text/csv'
)

st.markdown(
    """
### 🧠 ¿Qué es el Movimiento Browniano Geométrico (GBM)?

El GBM es un modelo estocástico que simula cómo evoluciona el precio de un activo financiero en el tiempo considerando:
- Una **tendencia media** (media diaria de retornos),
- La **volatilidad** (variabilidad aleatoria),
- Y la imposibilidad de que el precio sea negativo.

Este modelo es ampliamente usado en finanzas para simulaciones y valoración de opciones.

---
"""
)

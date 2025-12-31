import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os

# 1. Configuración de la página
st.set_page_config(page_title="StockAI V3 Pro - Multi-Temporal", layout="wide")

# Función para construir y entrenar el modelo rápidamente según la temporalidad
def train_temporal_model(X, y):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    # Entrenamiento corto pero efectivo para entorno web
    model.fit(X, y, epochs=8, batch_size=32, verbose=0)
    return model

# 2. Interfaz y Sidebar
st.title("🤖 StockAI V3: Inteligencia Multi-Temporal")
st.markdown("Analiza cualquier activo en marcos de tiempo Diarios, Semanales o Mensuales.")

with st.sidebar:
    st.header("Configuración de Análisis")
    mode = st.radio("Método de Selección:", ["Lista Predefinida", "Entrada Manual"])
    
    if mode == "Lista Predefinida":
        ticker = st.selectbox("Activo:", ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'BTC-USD', 'ETH-USD', 'GLD', 'SPY'])
    else:
        ticker = st.text_input("Ticker Manual (ej: MELI, EURUSD=X):", "NVDA").upper()
    
    # --- SELECTOR DE TEMPORALIDAD ---
    interval_label = st.selectbox("Marco de Tiempo (Timeframe):", ["Diario", "Semanal", "Mensual"])
    
    interval_map = {"Diario": "1d", "Semanal": "1wk", "Mensual": "1mo"}
    interval_code = interval_map[interval_label]

# 3. Descarga de Datos
data = yf.download(ticker, period="5y", interval=interval_code)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if not data.empty:
    prices_series = data['Close'].dropna()
    prices_array = prices_series.values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices_array)

    # 4. Botón de Ejecución
    if st.button(f"🚀 Analizar Tendencia {interval_label}"):
        with st.spinner(f"Entrenando IA para análisis {interval_label}..."):
            window = 30
            if len(scaled_data) > window:
                X, y = [], []
                for i in range(window, len(scaled_data)):
                    X.append(scaled_data[i-window:i, 0])
                    y.append(scaled_data[i, 0])
                X, y = np.array(X), np.array(y)
                X = X.reshape(X.shape[0], X.shape[1], 1)

                model = train_temporal_model(X, y)
                
                last_window = scaled_data[-window:].reshape(1, window, 1)
                pred_scaled = model.predict(last_window)
                pred_final = float(scaler.inverse_transform(pred_scaled)[0][0])
                
                current_price = float(prices_series.iloc[-1])
                diff = pred_final - current_price
                pct = (diff / current_price) * 100

                # --- LÓGICA DE PRECISIÓN DINÁMICA CORREGIDA ---
                if current_price < 2:
                    precision = "4f"
                elif current_price < 10:
                    precision = "3f"
                else:
                    precision = "2f"
                
                # 5. Visualización de Resultados
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Precio Actual", f"${current_price:,.{precision}}")
                m2.metric(f"Proyección Próx. {interval_label[:-2]}", f"${pred_final:,.{precision}}", f"{diff:,.{precision}}")
                m3.metric("Movimiento Esperado", f"{pct:.2f}%")

                # Gráfico con Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prices_series.index, y=prices_series.values, name="Histórico"))
                
                delta = pd.Timedelta(days=1 if interval_code=="1d" else 7 if interval_code=="1wk" else 30)
                future_date = prices_series.index[-1] + delta
                
                fig.add_trace(go.Scatter(
                    x=[prices_series.index[-1], future_date],
                    y=[current_price, pred_final],
                    name="Tendencia IA",
                    line=dict(color='orange', dash='dash', width=3),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                if diff > 0:
                    st.success(f"📈 La IA sugiere una continuación ALCISTA en el marco {interval_label.lower()}.")
                else:
                    st.warning(f"📉 La IA sugiere una corrección o tendencia BAJISTA en el marco {interval_label.lower()}.")
            else:
                st.error("No hay suficientes datos históricos para este marco de tiempo.")
else:
    st.error("No se pudieron obtener datos del activo seleccionado.")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

def train_v2_multitoken():
    # 1. Cargar el dataset
    if not os.path.exists('data/multi_stock_data.csv'):
        print("❌ No se encontró el archivo de datos. Ejecuta data_downloader.py primero.")
        return

    # Leemos el CSV con low_memory=False para evitar advertencias de tipos mixtos
    df = pd.read_csv('data/multi_stock_data.csv', low_memory=False)
    
    # LIMPIEZA CRUCIAL: Convertir 'Close' a numérico y eliminar errores (NaN)
    # Esto limpia las cabeceras extra que causaron el error anterior
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close', 'Ticker'])
    
    data_list = []
    scalers = {}
    
    tickers = df['Ticker'].unique()
    print(f"🔍 Iniciando procesamiento de {len(tickers)} activos...")

    for ticker in tickers:
        # Filtrar datos por ticker y asegurar orden cronológico
        ticker_data = df[df['Ticker'] == ticker]['Close'].values.reshape(-1, 1)
        
        # Validar que existan suficientes datos para la ventana de 60 días
        if len(ticker_data) <= 60:
            print(f"⚠️ Saltando {ticker}: Datos insuficientes ({len(ticker_data)})")
            continue
            
        scaler = MinMaxScaler(feature_range=(0, 1))
        try:
            # Escalamos cada activo de forma independiente
            scaled_data = scaler.fit_transform(ticker_data)
            scalers[ticker] = scaler
            
            # Crear secuencias de entrenamiento
            prediction_days = 60
            for x in range(prediction_days, len(scaled_data)):
                data_list.append((scaled_data[x-prediction_days:x, 0], scaled_data[x, 0]))
        except Exception as e:
            print(f"❌ Error en {ticker}: {e}")
            continue

    if not data_list:
        print("❌ Error: No se pudieron generar secuencias de entrenamiento. Revisa el CSV.")
        return

    # Convertir a formato numpy para la red neuronal
    X = np.array([item[0] for item in data_list])
    y = np.array([item[1] for item in data_list])
    
    # Reshape para LSTM: [muestras, pasos de tiempo, características]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    print(f"📊 Dataset preparado. Total de secuencias: {len(X)}")

    # 3. Arquitectura del Modelo V2
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=100, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("🚀 Entrenando el 'Cerebro' V2 (esto puede tardar unos minutos)...")
    # 10 épocas es un buen balance para empezar
    model.fit(X, y, epochs=10, batch_size=64)

    # 4. Guardar el modelo y el escalador de referencia (usaremos NVDA como base)
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_model_v2.h5')
    
    if 'NVDA' in scalers:
        joblib.dump(scalers['NVDA'], 'models/scaler.pkl')
    else:
        # Si NVDA no está, guardamos el primero que encontremos
        first_ticker = list(scalers.keys())[0]
        joblib.dump(scalers[first_ticker], 'models/scaler.pkl')
    
    print("\n✅ V2 lista. Modelo guardado en: models/lstm_model_v2.h5")

if __name__ == "__main__":
    train_v2_multitoken()
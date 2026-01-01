import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

def train_model():
    # 1. Cargar el dataset
    if not os.path.exists('data/multi_stock_data.csv'):
        print("❌ Error: No se encuentra el archivo de datos. Ejecuta primero data_downloader.py")
        return

    df = pd.read_csv('data/multi_stock_data.csv')
    
    # 2. Seleccionar las 4 columnas clave
    features = ['Close', 'SMA_100', 'SMA_200', 'RSI']
    # Aseguramos que no haya valores nulos residuales
    data = df[features].dropna().values
    
    # 3. Escalar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 4. Crear secuencias
    X, y = [], []
    window = 60
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i, :]) 
        y.append(scaled_data[i, 0])          
    
    X, y = np.array(X), np.array(y)
    
    # 5. Definir la Red Neuronal
    # Reducimos un poco el ruido con EarlyStopping (se detiene si ya no mejora)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 6. Entrenar
    print(f"Iniciando entrenamiento multivariante con {len(X)} muestras...")
    # shuffle=True es vital cuando los datos vienen de muchos activos diferentes
    model.fit(X, y, batch_size=64, epochs=12, shuffle=True, verbose=1)
    
    # 7. Guardar (Usaremos nombres estándar para que tu App los reconozca)
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_model.h5') # Nombre estándar para la App
    joblib.dump(scaler, 'models/scaler.pkl') # Nombre estándar para la App
    
    print("✅ Modelo entrenado y guardado como 'models/lstm_model.h5'")

if __name__ == "__main__":
    train_model()
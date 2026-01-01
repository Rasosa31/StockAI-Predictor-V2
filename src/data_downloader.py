import yfinance as yf
import pandas as pd
import os

def download_multitoken_data():
    # Lista extendida y diversificada
    tickers = [
        # Big Tech & Crecimiento
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 
        'AMD', 'INTC', 'PYPL', 'NFLX', 'ADBE',
        # Índices
        'SPY', 'QQQ', 'DIA',
        # Cripto
        'BTC-USD', 'ETH-USD',
        # Refugios y Deuda
        'GLD', 'VTI', 'TLT', '^TNX', # ^TNX es el Yield de 10 años
        # Commodities
        'CL=F', # Petróleo Crudo
        # Forex (FX) - Formato Yahoo Finance
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCAD=X', 
        'AUDUSD=X', 'USDCHF=X', 'EURJPY=X'
    ]
    
    all_data = []
    print(f"Iniciando descarga de {len(tickers)} activos con indicadores técnicos...")

    for ticker in tickers:
        try:
            print(f"Procesando {ticker}...")
            # Descargamos 5 años para tener suficiente margen para la SMA 200
            df = yf.download(ticker, period="5y", interval="1d")
            
            if not df.empty:
                # --- CÁLCULO DE INDICADORES (Mecánica del motor) ---
                
                # 1. Medias Móviles (Tendencia)
                df['SMA_100'] = df['Close'].rolling(window=100).mean()
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
                
                # 2. RSI (Momento - 14 periodos)
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Identificadores
                df['Ticker'] = ticker
                
                # Limpiar filas iniciales sin indicadores (los primeros 200 días)
                df.dropna(inplace=True)
                
                all_data.append(df)
        except Exception as e:
            print(f"Error con {ticker}: {e}")

    # Combinamos todo
    if all_data:
        final_df = pd.concat(all_data)
        
        # Crear carpeta data si no existe
        os.makedirs('data', exist_ok=True)
        
        # Guardar el dataset maestro con indicadores
        final_df.to_csv('data/multi_stock_data.csv')
        
        print("\n✅ Proceso completado con éxito.")
        print(f"Archivo guardado en: data/multi_stock_data.csv")
        print(f"Activos procesados: {len(all_data)}")
        print(f"Total de registros (con SMA/RSI): {len(final_df)}")
    else:
        print("❌ No se pudo descargar ningún dato.")

if __name__ == "__main__":
    download_multitoken_data()
# 🤖 StockAI V3 Pro: Inteligencia Multi-Indicador Adaptativa

StockAI V3 Pro es una herramienta avanzada de análisis financiero que utiliza Redes Neuronales Recurrentes (LSTM) combinadas con indicadores técnicos clásicos (RSI y Medias Móviles) para proyectar tendencias de precios en activos financieros.

## 🚀 Características Principales
* **Análisis Multivariante**: El modelo no solo mira el precio, sino también el **RSI** y las **SMA 100/200** para mayor precisión.
* **Motor Adaptativo**: Capacidad única para ajustar el cálculo de indicadores en activos jóvenes o marcos de tiempo con poco historial (como el mensual).
* **Backtesting Integrado**: Permite validar la precisión del modelo (RMSE) antes de realizar proyecciones futuras.
* **Interfaz Profesional**: Dashboard interactivo construido en Streamlit con visualizaciones dinámicas de Plotly.

## 📊 Hallazgos Técnicos (EDA)
Durante el desarrollo y las pruebas (backtesting), se determinó que:
* **Precisión en Estabilidad**: En activos con tendencias cíclicas como Ecopetrol (EC), el modelo logra un **RMSE Diario de ~0.43**, demostrando alta fiabilidad.
* **Desempeño en Volatilidad**: En activos de alto crecimiento como NVIDIA (NVDA), la IA actúa como un seguidor de tendencia robusto, capturando la dirección general a pesar de la volatilidad extrema.
* **Optimizaciones**: La inclusión del RSI ayudó a la red LSTM a prever puntos de agotamiento de tendencia con mayor claridad que los modelos univariantes.

---

## 📖 Manual de Usuario

### 1. Instalación
Para ejecutar este proyecto localmente, asegúrate de tener Python 3.9+ instalado y sigue estos pasos:

1. Clona el repositorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt

   Lanza la aplicación:

Bash
streamlit run app/app.py
2. Configuración del Análisis

Selección de Activo: Puedes elegir uno de la lista predefinida o ingresar un ticker de Yahoo Finance manualmente (ej. MELI, ETH-USD, GC=F).

Marco de Tiempo: Selecciona entre Diario, Semanal o Mensual. El sistema adaptará automáticamente el motor de descarga de datos.

3. Ejecución de Proyecciones

Haz clic en "🚀 Ejecutar Proyección".

La IA entrenará una red LSTM en tiempo real con los últimos datos disponibles.

Observarás tres métricas clave: Precio Actual, Proyección IA y % de Cambio Esperado.

4. Uso del Backtesting (Opcional)

Para validar qué tan bien funciona la IA con el activo seleccionado:

Activa el checkbox "Activar Análisis de Backtesting" en la barra lateral.

Presiona "🔄 Ejecutar Prueba de Precisión".

El sistema comparará los últimos 30 periodos reales contra las predicciones de la IA y te entregará el valor RMSE (entre más bajo, mejor).

🛠️ Requisitos Técnicos
El proyecto requiere las versiones específicas listadas en requirements.txt para garantizar la compatibilidad entre TensorFlow y NumPy.
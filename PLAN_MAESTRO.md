# PLAN MAESTRO - Bot de Forex con IA
## Guia completa de desarrollo paso a paso

---

## VISION GENERAL

Construir un bot de trading automatizado para Forex que combine analisis tecnico clasico
con modelos de Machine Learning (XGBoost/LightGBM) para generar senales de compra/venta.
El bot operara a traves de MetaTrader 5 con gestion de riesgo estricta.

**Stack:** Python + MetaTrader5 + XGBoost + pandas-ta
**Capital inicial:** $100 USD (micro-lotes 0.01)
**Pares:** EURUSD, GBPUSD, USDJPY
**Timeframe principal:** H1

---

## FASE 0 - SETUP DEL PROYECTO
**Duracion estimada:** 30 minutos
**Objetivo:** Tener el entorno listo para desarrollar

### Tareas:
- [x] 0.1 - Crear estructura de carpetas completa (COMPLETADO - 29/29 tests passed)
  ```
  forex_bot/
  ├── config/          # Configuracion del bot
  ├── data/            # Pipeline de datos
  ├── models/          # Modelos ML
  │   └── saved/       # Modelos entrenados (.pkl)
  ├── strategy/        # Logica de trading
  ├── execution/       # Ejecucion de ordenes
  ├── backtesting/     # Motor de backtesting
  ├── utils/           # Utilidades
  └── logs/            # Logs de operaciones
  ```
- [x] 0.2 - Crear requirements.txt con todas las dependencias
  - MetaTrader5
  - pandas, numpy
  - pandas-ta (indicadores tecnicos)
  - scikit-learn
  - xgboost
  - lightgbm
  - joblib (serializar modelos)
  - matplotlib, plotly (graficos)
  - python-telegram-bot (notificaciones)
- [x] 0.3 - Crear .gitignore (excluir credenciales, modelos pesados, logs, __pycache__)
- [x] 0.4 - Crear __init__.py en todos los paquetes
- [x] 0.5 - Instalar dependencias con pip (todas instaladas OK)
- [x] 0.6 - Verificar que MetaTrader 5 esta instalado y accesible (import OK)

---

## FASE 1 - CONFIGURACION
**Duracion estimada:** 1 hora
**Objetivo:** Centralizar todos los parametros del sistema

### Tareas:

- [x] 1.1 - Crear config/settings.py (COMPLETADO - 40/40 tests passed)
  Parametros a incluir:
  - TRADING:
    - SYMBOLS: lista de pares ["EURUSD", "GBPUSD", "USDJPY"]
    - TIMEFRAME_PRIMARY: 60 (H1)
    - TIMEFRAME_HIGHER: 240 (H4)
    - TIMEFRAME_DAILY: 1440 (D1)
    - HISTORY_BARS: 5000
  - RISK MANAGEMENT:
    - INITIAL_BALANCE: 100.0
    - RISK_PER_TRADE: 0.02 (2%)
    - MAX_OPEN_TRADES: 3
    - MAX_DAILY_DRAWDOWN: 0.05 (5%)
    - MAX_TOTAL_DRAWDOWN: 0.15 (15%)
    - MIN_RISK_REWARD: 1.5
    - LOT_SIZE_MIN: 0.01
    - SL_ATR_MULTIPLIER: 1.5
    - TP_ATR_MULTIPLIER: 2.25
    - TRAILING_STOP_ENABLED: True
  - ML MODEL:
    - PREDICTION_HORIZON: 5 velas
    - MIN_MOVEMENT_PIPS: 15
    - CONFIDENCE_THRESHOLD: 0.60
    - MODEL_TYPE: "xgboost"
    - TRAIN_WINDOW_DAYS: 365
    - VALIDATION_WINDOW_DAYS: 60
  - INDICADORES:
    - SMA_PERIODS: [20, 50, 200]
    - EMA_PERIODS: [9, 21]
    - RSI_PERIOD: 14
    - MACD: (12, 26, 9)
    - ATR_PERIOD: 14
    - BB: (20, 2.0)
  - EJECUCION:
    - TRADING_MODE: "demo"
    - CHECK_INTERVAL_SECONDS: 60
    - MAX_SLIPPAGE: 20
    - MAGIC_NUMBER: 123456
  - BACKTESTING:
    - BACKTEST_START_DATE / END_DATE
    - BACKTEST_COMMISSION_PER_LOT: 7.0
    - BACKTEST_SPREAD_PIPS: 1.5

- [x] 1.2 - Crear config/mt5_config.py (template de credenciales)
  - MT5_PATH: ruta al terminal de MT5
  - MT5_LOGIN: numero de cuenta
  - MT5_PASSWORD: contrasena
  - MT5_SERVER: nombre del servidor del broker
  - IMPORTANTE: este archivo NO va al repositorio (.gitignore)
  - Crear config/mt5_config.example.py como template

---

## FASE 2 - DATA PIPELINE
**Duracion estimada:** 3-4 horas
**Objetivo:** Poder recolectar, limpiar y preparar datos de mercado

### Tareas:

- [x] 2.1 - Crear data/collector.py (COMPLETADO)
  Clase: DataCollector
  Metodos:
  - __init__(self): inicializar conexion MT5
  - connect(self) -> bool: conectar a MT5, verificar cuenta
  - disconnect(self): cerrar conexion
  - get_historical_data(symbol, timeframe, bars) -> pd.DataFrame:
    - Usar mt5.copy_rates_from_pos()
    - Retornar DataFrame con columnas: time, open, high, low, close, tick_volume
    - Convertir timestamp a datetime
    - Setear time como index
  - get_multiple_timeframes(symbol) -> dict:
    - Descargar H1, H4 y D1 del mismo par
    - Retornar diccionario {timeframe: DataFrame}
  - get_current_price(symbol) -> dict:
    - Usar mt5.symbol_info_tick()
    - Retornar {bid, ask, spread, time}
  - get_account_info() -> dict:
    - Balance, equity, margin, free_margin, profit
  - save_data(df, filename): guardar DataFrame a CSV
  - load_data(filename) -> pd.DataFrame: cargar CSV

  Manejo de errores:
  - Reconexion automatica si se pierde la conexion
  - Logging de todos los errores
  - Validar que MT5 este instalado antes de intentar conectar

- [x] 2.2 - Crear data/preprocessor.py (COMPLETADO)
  Clase: DataPreprocessor
  Metodos:
  - clean_data(df) -> pd.DataFrame:
    - Eliminar filas con NaN
    - Eliminar duplicados por timestamp
    - Verificar que no haya gaps grandes (fin de semana OK, pero alertar gaps inusuales)
    - Verificar que OHLC sea consistente (high >= open,close; low <= open,close)
  - fill_gaps(df, method="ffill") -> pd.DataFrame:
    - Rellenar gaps pequenos con forward fill
  - normalize_data(df, columns, method="zscore") -> pd.DataFrame:
    - Z-score normalization para features del modelo
    - Min-max normalization como alternativa
    - Guardar scalers para poder des-normalizar
  - split_data(df, train_ratio, val_ratio) -> tuple:
    - Split TEMPORAL (no aleatorio)
    - Retornar (train_df, val_df, test_df)
  - create_walk_forward_splits(df, train_window, val_window, step) -> list:
    - Generar multiples splits para walk-forward analysis
    - Retornar lista de (train_df, val_df) tuples

- [x] 2.3 - Crear data/feature_engine.py (COMPLETADO - 36/36 tests passed)
  Clase: FeatureEngine
  Metodos:
  - add_all_features(df) -> pd.DataFrame:
    - Llamar a todos los metodos de features
    - Retornar DataFrame con todas las features agregadas
  
  Features de TENDENCIA:
  - add_sma(df, periods=[20,50,200]):
    - SMA para cada periodo
    - Distancia del precio a cada SMA (en %)
    - Pendiente de cada SMA (SMA actual vs N periodos atras)
    - Cruces de SMAs (SMA20 > SMA50, SMA50 > SMA200)
  - add_ema(df, periods=[9,21]):
    - EMA para cada periodo
    - Distancia del precio a EMA
    - Cruces de EMAs
  - add_adx(df, period=14):
    - ADX, +DI, -DI
    - ADX > 25 = tendencia fuerte

  Features de MOMENTUM:
  - add_rsi(df, period=14):
    - RSI value
    - RSI zonas: sobrecompra (>70), sobreventa (<30)
    - RSI pendiente (momentum del momentum)
  - add_stochastic(df, k=14, d=3):
    - %K, %D
    - Cruces de stochastic
  - add_macd(df, fast=12, slow=26, signal=9):
    - MACD line, signal line, histograma
    - Cruces de MACD
    - Signo del histograma
  - add_cci(df, period=20):
    - CCI value
    - Zonas extremas (>100, <-100)

  Features de VOLATILIDAD:
  - add_atr(df, period=14):
    - ATR absoluto
    - ATR como % del precio (normalizado)
    - ATR ratio (ATR actual vs ATR promedio)
  - add_bollinger(df, period=20, std=2.0):
    - Banda superior, media, inferior
    - %B (posicion del precio dentro de las bandas)
    - Ancho de bandas (squeeze detection)

  Features de VOLUMEN:
  - add_volume_features(df):
    - Volume MA
    - Volume ratio (actual vs promedio)
    - OBV (On Balance Volume)

  Features de PRICE ACTION:
  - add_price_action(df):
    - Retornos % (1, 3, 5, 10 periodos)
    - Rango de vela (high-low) como % del precio
    - Cuerpo de vela (|close-open|) como % del rango
    - Sombras superior e inferior como % del rango
    - Distancia al high/low de N periodos
    - Higher highs / Lower lows (estructura de mercado)

  Features MULTI-TIMEFRAME:
  - add_higher_timeframe_features(df_primary, df_higher) -> pd.DataFrame:
    - RSI del timeframe superior
    - Tendencia del timeframe superior (por SMA)
    - ATR del timeframe superior
    - Alinear timestamps correctamente (merge asof)

  Feature TARGET (lo que predecimos):
  - create_target(df, horizon=5, min_pips=15) -> pd.Series:
    - Calcular retorno futuro a N velas
    - Clasificar en: 1 (BUY), 0 (HOLD), -1 (SELL)
    - BUY = precio sube > min_pips
    - SELL = precio baja > min_pips
    - HOLD = precio se mueve < min_pips en ambas direcciones
  - remove_lookahead_bias(df):
    - Verificar que ningun feature use datos futuros
    - Solo el target puede mirar hacia adelante

---

## FASE 3 - MODELO DE MACHINE LEARNING
**Duracion estimada:** 4-5 horas
**Objetivo:** Entrenar un modelo que prediga direccion del precio

### Tareas:

- [ ] 3.1 - Crear models/trainer.py
  Clase: ModelTrainer
  Metodos:
  - __init__(self, model_type="xgboost"):
    - Inicializar segun tipo de modelo
  - prepare_features(df) -> (X, y):
    - Separar features del target
    - Eliminar columnas que no son features (OHLCV originales, timestamps)
    - Eliminar filas con NaN (resultado de indicadores con lookback)
    - Retornar X (features matrix) e y (target vector)
  - train(X_train, y_train, X_val, y_val) -> model:
    - Entrenar modelo con early stopping usando validacion
    - Hiperparametros para XGBoost:
      - n_estimators: 500
      - max_depth: 6
      - learning_rate: 0.05
      - subsample: 0.8
      - colsample_bytree: 0.8
      - scale_pos_weight: calcular automaticamente por desbalance de clases
      - early_stopping_rounds: 50
      - eval_metric: "mlogloss"
    - Retornar modelo entrenado
  - optimize_hyperparameters(X, y) -> dict:
    - Grid search o random search de hiperparametros
    - Usar TimeSeriesSplit para cross-validation temporal
    - Parametros a optimizar: max_depth, learning_rate, n_estimators, min_child_weight
    - Retornar mejores hiperparametros
  - evaluate(model, X_test, y_test) -> dict:
    - Classification report (precision, recall, f1 por clase)
    - Confusion matrix
    - Accuracy global
    - Probabilidades de prediccion (para threshold tuning)
  - get_feature_importance(model) -> pd.DataFrame:
    - Feature importance del modelo
    - Ordenar de mayor a menor
    - Util para entender que indicadores son mas predictivos
  - save_model(model, filepath):
    - Guardar modelo con joblib
    - Guardar metadata (fecha entrenamiento, metricas, features usadas)
  - load_model(filepath) -> model:
    - Cargar modelo guardado
  - walk_forward_validation(df, train_window, val_window, step) -> dict:
    - Para cada ventana temporal:
      1. Entrenar en train_window
      2. Predecir en val_window
      3. Calcular metricas
      4. Avanzar step periodos
    - Agregar resultados de todas las ventanas
    - Retornar metricas promedio y por ventana
    - ESTO ES CRITICO para evitar overfitting

- [ ] 3.2 - Crear models/predictor.py
  Clase: ModelPredictor
  Metodos:
  - __init__(self, model_path):
    - Cargar modelo guardado
    - Cargar metadata (features esperadas, scalers)
  - predict(features_df) -> dict:
    - Recibir DataFrame con features de la vela actual
    - Verificar que tiene las features correctas
    - Predecir clase (BUY/SELL/HOLD) y probabilidades
    - Retornar {signal, confidence, probabilities}
  - should_trade(prediction) -> bool:
    - Verificar que confidence > CONFIDENCE_THRESHOLD
    - Retornar True/False
  - get_model_age_days() -> int:
    - Cuantos dias tiene el modelo desde ultimo entrenamiento
    - Si > MODEL_RETRAIN_DAYS, sugerir reentrenamiento

---

## FASE 4 - ESTRATEGIA Y GESTION DE RIESGO
**Duracion estimada:** 3-4 horas
**Objetivo:** Definir cuando operar y cuanto arriesgar

### Tareas:

- [ ] 4.1 - Crear strategy/signal_generator.py
  Clase: SignalGenerator
  Metodos:
  - __init__(self, predictor, feature_engine):
    - Recibir instancias del predictor y feature engine
  - generate_signal(df_h1, df_h4, df_d1) -> dict:
    - Pipeline completo:
      1. Calcular features en H1
      2. Agregar features de H4 y D1 (multi-timeframe)
      3. Pasar al modelo ML para prediccion
      4. Aplicar filtros adicionales
      5. Retornar senal final
    - Retornar {
        symbol, signal (BUY/SELL/HOLD),
        confidence, entry_price,
        sl_price, tp_price,
        reason (texto explicativo)
      }
  - apply_filters(signal, df) -> dict:
    - Filtro de tendencia: no operar contra la tendencia D1
    - Filtro de volatilidad: no operar si ATR es anormalmente bajo/alto
    - Filtro de spread: no operar si spread > umbral
    - Filtro de horario: evitar operar en baja liquidez
      (evitar domingo noche, viernes tarde)
    - Filtro de noticias: (futuro) evitar operar antes de noticias de alto impacto
    - Retornar senal filtrada (puede pasar a HOLD si no pasa filtros)
  - calculate_sl_tp(signal, df) -> (sl, tp):
    - SL basado en ATR: entry +/- (ATR * SL_ATR_MULTIPLIER)
    - TP basado en ratio R:R: SL distance * MIN_RISK_REWARD
    - Ajustar a niveles de soporte/resistencia si estan cerca
    - Verificar que SL no sea demasiado tight (minimo X pips)

- [ ] 4.2 - Crear strategy/risk_manager.py
  Clase: RiskManager
  Metodos:
  - __init__(self, initial_balance):
    - Balance actual
    - Tracking de drawdown diario y total
    - Historial de operaciones del dia
  - calculate_position_size(balance, sl_pips, symbol) -> float:
    - Riesgo en $ = balance * RISK_PER_TRADE
    - Pip value depende del par:
      - EURUSD: $0.10 por pip con 0.01 lot
      - GBPUSD: $0.10 por pip con 0.01 lot
      - USDJPY: ~$0.07 por pip con 0.01 lot (variable)
    - Lot size = riesgo_$ / (sl_pips * pip_value)
    - Redondear al LOT_SIZE_STEP mas cercano
    - Clamp entre LOT_SIZE_MIN y LOT_SIZE_MAX
  - can_open_trade(self) -> (bool, str):
    - Verificar: num operaciones abiertas < MAX_OPEN_TRADES
    - Verificar: drawdown diario < MAX_DAILY_DRAWDOWN
    - Verificar: drawdown total < MAX_TOTAL_DRAWDOWN
    - Verificar: hay suficiente margen libre
    - Retornar (True/False, razon)
  - update_daily_stats(self, trade_result):
    - Actualizar P&L del dia
    - Actualizar drawdown diario
    - Actualizar balance actual
  - reset_daily_stats(self):
    - Reset al inicio de cada dia de trading
  - get_risk_report(self) -> dict:
    - Resumen de riesgo actual
    - Drawdown diario/total
    - Operaciones abiertas
    - Margen usado
  - emergency_close_all(self) -> bool:
    - Si drawdown excede limites, cerrar TODAS las posiciones
    - Desactivar bot hasta revision manual

---

## FASE 5 - EJECUCION DE ORDENES
**Duracion estimada:** 3-4 horas
**Objetivo:** Poder abrir, modificar y cerrar ordenes en MT5

### Tareas:

- [ ] 5.1 - Crear execution/mt5_executor.py
  Clase: MT5Executor
  Metodos:
  - __init__(self):
    - Verificar conexion MT5
  - send_market_order(symbol, order_type, lot, sl, tp, comment) -> dict:
    - Crear request para mt5.order_send()
    - order_type: mt5.ORDER_TYPE_BUY o mt5.ORDER_TYPE_SELL
    - Incluir MAGIC_NUMBER para identificar ordenes del bot
    - Incluir slippage maximo
    - Manejo de errores:
      - Requote -> reintentar con nuevo precio
      - Invalid stops -> ajustar SL/TP a minimos del broker
      - No money -> reportar y no operar
    - Retornar {success, order_ticket, price_executed, error_msg}
  - modify_order(ticket, new_sl, new_tp) -> bool:
    - Modificar SL/TP de una orden existente
    - Usado para trailing stop
  - close_order(ticket, lot=None) -> bool:
    - Cerrar posicion (total o parcial si se especifica lot)
    - Determinar tipo opuesto automaticamente
  - close_all_orders(symbol=None) -> list:
    - Cerrar todas las ordenes (o solo las de un symbol)
    - Retornar lista de resultados
  - get_open_positions() -> list:
    - Obtener todas las posiciones abiertas del bot (filtrar por MAGIC_NUMBER)
    - Retornar lista de dicts con info de cada posicion
  - get_order_history(days=30) -> pd.DataFrame:
    - Historial de ordenes cerradas
    - Para analisis de rendimiento

- [ ] 5.2 - Crear execution/order_manager.py
  Clase: OrderManager
  Metodos:
  - __init__(self, executor, risk_manager):
    - Instancia del executor y risk manager
    - Cache de ordenes abiertas
  - process_signal(signal) -> dict:
    - Recibir senal del SignalGenerator
    - Verificar con RiskManager si puede operar
    - Calcular position size
    - Enviar orden via MT5Executor
    - Registrar operacion
    - Retornar resultado
  - manage_open_positions(self):
    - Revisar todas las posiciones abiertas
    - Para cada posicion:
      - Actualizar trailing stop si aplica
      - Verificar si se debe cerrar por logica adicional
        (ej: cerrar si senal contraria)
    - Actualizar stats del risk manager
  - update_trailing_stops(self):
    - Para cada posicion con trailing stop habilitado:
      - Calcular nuevo SL basado en precio actual y ATR
      - Solo mover SL a favor (nunca alejarlo del precio)
      - BUY: solo subir SL, nunca bajarlo
      - SELL: solo bajar SL, nunca subirlo
  - check_exit_signals(self, current_signals):
    - Si hay senal contraria a una posicion abierta, cerrarla
    - Si el modelo cambia a HOLD, considerar cerrar

---

## FASE 6 - BACKTESTING
**Duracion estimada:** 4-5 horas
**Objetivo:** Poder simular la estrategia con datos historicos

### Tareas:

- [ ] 6.1 - Crear backtesting/backtester.py
  Clase: Backtester
  Metodos:
  - __init__(self, initial_balance, commission, spread):
    - Configurar parametros de simulacion
    - Balance virtual
    - Historial de trades simulados
  - run(df, signals) -> dict:
    - Iterar vela por vela cronologicamente
    - Para cada vela:
      1. Verificar si hay SL/TP activados en posiciones abiertas
         (usar high/low de la vela, no solo close)
      2. Procesar senal (si hay)
      3. Aplicar gestion de riesgo virtual
      4. Registrar estado del portfolio
    - NO USAR DATOS FUTUROS (critical - no lookahead bias)
    - Retornar resultados completos
  - simulate_trade(entry, sl, tp, candles) -> dict:
    - Simular una operacion individual
    - Verificar si se activa SL o TP primero
    - Considerar spread y comision
    - Retornar {profit, duration, exit_reason, exit_price}
  - generate_equity_curve(trades) -> pd.Series:
    - Curva de equity a lo largo del tiempo
    - Para graficar rendimiento
  - generate_report(trades) -> dict:
    - Llamar a metrics.py para calcular todas las metricas
    - Retornar reporte completo

- [ ] 6.2 - Crear backtesting/metrics.py
  Clase: PerformanceMetrics
  Metodos estaticos:
  - calculate_all(trades, equity_curve) -> dict:
    Metricas a calcular:
    
    RENDIMIENTO:
    - total_return: retorno total en %
    - annual_return: retorno anualizado
    - monthly_returns: retorno por mes

    RIESGO:
    - max_drawdown: maximo drawdown en % (y duracion)
    - sharpe_ratio: (retorno - risk_free) / std_retorno
      - Usar risk_free = 0 para simplificar
      - Sharpe > 1.0 es aceptable, > 2.0 es bueno
    - sortino_ratio: como sharpe pero solo con volatilidad negativa
    - calmar_ratio: annual_return / max_drawdown

    OPERACIONES:
    - total_trades: numero total de operaciones
    - win_rate: % de operaciones ganadoras
    - profit_factor: gross_profit / gross_loss (queremos > 1.5)
    - avg_win / avg_loss: tamano promedio de ganancia vs perdida
    - avg_risk_reward: ratio R:R realizado promedio
    - max_consecutive_wins / losses
    - avg_trade_duration: duracion promedio de operaciones

    ESTABILIDAD:
    - recovery_factor: total_profit / max_drawdown
    - payoff_ratio: avg_win / avg_loss
    - expectancy: (win_rate * avg_win) - (loss_rate * avg_loss)

  - print_report(metrics):
    - Imprimir reporte formateado en consola
  - plot_equity_curve(equity_curve):
    - Grafico de curva de equity
  - plot_monthly_returns(monthly_returns):
    - Heatmap de retornos mensuales
  - plot_drawdown(equity_curve):
    - Grafico de drawdown a lo largo del tiempo

---

## FASE 7 - UTILIDADES
**Duracion estimada:** 2 horas
**Objetivo:** Logging y notificaciones

### Tareas:

- [ ] 7.1 - Crear utils/logger.py
  Clase: TradingLogger
  Metodos:
  - __init__(self, log_dir="logs"):
    - Crear directorio de logs si no existe
    - Configurar logging con rotacion diaria
  - log_trade(trade_info):
    - Loggear apertura/cierre de trade con todos los detalles
    - Guardar en archivo CSV para analisis posterior
  - log_signal(signal_info):
    - Loggear senales generadas (aun si no se ejecutan)
  - log_error(error_info):
    - Loggear errores con traceback
  - log_daily_summary(stats):
    - Resumen del dia: P&L, trades, drawdown
  - get_trade_history() -> pd.DataFrame:
    - Leer historial de trades del CSV

- [ ] 7.2 - Crear utils/notifications.py
  Clase: TelegramNotifier
  Metodos:
  - __init__(self, bot_token, chat_id):
    - Configurar bot de Telegram
  - send_message(text):
    - Enviar mensaje simple
  - notify_trade_opened(trade_info):
    - Formato: "BUY EURUSD @ 1.0850 | SL: 1.0820 | TP: 1.0895 | Lot: 0.01"
  - notify_trade_closed(trade_info):
    - Formato: "CERRADO EURUSD | P&L: +$2.50 | Razon: TP Hit"
  - notify_daily_summary(stats):
    - Resumen diario
  - notify_error(error):
    - Alertar errores criticos
  
  NOTA: Las notificaciones son opcionales. El bot funciona sin Telegram.
  Se configura TELEGRAM_ENABLED=True en settings.py si se quiere usar.

---

## FASE 8 - ENTRY POINTS (MAIN)
**Duracion estimada:** 3-4 horas
**Objetivo:** Unir todo en scripts ejecutables

### Tareas:

- [ ] 8.1 - Crear main.py (Bot principal)
  Flujo:
  ```
  1. Cargar configuracion
  2. Inicializar componentes:
     - DataCollector -> conectar a MT5
     - FeatureEngine
     - ModelPredictor -> cargar modelo entrenado
     - SignalGenerator
     - RiskManager
     - MT5Executor
     - OrderManager
     - TradingLogger
     - TelegramNotifier (si habilitado)
  3. Verificar que todo esta OK:
     - Conexion MT5 activa
     - Modelo cargado y no muy viejo
     - Balance suficiente
  4. LOOP PRINCIPAL:
     while True:
       a. Esperar cierre de vela H1 (sincronizar con reloj)
       b. Para cada SYMBOL:
          - Descargar ultimas velas (H1, H4, D1)
          - Calcular features
          - Generar senal
          - Si hay senal operable:
            * Verificar riesgo
            * Ejecutar orden
            * Loggear y notificar
       c. Gestionar posiciones abiertas:
          - Actualizar trailing stops
          - Verificar senales de salida
       d. Log de estado
       e. Dormir hasta proxima vela
  5. Manejo de errores:
     - Try/except general con logging
     - Reconexion automatica a MT5
     - Si error critico: cerrar posiciones y parar
  6. Shutdown graceful con Ctrl+C:
     - Cerrar conexion MT5
     - Log final
     - No cerrar posiciones automaticamente (dejar SL/TP)
  ```

- [ ] 8.2 - Crear backtest_runner.py
  Flujo:
  ```
  1. Cargar configuracion
  2. Inicializar DataCollector y descargar datos historicos
  3. Inicializar FeatureEngine y calcular features
  4. Inicializar ModelTrainer:
     a. Preparar features y target
     b. Walk-forward validation:
        - Para cada ventana temporal:
          * Entrenar modelo
          * Generar predicciones en ventana de test
          * Acumular resultados
     c. Calcular metricas agregadas
  5. Ejecutar backtesting:
     a. Usar predicciones del walk-forward
     b. Simular trades con Backtester
     c. Aplicar gestion de riesgo virtual
  6. Generar reporte:
     a. Metricas de rendimiento
     b. Graficos (equity curve, drawdown, monthly returns)
     c. Feature importance
     d. Log de todas las operaciones simuladas
  7. Decision:
     - Si metricas son buenas -> guardar modelo para live
     - Si no -> ajustar parametros y repetir
  
  Criterios MINIMOS para pasar a demo:
  - Profit Factor > 1.3
  - Sharpe Ratio > 0.8
  - Max Drawdown < 25%
  - Win Rate > 45%
  - Total trades > 100 (muestra significativa)
  - Walk-forward consistente (no solo 1 ventana buena)
  ```

---

## FASE 9 - TESTING Y VALIDACION
**Duracion estimada:** 2-3 dias
**Objetivo:** Verificar que todo funciona antes de operar con dinero real

### Tareas:

- [ ] 9.1 - Test del data pipeline
  - Verificar que se descargan datos correctamente de MT5
  - Verificar que los indicadores se calculan bien
    (comparar RSI calculado vs RSI de TradingView)
  - Verificar que no hay lookahead bias en features
  - Verificar manejo de datos faltantes

- [ ] 9.2 - Test del modelo ML
  - Verificar walk-forward validation
  - Analizar feature importance (tiene sentido?)
  - Verificar que no hay overfitting:
    - Accuracy en train vs test no debe diferir mucho
    - Rendimiento consistente en diferentes ventanas temporales
  - Probar con datos recientes (ultimos 3 meses)

- [ ] 9.3 - Test del backtester
  - Verificar que los resultados son realistas
  - Comparar con operaciones manuales conocidas
  - Verificar que spread y comisiones se aplican correctamente
  - Verificar que SL/TP se activan correctamente con high/low

- [ ] 9.4 - Test de ejecucion en demo
  - Correr bot en cuenta demo de MT5 por 1-2 semanas
  - Verificar que las ordenes se envian correctamente
  - Verificar que SL/TP se configuran bien
  - Verificar trailing stop
  - Verificar gestion de riesgo en vivo
  - Monitorear performance vs backtesting

- [ ] 9.5 - Test de edge cases
  - Que pasa si MT5 se desconecta?
  - Que pasa si el mercado esta cerrado (fin de semana)?
  - Que pasa si no hay suficiente margen?
  - Que pasa si el spread se amplia mucho?
  - Que pasa si el bot se reinicia con posiciones abiertas?

---

## FASE 10 - DEPLOY Y MONITOREO
**Duracion estimada:** 1-2 dias
**Objetivo:** Poner el bot en produccion de forma segura

### Tareas:

- [ ] 10.1 - Configurar para operacion continua
  - Configurar el bot para correr como servicio o tarea programada
  - Asegurar que se reinicia automaticamente si crashea
  - Verificar que MT5 se mantiene abierto

- [ ] 10.2 - Monitoreo
  - Configurar alertas de Telegram para:
    - Cada trade abierto/cerrado
    - Resumen diario
    - Errores criticos
    - Drawdown excesivo
  - Revisar rendimiento semanalmente
  - Comparar rendimiento live vs backtest

- [ ] 10.3 - Mantenimiento
  - Reentrenar modelo cada 30 dias con datos nuevos
  - Revisar feature importance mensualmente
  - Ajustar parametros si las condiciones del mercado cambian
  - Mantener logs para audit

---

## CRITERIOS DE EXITO POR FASE

| Fase | Criterio para avanzar |
|------|----------------------|
| 0-Setup | Entorno funcional, pip install OK |
| 1-Config | Todos los parametros definidos y documentados |
| 2-Data | Se descargan datos de MT5 y se calculan features correctamente |
| 3-ML | Walk-forward validation con Profit Factor > 1.0 |
| 4-Strategy | Senales coherentes con filtros funcionando |
| 5-Execution | Ordenes se envian y reciben correctamente en demo |
| 6-Backtest | Resultados realistas con metricas calculadas |
| 7-Utils | Logs y notificaciones funcionando |
| 8-Main | Bot corre sin errores en modo demo |
| 9-Testing | 2 semanas en demo sin problemas criticos |
| 10-Deploy | Bot operando en live con $100 |

---

## NOTAS IMPORTANTES

1. **NO OPERAR EN LIVE** hasta completar TODAS las fases anteriores
2. **SIEMPRE** empezar en cuenta DEMO
3. El mercado Forex es MUY dificil - la mayoria de traders pierden dinero
4. La IA no garantiza ganancias, solo mejora probabilidades
5. $100 es un capital muy pequeno - las comisiones y spreads pesan mas
6. Gestionar el riesgo es MAS IMPORTANTE que la senal de entrada
7. Nunca arriesgar dinero que no puedas permitirte perder
8. Reentrenar el modelo regularmente - los mercados cambian
9. Mantener un diario de trading para aprender de errores
10. Ser paciente - los resultados consistentes toman meses

---

## ORDEN DE IMPLEMENTACION (RESUMEN)

```
Fase 0: Setup          -> 30 min
Fase 1: Config         -> 1 hora
Fase 2: Data Pipeline  -> 3-4 horas
Fase 3: Modelo ML      -> 4-5 horas
Fase 4: Estrategia     -> 3-4 horas
Fase 5: Ejecucion      -> 3-4 horas
Fase 6: Backtesting    -> 4-5 horas
Fase 7: Utilidades     -> 2 horas
Fase 8: Main           -> 3-4 horas
Fase 9: Testing        -> 2-3 dias
Fase 10: Deploy        -> 1-2 dias
                          --------
TOTAL ESTIMADO:           ~30-40 horas de desarrollo
                          + 2-3 semanas de testing en demo
```

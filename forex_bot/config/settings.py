"""
Configuracion general del bot de Forex.
Todos los parametros ajustables del sistema estan centralizados aqui.
"""
from pathlib import Path
import os

# =============================================================================
# PATHS
# =============================================================================

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Directorios
MODELS_DIR = BASE_DIR / "models" / "saved"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = PROJECT_ROOT / "data_cache"

# Crear directorios si no existen
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TRADING PARAMETERS
# =============================================================================

# Pares de divisas a operar
# NOTA: Exness usa sufijo "m" para cuentas Standard (EURUSDm, GBPUSDm, etc.)
# Otros brokers pueden usar sin sufijo (EURUSD) o con otro sufijo (.i, .e, etc.)
# Ajustar segun tu broker
SYMBOL_SUFFIX = "m"  # Sufijo del broker (Exness Standard = "m", IC Markets = "")
SYMBOLS = ["EURUSDm", "GBPUSDm", "USDJPYm"]

# Nombre base sin sufijo (para calculos de pip, etc.)
SYMBOLS_BASE = {
    "EURUSDm": "EURUSD",
    "GBPUSDm": "GBPUSD",
    "USDJPYm": "USDJPY",
}

# Timeframes
# Constantes de MT5: M1=1, M5=5, M15=15, M30=30, H1=16385, H4=16388, D1=16408
# Usamos los valores reales del enum de MT5
TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
    "W1": 32769,
    "MN1": 49153,
}

TIMEFRAME_PRIMARY = "H1"        # Timeframe principal para senales
TIMEFRAME_HIGHER = "H4"         # Contexto de tendencia
TIMEFRAME_DAILY = "D1"          # Contexto macro

# Cantidad de velas historicas para descargar
HISTORY_BARS = 5000              # ~208 dias en H1

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

INITIAL_BALANCE = 100.0          # Balance inicial en USD
RISK_PER_TRADE = 0.02            # 2% del balance por operacion
MAX_OPEN_TRADES = 3              # Maximo de operaciones simultaneas
MAX_DAILY_DRAWDOWN = 0.05        # 5% drawdown maximo diario
MAX_TOTAL_DRAWDOWN = 0.15        # 15% drawdown maximo total
MIN_RISK_REWARD = 1.5            # Ratio minimo riesgo:recompensa

# Lot sizes
LOT_SIZE_MIN = 0.01              # Micro-lote minimo
LOT_SIZE_MAX = 0.10              # Maximo lot size con $100
LOT_SIZE_STEP = 0.01             # Incremento de lot size

# Stop Loss / Take Profit
SL_ATR_MULTIPLIER = 1.5          # SL = ATR * multiplier
TP_ATR_MULTIPLIER = 2.25         # TP = ATR * multiplier (SL * MIN_RISK_REWARD)
TRAILING_STOP_ENABLED = True
TRAILING_STOP_ATR_MULT = 1.0     # Trailing stop a 1x ATR
TRAILING_STOP_CHECK_SECONDS = 300  # Revisar trailing stop cada 5 minutos (300 seg)

# Pip values por par (para 0.01 lot / micro-lote)
# Estos son valores aproximados, el bot calcula el real en runtime
# Usa nombre con sufijo del broker para busqueda directa
PIP_VALUES = {
    "EURUSDm": 0.10,   # $0.10 por pip con 0.01 lot
    "GBPUSDm": 0.10,   # $0.10 por pip con 0.01 lot
    "USDJPYm": 0.07,   # ~$0.07 por pip con 0.01 lot (variable)
    # Alias sin sufijo para backtesting con datos genericos
    "EURUSD": 0.10,
    "GBPUSD": 0.10,
    "USDJPY": 0.07,
}

# Pip size por par (cuanto vale 1 pip en precio)
PIP_SIZE = {
    "EURUSDm": 0.0001,
    "GBPUSDm": 0.0001,
    "USDJPYm": 0.01,
    # Alias sin sufijo
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
}

# =============================================================================
# ML MODEL PARAMETERS
# =============================================================================

# Target variable
PREDICTION_HORIZON = 3           # Predecir N velas hacia adelante (3 = mejor prediccion que 5)
MIN_MOVEMENT_PIPS = 10           # 0 = target binario (UP/DOWN). >0 = ternario (BUY/HOLD/SELL)
                                 # 10 pips = zona muerta para filtrar ruido (~1 ATR en H1)
CONFIDENCE_THRESHOLD = 0.55      # Probabilidad minima para operar (subido de 0.52 para ser mas selectivo)

# Modelo
MODEL_TYPE = "ensemble"          # "xgboost", "lightgbm", "ensemble" (XGBoost + LightGBM)
CALIBRATE_PROBABILITIES = True   # Calibrar probabilidades post-entrenamiento
MODEL_RETRAIN_DAYS = 30          # Reentrenar cada N dias
TRAIN_WINDOW_DAYS = 365          # Ventana de entrenamiento en dias
VALIDATION_WINDOW_DAYS = 60      # Ventana de validacion en dias

# Features
FEATURE_LOOKBACK = 20            # Periodos de lookback para features

# Feature Selection (basada en analisis SHAP + correlacion + importancia)
# Activar para usar solo features seleccionadas en vez de todas las ~71
FEATURE_SELECTION_ENABLED = True
SELECTED_FEATURES = [
    # === Multi-timeframe (alto SHAP) ===
    "rsi_h4", "trend_h4", "rsi_d1", "trend_d1", "sma50_d1",
    # === Tendencia ===
    "sma_200_dist", "sma_50_dist", "sma_20_dist", "sma_200_slope",
    "sma_50_slope", "sma_20_slope", "sma_cross_50_200",
    # === Momentum ===
    "rsi_14", "rsi_slope", "macd_histogram", "di_minus_14",
    # === Volatilidad ===
    "atr_14", "atr_ratio", "bb_bandwidth", "candle_range_pct",
    # === Volumen ===
    "obv", "volume_ratio", "volume_ma",
    # === Price action ===
    "return_1", "return_3", "return_5",
    "dist_low_10", "dist_low_20", "dist_high_10", "dist_high_20",
    "bullish_candle",
]
# 31 features seleccionadas de 71 (eliminadas 40 = 56% de reduccion)

# XGBoost hyperparameters (API verificada contra XGBoost 3.2.0)
# Nota: early_stopping_rounds y eval_metric van en el constructor de XGBClassifier
# Nota: objective "binary:logistic" para target binario, "multi:softprob" para ternario
XGBOOST_PARAMS = {
    "n_estimators": 500,            # Optimo: early stopping para en ~328
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "gamma": 0.2,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
}

# LightGBM hyperparameters (API verificada contra LightGBM 4.6.0)
# Nota: early stopping se maneja via callbacks en fit(): lgb.early_stopping(stopping_rounds=50)
# Nota: eval_metric se pasa en fit(), no aqui
# Nota: num_class NO se pasa, LGBMClassifier lo infiere automaticamente
LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "num_leaves": 31,
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# =============================================================================
# INDICATORS PARAMETERS
# =============================================================================

# Moving Averages
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [9, 21]

# Oscillators
RSI_PERIOD = 14
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
CCI_PERIOD = 20
ADX_PERIOD = 14

# Volatility
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0

# Volume
OBV_ENABLED = True
VOLUME_MA_PERIOD = 20

# =============================================================================
# EXECUTION PARAMETERS
# =============================================================================

# Modo de operacion
TRADING_MODE = "demo"            # "demo" o "live"
CHECK_INTERVAL_SECONDS = 60      # Revisar cada 60 segundos

# Slippage maximo en puntos
MAX_SLIPPAGE = 20

# Magic number para identificar ordenes del bot
MAGIC_NUMBER = 234567

# Reintentos de conexion
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_SECONDS = 10

# =============================================================================
# NOTIFICATIONS
# =============================================================================

TELEGRAM_ENABLED = True
# Token y chat_id se configuran en mt5_config.py (no va al repo)

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"               # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TRADES = True                # Guardar log de todas las operaciones
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# BACKTESTING
# =============================================================================

BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2025-12-31"
BACKTEST_INITIAL_BALANCE = 100.0
BACKTEST_COMMISSION_PER_LOT = 0.0    # $0 - Exness Standard no cobra comision
BACKTEST_SPREAD_PIPS = 1.0           # ~1 pip spread real de Exness Standard EURUSD

# Criterios minimos para aprobar backtest
BACKTEST_MIN_PROFIT_FACTOR = 1.3
BACKTEST_MIN_SHARPE = 0.8
BACKTEST_MAX_DRAWDOWN = 0.25         # 25%
BACKTEST_MIN_WIN_RATE = 0.45         # 45%
BACKTEST_MIN_TRADES = 100            # Muestra minima significativa

# =============================================================================
# TRADING SCHEDULE (horas UTC)
# =============================================================================

# Evitar operar en estos horarios (baja liquidez)
AVOID_TRADING_HOURS = {
    "friday": list(range(20, 24)),     # Viernes despues de las 20:00 UTC
    "sunday": list(range(0, 22)),      # Domingo antes de las 22:00 UTC
}

# Sesiones de mercado (UTC)
SESSIONS = {
    "sydney":  {"open": 22, "close": 7},
    "tokyo":   {"open": 0,  "close": 9},
    "london":  {"open": 8,  "close": 17},
    "new_york": {"open": 13, "close": 22},
}

# Filtro de sesion: solo operar durante sesiones con buena liquidez
SESSION_FILTER_ENABLED = True
ALLOWED_SESSIONS = ["london", "new_york"]  # Mejores sesiones para EUR/GBP/USD

# Filtro de regimen de mercado
REGIME_FILTER_ENABLED = True    # Evitar operar en mercado lateral (ranging)

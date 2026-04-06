"""
VALIDACION FASE 1 - Configuracion
Verifica que todos los parametros de configuracion estan correctos.
Ejecutar: python -m pytest tests/test_fase1_config.py -v
"""
import sys
from pathlib import Path

# Agregar el directorio raiz al path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


class TestSettingsImport:
    """Verifica que settings.py se importa correctamente."""

    def test_import_settings(self):
        from forex_bot.config import settings
        assert settings is not None

    def test_import_from_init(self):
        """Verificar que el __init__.py exporta las variables."""
        from forex_bot.config import SYMBOLS, TIMEFRAME_PRIMARY
        assert SYMBOLS is not None
        assert TIMEFRAME_PRIMARY is not None


class TestTradingParams:
    """Verifica parametros de trading."""

    def setup_method(self):
        from forex_bot.config import settings
        self.s = settings

    def test_symbols_es_lista(self):
        assert isinstance(self.s.SYMBOLS, list)
        assert len(self.s.SYMBOLS) > 0

    def test_symbols_son_strings(self):
        for sym in self.s.SYMBOLS:
            assert isinstance(sym, str)
            assert len(sym) == 6, f"Simbolo {sym} debe tener 6 caracteres (ej: EURUSD)"

    def test_timeframes_validos(self):
        assert self.s.TIMEFRAME_PRIMARY in self.s.TIMEFRAME_MAP
        assert self.s.TIMEFRAME_HIGHER in self.s.TIMEFRAME_MAP
        assert self.s.TIMEFRAME_DAILY in self.s.TIMEFRAME_MAP

    def test_history_bars_positivo(self):
        assert self.s.HISTORY_BARS > 0
        assert self.s.HISTORY_BARS >= 1000, "Se necesitan al menos 1000 barras para ML"


class TestRiskManagement:
    """Verifica parametros de gestion de riesgo."""

    def setup_method(self):
        from forex_bot.config import settings
        self.s = settings

    def test_balance_positivo(self):
        assert self.s.INITIAL_BALANCE > 0

    def test_risk_per_trade_rango(self):
        assert 0 < self.s.RISK_PER_TRADE <= 0.05, \
            "Risk per trade debe estar entre 0% y 5%"

    def test_max_open_trades_razonable(self):
        assert 1 <= self.s.MAX_OPEN_TRADES <= 10

    def test_drawdown_diario_menor_que_total(self):
        assert self.s.MAX_DAILY_DRAWDOWN < self.s.MAX_TOTAL_DRAWDOWN, \
            "Drawdown diario debe ser menor que el total"

    def test_risk_reward_minimo(self):
        assert self.s.MIN_RISK_REWARD >= 1.0, \
            "El ratio R:R minimo debe ser al menos 1:1"

    def test_lot_sizes_consistentes(self):
        assert self.s.LOT_SIZE_MIN > 0
        assert self.s.LOT_SIZE_MAX >= self.s.LOT_SIZE_MIN
        assert self.s.LOT_SIZE_STEP > 0

    def test_sl_tp_multipliers(self):
        assert self.s.SL_ATR_MULTIPLIER > 0
        assert self.s.TP_ATR_MULTIPLIER > 0
        # TP debe ser >= SL * risk_reward
        expected_tp = self.s.SL_ATR_MULTIPLIER * self.s.MIN_RISK_REWARD
        assert self.s.TP_ATR_MULTIPLIER >= expected_tp, \
            f"TP multiplier ({self.s.TP_ATR_MULTIPLIER}) debe ser >= SL * R:R ({expected_tp})"

    def test_pip_values_para_todos_los_symbols(self):
        for sym in self.s.SYMBOLS:
            assert sym in self.s.PIP_VALUES, f"Falta PIP_VALUE para {sym}"
            assert sym in self.s.PIP_SIZE, f"Falta PIP_SIZE para {sym}"
            assert self.s.PIP_VALUES[sym] > 0
            assert self.s.PIP_SIZE[sym] > 0


class TestMLParams:
    """Verifica parametros del modelo ML."""

    def setup_method(self):
        from forex_bot.config import settings
        self.s = settings

    def test_prediction_horizon_positivo(self):
        assert self.s.PREDICTION_HORIZON > 0

    def test_confidence_threshold_rango(self):
        assert 0.5 <= self.s.CONFIDENCE_THRESHOLD <= 0.95, \
            "Confidence threshold debe estar entre 0.5 y 0.95"

    def test_model_type_valido(self):
        assert self.s.MODEL_TYPE in ["xgboost", "lightgbm", "random_forest"]

    def test_train_window_mayor_que_validation(self):
        assert self.s.TRAIN_WINDOW_DAYS > self.s.VALIDATION_WINDOW_DAYS

    def test_xgboost_params_completos(self):
        params = self.s.XGBOOST_PARAMS
        required_keys = ["n_estimators", "max_depth", "learning_rate", "objective"]
        for key in required_keys:
            assert key in params, f"Falta '{key}' en XGBOOST_PARAMS"

    def test_lightgbm_params_completos(self):
        params = self.s.LIGHTGBM_PARAMS
        required_keys = ["n_estimators", "max_depth", "learning_rate", "objective"]
        for key in required_keys:
            assert key in params, f"Falta '{key}' en LIGHTGBM_PARAMS"


class TestIndicatorParams:
    """Verifica parametros de indicadores tecnicos."""

    def setup_method(self):
        from forex_bot.config import settings
        self.s = settings

    def test_sma_periods_ordenados(self):
        assert self.s.SMA_PERIODS == sorted(self.s.SMA_PERIODS), \
            "SMA_PERIODS debe estar ordenado de menor a mayor"

    def test_ema_periods_positivos(self):
        for p in self.s.EMA_PERIODS:
            assert p > 0

    def test_rsi_period_estandar(self):
        assert self.s.RSI_PERIOD > 0
        assert self.s.RSI_PERIOD == 14, "RSI standard es 14 periodos"

    def test_macd_params_consistentes(self):
        assert self.s.MACD_FAST < self.s.MACD_SLOW, \
            "MACD fast debe ser menor que slow"
        assert self.s.MACD_SIGNAL > 0

    def test_bollinger_params(self):
        assert self.s.BB_PERIOD > 0
        assert self.s.BB_STD > 0

    def test_atr_period_positivo(self):
        assert self.s.ATR_PERIOD > 0


class TestExecutionParams:
    """Verifica parametros de ejecucion."""

    def setup_method(self):
        from forex_bot.config import settings
        self.s = settings

    def test_trading_mode_valido(self):
        assert self.s.TRADING_MODE in ["demo", "live"]

    def test_trading_mode_es_demo(self):
        """SEGURIDAD: en desarrollo siempre debe ser demo."""
        assert self.s.TRADING_MODE == "demo", \
            "PELIGRO: TRADING_MODE debe ser 'demo' durante desarrollo"

    def test_check_interval_positivo(self):
        assert self.s.CHECK_INTERVAL_SECONDS > 0

    def test_magic_number_positivo(self):
        assert self.s.MAGIC_NUMBER > 0

    def test_max_slippage_razonable(self):
        assert 0 < self.s.MAX_SLIPPAGE <= 50


class TestBacktestParams:
    """Verifica parametros de backtesting."""

    def setup_method(self):
        from forex_bot.config import settings
        self.s = settings

    def test_fechas_formato_correcto(self):
        from datetime import datetime
        start = datetime.strptime(self.s.BACKTEST_START_DATE, "%Y-%m-%d")
        end = datetime.strptime(self.s.BACKTEST_END_DATE, "%Y-%m-%d")
        assert end > start, "Fecha fin debe ser posterior a fecha inicio"

    def test_criterios_minimos_razonables(self):
        assert self.s.BACKTEST_MIN_PROFIT_FACTOR > 1.0, \
            "Profit factor minimo debe ser > 1.0"
        assert self.s.BACKTEST_MIN_SHARPE > 0
        assert 0 < self.s.BACKTEST_MAX_DRAWDOWN < 1.0
        assert 0 < self.s.BACKTEST_MIN_WIN_RATE < 1.0
        assert self.s.BACKTEST_MIN_TRADES >= 50


class TestPaths:
    """Verifica que los paths se crean correctamente."""

    def test_base_dir(self):
        from forex_bot.config import settings
        assert settings.BASE_DIR.is_dir()

    def test_models_dir_creado(self):
        from forex_bot.config import settings
        assert settings.MODELS_DIR.is_dir()

    def test_logs_dir_creado(self):
        from forex_bot.config import settings
        assert settings.LOGS_DIR.is_dir()


class TestMT5Config:
    """Verifica que mt5_config existe y tiene la estructura correcta."""

    def test_mt5_config_existe(self):
        config_path = ROOT_DIR / "forex_bot" / "config" / "mt5_config.py"
        assert config_path.is_file(), \
            "mt5_config.py no existe. Copia mt5_config.example.py y llena tus datos"

    def test_mt5_config_example_existe(self):
        example_path = ROOT_DIR / "forex_bot" / "config" / "mt5_config.example.py"
        assert example_path.is_file()

    def test_mt5_config_importable(self):
        from forex_bot.config import MT5_CONFIG_LOADED
        # Solo verificar que se puede importar, no que las credenciales sean reales
        assert isinstance(MT5_CONFIG_LOADED, bool)

    def test_mt5_config_tiene_campos_requeridos(self):
        from forex_bot.config import mt5_config
        assert hasattr(mt5_config, "MT5_PATH")
        assert hasattr(mt5_config, "MT5_LOGIN")
        assert hasattr(mt5_config, "MT5_PASSWORD")
        assert hasattr(mt5_config, "MT5_SERVER")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])

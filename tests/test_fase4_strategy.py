"""
VALIDACION FASE 4 - Estrategia y Gestion de Riesgo
Testea signal_generator y risk_manager.
Ejecutar: python -m pytest tests/test_fase4_strategy.py -v
"""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def sample_ohlcv_h1():
    """500 velas H1 sinteticas."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    returns = np.random.normal(0, 0.0005, n)
    close = 1.08 + np.cumsum(returns)
    high = close + np.abs(np.random.normal(0, 0.0003, n))
    low = close - np.abs(np.random.normal(0, 0.0003, n))
    open_p = close + np.random.normal(0, 0.0002, n)
    high = np.maximum(high, np.maximum(open_p, close))
    low = np.minimum(low, np.minimum(open_p, close))

    df = pd.DataFrame({
        "open": open_p, "high": high, "low": low, "close": close,
        "tick_volume": np.random.randint(100, 5000, n),
        "spread": np.random.randint(5, 20, n),
    }, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def sample_ohlcv_d1():
    """200 velas D1 sinteticas."""
    np.random.seed(99)
    n = 200
    dates = pd.date_range("2023-06-01", periods=n, freq="D")
    returns = np.random.normal(0, 0.003, n)
    close = 1.08 + np.cumsum(returns)
    high = close + np.abs(np.random.normal(0, 0.002, n))
    low = close - np.abs(np.random.normal(0, 0.002, n))
    open_p = close + np.random.normal(0, 0.001, n)
    high = np.maximum(high, np.maximum(open_p, close))
    low = np.minimum(low, np.minimum(open_p, close))

    df = pd.DataFrame({
        "open": open_p, "high": high, "low": low, "close": close,
        "tick_volume": np.random.randint(500, 10000, n),
        "spread": np.random.randint(5, 20, n),
    }, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def trained_predictor(sample_ohlcv_h1, tmp_path):
    """Predictor con modelo XGBoost entrenado sobre datos sinteticos."""
    from forex_bot.data.feature_engine import FeatureEngine
    from forex_bot.models.trainer import ModelTrainer
    from forex_bot.models.predictor import ModelPredictor

    fe = FeatureEngine()
    df = fe.add_all_features(sample_ohlcv_h1)
    df["target"] = fe.create_target(df, horizon=5, min_pips=5, pip_size=0.0001)

    trainer = ModelTrainer(model_type="xgboost")
    X, y = trainer.prepare_features(df)
    split = int(len(X) * 0.8)
    trainer.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
    model_path = trainer.save_model(str(tmp_path / "test_signal"))

    predictor = ModelPredictor(str(model_path))
    return predictor


@pytest.fixture
def risk_manager():
    from forex_bot.strategy.risk_manager import RiskManager
    return RiskManager(initial_balance=100.0)


# =====================================================================
# TESTS: RiskManager - Position Sizing
# =====================================================================

class TestRiskManagerPositionSizing:
    def test_basic_position_size(self, risk_manager):
        lot = risk_manager.calculate_position_size(sl_pips=30, symbol="EURUSD")
        assert lot >= 0.01
        assert lot <= 0.10

    def test_position_size_never_below_min(self, risk_manager):
        # SL muy grande -> lot minimo
        lot = risk_manager.calculate_position_size(sl_pips=500, symbol="EURUSD")
        assert lot == 0.01

    def test_position_size_never_above_max(self, risk_manager):
        # SL muy pequeno -> lot capped at max
        lot = risk_manager.calculate_position_size(sl_pips=1, symbol="EURUSD")
        assert lot <= 0.10

    def test_position_size_respects_step(self, risk_manager):
        from forex_bot.config import settings
        lot = risk_manager.calculate_position_size(sl_pips=25, symbol="EURUSD")
        # Debe ser multiplo del step
        remainder = round(lot % settings.LOT_SIZE_STEP, 10)
        assert remainder == 0 or remainder == settings.LOT_SIZE_STEP

    def test_position_size_zero_sl_raises(self, risk_manager):
        lot = risk_manager.calculate_position_size(sl_pips=0, symbol="EURUSD")
        assert lot == 0.01  # Fallback al minimo

    def test_position_size_different_symbols(self, risk_manager):
        lot_eur = risk_manager.calculate_position_size(sl_pips=30, symbol="EURUSD")
        lot_jpy = risk_manager.calculate_position_size(sl_pips=30, symbol="USDJPY")
        # USDJPY tiene pip_value diferente, lot puede ser diferente
        assert lot_eur >= 0.01
        assert lot_jpy >= 0.01

    def test_position_size_with_custom_balance(self, risk_manager):
        lot_100 = risk_manager.calculate_position_size(sl_pips=30, symbol="EURUSD", balance=100)
        lot_1000 = risk_manager.calculate_position_size(sl_pips=30, symbol="EURUSD", balance=1000)
        # Con mas balance, lot debe ser mayor o igual
        assert lot_1000 >= lot_100


# =====================================================================
# TESTS: RiskManager - Can Open Trade
# =====================================================================

class TestRiskManagerCanTrade:
    def test_can_trade_initially(self, risk_manager):
        can, reason = risk_manager.can_open_trade()
        assert can is True
        assert reason == "OK"

    def test_max_trades_limit(self, risk_manager):
        from forex_bot.config import settings
        # Abrir max trades
        for _ in range(settings.MAX_OPEN_TRADES):
            risk_manager.register_trade_opened()

        can, reason = risk_manager.can_open_trade()
        assert can is False
        assert "Max operaciones" in reason

    def test_daily_drawdown_limit(self, risk_manager):
        from forex_bot.config import settings
        # Simular perdida grande del dia
        loss = -(risk_manager.current_balance * settings.MAX_DAILY_DRAWDOWN + 1)
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(loss)

        can, reason = risk_manager.can_open_trade()
        assert can is False
        assert "Drawdown diario" in reason

    def test_total_drawdown_triggers_emergency(self, risk_manager):
        from forex_bot.config import settings
        # Simular perdida total grande
        loss = -(risk_manager.current_balance * settings.MAX_TOTAL_DRAWDOWN + 1)
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(loss)

        assert risk_manager.is_emergency_stopped

    def test_emergency_stop_blocks_trading(self, risk_manager):
        risk_manager.trigger_emergency_stop("Test")
        can, reason = risk_manager.can_open_trade()
        assert can is False
        assert "EMERGENCY" in reason

    def test_emergency_stop_reset(self, risk_manager):
        risk_manager.trigger_emergency_stop("Test")
        assert risk_manager.is_emergency_stopped
        risk_manager.reset_emergency_stop()
        assert not risk_manager.is_emergency_stopped

    def test_low_margin_blocks_trading(self, risk_manager):
        can, reason = risk_manager.can_open_trade(free_margin=5.0)
        assert can is False
        assert "Margen" in reason


# =====================================================================
# TESTS: RiskManager - Tracking
# =====================================================================

class TestRiskManagerTracking:
    def test_register_trade_updates_count(self, risk_manager):
        risk_manager.register_trade_opened()
        assert risk_manager._open_trades_count == 1
        risk_manager.register_trade_closed(5.0)
        assert risk_manager._open_trades_count == 0

    def test_winning_trade_updates_balance(self, risk_manager):
        initial = risk_manager.current_balance
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(10.0)
        assert risk_manager.current_balance == initial + 10.0

    def test_losing_trade_updates_balance(self, risk_manager):
        initial = risk_manager.current_balance
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(-5.0)
        assert risk_manager.current_balance == initial - 5.0

    def test_peak_balance_tracked(self, risk_manager):
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(20.0)
        assert risk_manager.peak_balance == 120.0

        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(-10.0)
        # Peak no debe bajar
        assert risk_manager.peak_balance == 120.0

    def test_daily_pnl_tracking(self, risk_manager):
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(5.0)
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(-3.0)

        assert risk_manager._daily_pnl == 2.0
        assert risk_manager._daily_trades == 2

    def test_total_pnl_tracking(self, risk_manager):
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(10.0)
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(-3.0)

        assert risk_manager._total_pnl == 7.0
        assert risk_manager._total_trades == 2

    def test_update_balance_syncs(self, risk_manager):
        risk_manager.update_balance(150.0)
        assert risk_manager.current_balance == 150.0
        assert risk_manager.peak_balance == 150.0


# =====================================================================
# TESTS: RiskManager - Drawdown
# =====================================================================

class TestRiskManagerDrawdown:
    def test_no_drawdown_initially(self, risk_manager):
        assert risk_manager.get_daily_drawdown() == 0.0
        assert risk_manager.get_total_drawdown() == 0.0

    def test_daily_drawdown_after_loss(self, risk_manager):
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(-3.0)
        # 3/100 = 3%
        assert abs(risk_manager.get_daily_drawdown() - 0.03) < 0.001

    def test_daily_drawdown_zero_if_profitable(self, risk_manager):
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(10.0)
        assert risk_manager.get_daily_drawdown() == 0.0

    def test_total_drawdown_from_peak(self, risk_manager):
        # Subir a 120
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(20.0)
        assert risk_manager.peak_balance == 120.0

        # Bajar a 108
        risk_manager.register_trade_opened()
        risk_manager.register_trade_closed(-12.0)
        # DD = (120 - 108) / 120 = 10%
        assert abs(risk_manager.get_total_drawdown() - 0.10) < 0.001


# =====================================================================
# TESTS: RiskManager - Report
# =====================================================================

class TestRiskManagerReport:
    def test_risk_report_structure(self, risk_manager):
        report = risk_manager.get_risk_report()

        required_keys = [
            "current_balance", "initial_balance", "peak_balance",
            "total_pnl", "total_trades", "daily_pnl", "daily_trades",
            "daily_drawdown", "total_drawdown", "open_trades",
            "emergency_stop", "risk_per_trade",
        ]
        for key in required_keys:
            assert key in report, f"Falta '{key}' en risk report"

    def test_repr(self, risk_manager):
        text = repr(risk_manager)
        assert "RiskManager" in text
        assert "balance" in text


# =====================================================================
# TESTS: SignalGenerator
# =====================================================================

class TestSignalGeneratorSLTP:
    """Test del calculo de SL/TP."""

    def test_calculate_sl_tp_buy(self, sample_ohlcv_h1):
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv_h1)

        sg = SignalGenerator(predictor=None, feature_engine=fe)
        entry = df["close"].iloc[-1]
        sl, tp, sl_pips, tp_pips = sg.calculate_sl_tp("BUY", entry, df, "EURUSD")

        # BUY: SL < entry < TP
        assert sl < entry
        assert tp > entry
        assert sl_pips > 0
        assert tp_pips > 0
        # R:R minimo
        assert tp_pips >= sl_pips * 1.0  # Al menos 1:1

    def test_calculate_sl_tp_sell(self, sample_ohlcv_h1):
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv_h1)

        sg = SignalGenerator(predictor=None, feature_engine=fe)
        entry = df["close"].iloc[-1]
        sl, tp, sl_pips, tp_pips = sg.calculate_sl_tp("SELL", entry, df, "EURUSD")

        # SELL: TP < entry < SL
        assert sl > entry
        assert tp < entry
        assert sl_pips > 0
        assert tp_pips > 0

    def test_sl_minimum_10_pips(self, sample_ohlcv_h1):
        """SL nunca debe ser menor a 10 pips."""
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv_h1)

        sg = SignalGenerator(predictor=None, feature_engine=fe)
        entry = df["close"].iloc[-1]
        _, _, sl_pips, _ = sg.calculate_sl_tp("BUY", entry, df, "EURUSD")

        assert sl_pips >= 10.0

    def test_sl_tp_jpy_pair(self, sample_ohlcv_h1):
        """JPY pairs tienen pip_size diferente (0.01 vs 0.0001)."""
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        # Simular datos de USDJPY (precios alrededor de 150)
        df = sample_ohlcv_h1.copy()
        df["close"] = df["close"] * 139  # ~150
        df["open"] = df["open"] * 139
        df["high"] = df["high"] * 139
        df["low"] = df["low"] * 139
        df = fe.add_all_features(df)

        sg = SignalGenerator(predictor=None, feature_engine=fe)
        entry = df["close"].iloc[-1]
        sl, tp, sl_pips, tp_pips = sg.calculate_sl_tp("BUY", entry, df, "USDJPY")

        assert sl < entry
        assert tp > entry
        assert sl_pips >= 10.0


class TestSignalGeneratorFilters:
    """Test de los filtros de la senal."""

    def test_filter_spread_high(self, sample_ohlcv_h1):
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv_h1)

        sg = SignalGenerator(predictor=None, feature_engine=fe)
        prediction = {"signal": "BUY", "confidence": 0.8}

        # Spread alto
        price = {"spread_pips": 10.0}
        result = sg.apply_filters(prediction, df, None, price)
        assert result["passed"] is False
        assert "Spread" in result["reason"]

    def test_filter_spread_normal_passes(self, sample_ohlcv_h1):
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv_h1)

        sg = SignalGenerator(predictor=None, feature_engine=fe)
        prediction = {"signal": "BUY", "confidence": 0.8}

        price = {"spread_pips": 1.5}
        result = sg.apply_filters(prediction, df, None, price)
        # Podria pasar o no dependiendo de otros filtros (ATR, horario)
        # Pero al menos no falla por spread
        if not result["passed"]:
            assert "Spread" not in result["reason"]

    def test_filter_no_crash_without_optional_data(self, sample_ohlcv_h1):
        """Filtros no deben crashear sin datos opcionales."""
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv_h1)

        sg = SignalGenerator(predictor=None, feature_engine=fe)
        prediction = {"signal": "BUY", "confidence": 0.8}

        # Sin D1, sin price
        result = sg.apply_filters(prediction, df, None, None)
        assert isinstance(result, dict)
        assert "passed" in result


class TestSignalGeneratorFullPipeline:
    """Test del pipeline completo de generacion de senal."""

    def test_generate_signal_returns_structure(self, sample_ohlcv_h1, trained_predictor):
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        sg = SignalGenerator(predictor=trained_predictor, feature_engine=fe)

        signal = sg.generate_signal("EURUSD", sample_ohlcv_h1)

        required_keys = [
            "symbol", "signal", "confidence", "entry_price",
            "sl_price", "tp_price", "sl_pips", "tp_pips",
            "reason", "filters_passed", "timestamp",
        ]
        for key in required_keys:
            assert key in signal, f"Falta '{key}' en signal"

        assert signal["symbol"] == "EURUSD"
        assert signal["signal"] in ("BUY", "SELL", "HOLD")

    def test_generate_signal_with_d1(self, sample_ohlcv_h1, sample_ohlcv_d1, trained_predictor):
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        sg = SignalGenerator(predictor=trained_predictor, feature_engine=fe)

        signal = sg.generate_signal("EURUSD", sample_ohlcv_h1, df_d1=sample_ohlcv_d1)
        assert signal["signal"] in ("BUY", "SELL", "HOLD")

    def test_generate_signal_without_model(self, sample_ohlcv_h1):
        """Sin modelo cargado debe retornar HOLD."""
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.models.predictor import ModelPredictor
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        predictor = ModelPredictor()  # Sin modelo
        sg = SignalGenerator(predictor=predictor, feature_engine=fe)

        signal = sg.generate_signal("EURUSD", sample_ohlcv_h1)
        assert signal["signal"] == "HOLD"
        assert "Modelo no cargado" in signal["reason"]

    def test_buy_signal_sl_below_entry(self, sample_ohlcv_h1, trained_predictor):
        """Si hay BUY, SL debe estar debajo del entry."""
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        sg = SignalGenerator(predictor=trained_predictor, feature_engine=fe)

        signal = sg.generate_signal("EURUSD", sample_ohlcv_h1)
        if signal["signal"] == "BUY":
            assert signal["sl_price"] < signal["entry_price"]
            assert signal["tp_price"] > signal["entry_price"]
        elif signal["signal"] == "SELL":
            assert signal["sl_price"] > signal["entry_price"]
            assert signal["tp_price"] < signal["entry_price"]


# =====================================================================
# TEST: Integracion Signal + Risk
# =====================================================================

class TestSignalRiskIntegration:
    """Test de integracion entre SignalGenerator y RiskManager."""

    def test_signal_to_position_size(self, sample_ohlcv_h1, trained_predictor, risk_manager):
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.strategy.signal_generator import SignalGenerator

        fe = FeatureEngine()
        sg = SignalGenerator(predictor=trained_predictor, feature_engine=fe)

        signal = sg.generate_signal("EURUSD", sample_ohlcv_h1)

        if signal["signal"] != "HOLD" and signal["sl_pips"] > 0:
            # Calcular position size basado en SL
            can_trade, reason = risk_manager.can_open_trade()
            if can_trade:
                lot = risk_manager.calculate_position_size(signal["sl_pips"], signal["symbol"])
                assert lot >= 0.01
                assert lot <= 0.10

    def test_multiple_trades_respect_limits(self, risk_manager):
        from forex_bot.config import settings

        # Abrir trades hasta el limite
        for i in range(settings.MAX_OPEN_TRADES):
            can, _ = risk_manager.can_open_trade()
            assert can is True
            risk_manager.register_trade_opened()

        # El siguiente debe fallar
        can, _ = risk_manager.can_open_trade()
        assert can is False

        # Cerrar uno y volver a intentar
        risk_manager.register_trade_closed(2.0)
        can, _ = risk_manager.can_open_trade()
        assert can is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

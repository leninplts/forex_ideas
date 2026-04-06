"""
VALIDACION FASE 9 - Tests de Integracion con MT5 Real
Estos tests REQUIEREN MetaTrader 5 abierto y conectado.
Se saltan automaticamente si MT5 no esta disponible.

Ejecutar:
  python -m pytest tests/test_fase9_integration.py -v
  python -m pytest tests/test_fase9_integration.py -v -k "not slow"  (sin tests lentos)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =====================================================================
# Verificar si MT5 esta disponible
# =====================================================================

def mt5_available() -> bool:
    """Verificar si MT5 esta conectado."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return False
        info = mt5.account_info()
        mt5.shutdown()
        return info is not None
    except Exception:
        return False


MT5_AVAILABLE = mt5_available()
skip_no_mt5 = pytest.mark.skipif(not MT5_AVAILABLE, reason="MT5 no disponible")

# Simbolo de test - usar el del settings (con sufijo del broker)
from forex_bot.config import settings
TEST_SYMBOL = settings.SYMBOLS[0]  # ej: "EURUSDm" en Exness


# =====================================================================
# 9.1 - Test del Data Pipeline con MT5 real
# =====================================================================

@skip_no_mt5
class TestDataPipelineIntegration:
    """Verificar descarga de datos reales de MT5."""

    def test_connect_and_disconnect(self):
        from forex_bot.data.collector import DataCollector

        collector = DataCollector()
        assert collector.connect() is True
        assert collector.is_connected is True

        account = collector.get_account_info()
        assert account is not None
        assert account["balance"] > 0
        assert account["currency"] in ("USD", "EUR", "GBP")

        collector.disconnect()
        assert collector.is_connected is False

    def test_download_primary_symbol_h1(self):
        from forex_bot.data.collector import DataCollector

        with DataCollector() as collector:
            df = collector.get_historical_data(TEST_SYMBOL, "H1", bars=500)

            assert df is not None
            assert len(df) >= 400  # Puede ser menos si mercado cerrado recientemente
            assert all(col in df.columns for col in ["open", "high", "low", "close", "tick_volume"])

            # OHLC consistente
            assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
            assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()

            # Precios realistas para EUR/USD (entre 0.9 y 1.5)
            assert df["close"].min() > 0.9
            assert df["close"].max() < 1.5

    def test_download_multiple_timeframes(self):
        from forex_bot.data.collector import DataCollector

        with DataCollector() as collector:
            data = collector.get_multiple_timeframes(TEST_SYMBOL)

            assert data is not None
            assert "H1" in data
            assert "H4" in data
            assert "D1" in data

            # H1 tiene mas barras que H4, H4 mas que D1
            assert len(data["H1"]) >= len(data["H4"])

    def test_get_current_price(self):
        from forex_bot.data.collector import DataCollector

        with DataCollector() as collector:
            price = collector.get_current_price(TEST_SYMBOL)

            if price is not None:  # Puede ser None si mercado cerrado
                assert price["bid"] > 0
                assert price["ask"] > 0
                assert price["ask"] >= price["bid"]  # Ask siempre >= bid
                assert price["spread_pips"] >= 0

    def test_features_on_real_data(self):
        """Calcular features sobre datos reales y verificar consistencia."""
        from forex_bot.data.collector import DataCollector
        from forex_bot.data.preprocessor import DataPreprocessor
        from forex_bot.data.feature_engine import FeatureEngine

        with DataCollector() as collector:
            df = collector.get_historical_data(TEST_SYMBOL, "H1", bars=500)

        pp = DataPreprocessor()
        df = pp.clean_data(df)

        fe = FeatureEngine()
        df = fe.add_all_features(df)

        # Verificar que no hay infinitos
        feature_cols = fe.get_feature_columns(df)
        for col in feature_cols:
            inf_count = np.isinf(df[col].dropna()).sum()
            assert inf_count == 0, f"Feature '{col}' tiene {inf_count} valores infinitos"

        # RSI debe estar entre 0 y 100
        if "rsi_14" in df.columns:
            valid_rsi = df["rsi_14"].dropna()
            assert (valid_rsi >= 0).all(), "RSI < 0 detectado"
            assert (valid_rsi <= 100).all(), "RSI > 100 detectado"

        # ATR debe ser positivo
        if "atr_14" in df.columns:
            valid_atr = df["atr_14"].dropna()
            assert (valid_atr >= 0).all(), "ATR negativo detectado"

        # Bollinger: upper >= middle >= lower
        if all(c in df.columns for c in ["bb_upper", "bb_middle", "bb_lower"]):
            valid = df[["bb_upper", "bb_middle", "bb_lower"]].dropna()
            assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_save_and_reload_real_data(self, tmp_path):
        from forex_bot.data.collector import DataCollector
        from forex_bot.config import settings

        original_dir = settings.DATA_DIR
        settings.DATA_DIR = tmp_path

        try:
            with DataCollector() as collector:
                df = collector.get_historical_data(TEST_SYMBOL, "H1", bars=200)
                collector.save_data(df, f"{TEST_SYMBOL}_H1_test")

                loaded = collector.load_data(f"{TEST_SYMBOL}_H1_test")
                assert loaded is not None
                assert len(loaded) == len(df)

                # Precios deben coincidir
                np.testing.assert_allclose(loaded["close"].values, df["close"].values, rtol=1e-10)
        finally:
            settings.DATA_DIR = original_dir


# =====================================================================
# 9.2 - Test del Modelo ML con datos reales
# =====================================================================

@skip_no_mt5
class TestModelIntegration:
    """Entrenar y evaluar modelo con datos reales de MT5."""

    def test_train_model_real_data(self):
        from forex_bot.data.collector import DataCollector
        from forex_bot.data.preprocessor import DataPreprocessor
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.models.trainer import ModelTrainer
        from forex_bot.config import settings

        # Descargar datos
        with DataCollector() as collector:
            df = collector.get_historical_data(TEST_SYMBOL, "H1", bars=2000)

        # Preparar
        pp = DataPreprocessor()
        df = pp.clean_data(df)

        fe = FeatureEngine()
        df = fe.add_all_features(df)

        pip_size = settings.PIP_SIZE.get(TEST_SYMBOL, 0.0001)
        df["target"] = fe.create_target(df, pip_size=pip_size)

        # Entrenar
        trainer = ModelTrainer(model_type="xgboost")
        X, y = trainer.prepare_features(df)

        assert len(X) > 500, f"Muy pocas muestras: {len(X)}"

        split = int(len(X) * 0.8)
        trainer.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        # Evaluar
        metrics = trainer.evaluate(X.iloc[split:], y.iloc[split:])

        assert metrics["accuracy"] > 0
        assert 0 <= metrics["accuracy"] <= 1

        # Feature importance
        fi = trainer.get_feature_importance()
        assert len(fi) > 0
        print(f"\n  Top 5 features con datos reales:")
        for _, row in fi.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

    def test_train_vs_test_overfit_check(self):
        """Verificar que no hay overfitting extremo (train acc >> test acc)."""
        from forex_bot.data.collector import DataCollector
        from forex_bot.data.preprocessor import DataPreprocessor
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.models.trainer import ModelTrainer
        from forex_bot.config import settings

        with DataCollector() as collector:
            df = collector.get_historical_data(TEST_SYMBOL, "H1", bars=2000)

        pp = DataPreprocessor()
        df = pp.clean_data(df)

        fe = FeatureEngine()
        df = fe.add_all_features(df)
        df["target"] = fe.create_target(df, pip_size=settings.PIP_SIZE.get(TEST_SYMBOL, 0.0001))

        trainer = ModelTrainer(model_type="xgboost")
        X, y = trainer.prepare_features(df)

        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]

        val_split = int(len(X_train) * 0.85)
        trainer.train(X_train.iloc[:val_split], y_train.iloc[:val_split],
                      X_train.iloc[val_split:], y_train.iloc[val_split:])

        train_metrics = trainer.evaluate(X_train, y_train)
        test_metrics = trainer.evaluate(X_test, y_test)

        train_acc = train_metrics["accuracy"]
        test_acc = test_metrics["accuracy"]

        print(f"\n  Train accuracy: {train_acc:.2%}")
        print(f"  Test accuracy:  {test_acc:.2%}")
        print(f"  Diferencia:     {train_acc - test_acc:.2%}")

        # Si train >> test por mas de 30%, hay overfitting serio
        assert train_acc - test_acc < 0.30, \
            f"OVERFITTING: train={train_acc:.2%}, test={test_acc:.2%}, diff={train_acc-test_acc:.2%}"


# =====================================================================
# 9.3 - Test del Backtester con datos reales
# =====================================================================

@skip_no_mt5
class TestBacktesterIntegration:
    """Backtest completo con datos reales."""

    def test_full_backtest_real_data(self):
        from forex_bot.backtest_runner import run_backtest
        from forex_bot.data.collector import DataCollector
        from forex_bot.config import settings

        # Descargar datos y guardar a CSV
        with DataCollector() as collector:
            df = collector.get_historical_data(TEST_SYMBOL, "H1", bars=2000)
            csv_path = collector.save_data(df, f"{TEST_SYMBOL}_H1_integration_test")

        # Correr backtest desde CSV
        result = run_backtest(
            symbol=TEST_SYMBOL,
            model_type="xgboost",
            from_csv=str(csv_path),
            save_model=False,
        )

        metrics = result["metrics"]

        print(f"\n  === BACKTEST CON DATOS REALES ===")
        print(f"  Trades:       {metrics['total_trades']}")
        print(f"  Win rate:     {metrics['win_rate_pct']:.1f}%")
        print(f"  Profit:       ${metrics['total_profit']:.2f}")
        print(f"  Sharpe:       {metrics['sharpe_ratio']:.2f}")
        print(f"  Max DD:       {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Profit Factor:{metrics['profit_factor']}")

        # Verificaciones basicas (no importa si gana o pierde, solo que sea realista)
        assert metrics["total_trades"] >= 0
        assert isinstance(result["equity_curve"], pd.Series)
        assert len(result["equity_curve"]) > 0


# =====================================================================
# 9.5 - Test de Edge Cases
# =====================================================================

@skip_no_mt5
class TestEdgeCases:
    """Tests de situaciones limite."""

    def test_reconnection_after_disconnect(self):
        """Verificar que ensure_connected reconecta."""
        from forex_bot.data.collector import DataCollector

        collector = DataCollector()
        collector.connect()
        assert collector.is_connected

        # Simular desconexion
        collector.disconnect()
        assert not collector.is_connected

        # ensure_connected debe reconectar
        result = collector.ensure_connected()
        assert result is True
        assert collector.is_connected

        collector.disconnect()

    def test_invalid_symbol(self):
        from forex_bot.data.collector import DataCollector

        with DataCollector() as collector:
            df = collector.get_historical_data("FAKEPAIR123", "H1", bars=100)
            assert df is None  # Debe retornar None, no crashear

    def test_risk_manager_with_real_balance(self):
        """Verificar que risk manager funciona con balance real."""
        from forex_bot.data.collector import DataCollector
        from forex_bot.strategy.risk_manager import RiskManager

        with DataCollector() as collector:
            account = collector.get_account_info()

        rm = RiskManager(initial_balance=account["balance"])
        lot = rm.calculate_position_size(sl_pips=30, symbol=TEST_SYMBOL)

        assert lot >= 0.01
        assert lot <= 0.10

        can, reason = rm.can_open_trade(free_margin=account["free_margin"])
        assert can is True

    def test_signal_generation_with_real_data(self):
        """Generar senal con datos reales (sin ejecutar)."""
        from forex_bot.data.collector import DataCollector
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.models.predictor import ModelPredictor
        from forex_bot.models.trainer import ModelTrainer
        from forex_bot.strategy.signal_generator import SignalGenerator
        from forex_bot.config import settings

        with DataCollector() as collector:
            df_h1 = collector.get_historical_data(TEST_SYMBOL, "H1", bars=500)
            df_d1 = collector.get_historical_data(TEST_SYMBOL, "D1", bars=200)

        # Entrenar modelo rapido para tener predictor
        fe = FeatureEngine()
        df_feat = fe.add_all_features(df_h1.copy())
        df_feat["target"] = fe.create_target(df_feat, pip_size=settings.PIP_SIZE[TEST_SYMBOL])

        trainer = ModelTrainer(model_type="xgboost")
        X, y = trainer.prepare_features(df_feat)
        split = int(len(X) * 0.8)
        trainer.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        model_path = trainer.save_model()
        try:
            predictor = ModelPredictor(str(model_path))

            sg = SignalGenerator(predictor, fe)
            signal = sg.generate_signal(TEST_SYMBOL, df_h1, df_d1=df_d1)

            assert signal["symbol"] == TEST_SYMBOL
            assert signal["signal"] in ("BUY", "SELL", "HOLD")
            assert 0 <= signal["confidence"] <= 1

            print(f"\n  Senal generada: {signal['signal']} (confianza: {signal['confidence']:.1%})")
            print(f"  Razon: {signal['reason']}")

            if signal["signal"] != "HOLD":
                assert signal["sl_pips"] >= 10
                assert signal["tp_pips"] > 0
                print(f"  Entry: {signal['entry_price']:.5f}")
                print(f"  SL: {signal['sl_price']:.5f} ({signal['sl_pips']:.1f} pips)")
                print(f"  TP: {signal['tp_price']:.5f} ({signal['tp_pips']:.1f} pips)")
        finally:
            # Limpiar modelo guardado
            model_path.unlink(missing_ok=True)
            meta_path = model_path.with_suffix("").with_suffix(".meta.json")
            meta_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])

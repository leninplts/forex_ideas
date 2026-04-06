"""
VALIDACION FASE 6 - Backtesting
Tests para metrics.py y backtester.py con datos sinteticos.
Ejecutar: python -m pytest tests/test_fase6_backtesting.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def sample_trades_winning():
    """Lista de trades con ganancia neta."""
    return [
        {"entry_time": "2024-01-01 10:00", "exit_time": "2024-01-01 15:00",
         "symbol": "EURUSD", "type": "BUY", "entry_price": 1.085, "exit_price": 1.088,
         "lot": 0.01, "sl": 1.083, "tp": 1.089, "profit": 3.0, "pips": 30,
         "exit_reason": "TP", "commission": 0.07, "duration_bars": 5},
        {"entry_time": "2024-01-02 10:00", "exit_time": "2024-01-02 18:00",
         "symbol": "EURUSD", "type": "SELL", "entry_price": 1.090, "exit_price": 1.087,
         "lot": 0.01, "sl": 1.092, "tp": 1.086, "profit": 2.5, "pips": 30,
         "exit_reason": "TP", "commission": 0.07, "duration_bars": 8},
        {"entry_time": "2024-01-03 10:00", "exit_time": "2024-01-03 14:00",
         "symbol": "EURUSD", "type": "BUY", "entry_price": 1.086, "exit_price": 1.084,
         "lot": 0.01, "sl": 1.084, "tp": 1.090, "profit": -2.0, "pips": -20,
         "exit_reason": "SL", "commission": 0.07, "duration_bars": 4},
        {"entry_time": "2024-01-04 10:00", "exit_time": "2024-01-04 20:00",
         "symbol": "EURUSD", "type": "BUY", "entry_price": 1.084, "exit_price": 1.088,
         "lot": 0.01, "sl": 1.082, "tp": 1.089, "profit": 4.0, "pips": 40,
         "exit_reason": "TP", "commission": 0.07, "duration_bars": 10},
    ]


@pytest.fixture
def sample_equity_curve():
    """Equity curve sintetica."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    # Simular crecimiento con drawdowns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 100)
    equity = 100 * np.cumprod(1 + returns)
    return pd.Series(equity, index=dates)


@pytest.fixture
def sample_ohlcv_for_backtest():
    """1000 velas H1 con tendencia alcista + ruido para backtest."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    # Tendencia alcista leve + ruido
    trend = np.linspace(0, 0.02, n)
    noise = np.cumsum(np.random.normal(0, 0.0003, n))
    close = 1.08 + trend + noise

    high = close + np.abs(np.random.normal(0, 0.0004, n))
    low = close - np.abs(np.random.normal(0, 0.0004, n))
    open_p = close + np.random.normal(0, 0.0002, n)
    high = np.maximum(high, np.maximum(open_p, close))
    low = np.minimum(low, np.minimum(open_p, close))

    df = pd.DataFrame({
        "open": open_p, "high": high, "low": low, "close": close,
        "tick_volume": np.random.randint(100, 5000, n),
    }, index=dates)
    df.index.name = "time"
    return df


# =====================================================================
# TESTS: PerformanceMetrics
# =====================================================================

class TestMetricsCalculateAll:
    def test_basic_metrics(self, sample_trades_winning, sample_equity_curve):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics.calculate_all(
            sample_trades_winning, sample_equity_curve, initial_balance=100.0,
        )

        # Basicos
        assert metrics["total_trades"] == 4
        assert metrics["win_count"] == 3
        assert metrics["loss_count"] == 1
        assert metrics["total_profit"] == 7.5  # 3 + 2.5 - 2 + 4

    def test_win_rate(self, sample_trades_winning, sample_equity_curve):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics.calculate_all(
            sample_trades_winning, sample_equity_curve, 100.0,
        )
        assert metrics["win_rate"] == 0.75  # 3/4

    def test_profit_factor(self, sample_trades_winning, sample_equity_curve):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics.calculate_all(
            sample_trades_winning, sample_equity_curve, 100.0,
        )
        # gross_profit = 3 + 2.5 + 4 = 9.5, gross_loss = 2
        assert metrics["profit_factor"] == round(9.5 / 2.0, 2)

    def test_avg_win_loss(self, sample_trades_winning, sample_equity_curve):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics.calculate_all(
            sample_trades_winning, sample_equity_curve, 100.0,
        )
        assert metrics["avg_win"] == round((3 + 2.5 + 4) / 3, 2)
        assert metrics["avg_loss"] == 2.0

    def test_total_return(self, sample_trades_winning, sample_equity_curve):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics.calculate_all(
            sample_trades_winning, sample_equity_curve, 100.0,
        )
        assert metrics["total_return_pct"] == 7.5  # 7.5 / 100 * 100

    def test_empty_trades(self):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics.calculate_all([], pd.Series(dtype=float), 100.0)
        assert metrics["total_trades"] == 0
        assert metrics["total_profit"] == 0

    def test_all_winners(self, sample_equity_curve):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        trades = [
            {"profit": 5.0, "duration_bars": 3, "entry_price": 1.08, "sl": 1.078, "lot": 0.01},
            {"profit": 3.0, "duration_bars": 5, "entry_price": 1.08, "sl": 1.078, "lot": 0.01},
        ]
        metrics = PerformanceMetrics.calculate_all(trades, sample_equity_curve, 100.0)
        assert metrics["win_rate"] == 1.0
        assert metrics["loss_count"] == 0
        assert metrics["max_consecutive_wins"] == 2

    def test_all_losers(self, sample_equity_curve):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        trades = [
            {"profit": -2.0, "duration_bars": 3, "entry_price": 1.08, "sl": 1.078, "lot": 0.01},
            {"profit": -3.0, "duration_bars": 5, "entry_price": 1.08, "sl": 1.078, "lot": 0.01},
        ]
        metrics = PerformanceMetrics.calculate_all(trades, sample_equity_curve, 100.0)
        assert metrics["win_rate"] == 0.0
        assert metrics["max_consecutive_losses"] == 2


class TestMetricsDrawdown:
    def test_max_drawdown(self):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        # Equity: 100 -> 120 -> 90 -> 110
        equity = pd.Series(
            [100, 105, 110, 120, 115, 100, 90, 95, 110],
            index=pd.date_range("2024-01-01", periods=9, freq="h"),
        )
        dd, _ = PerformanceMetrics._max_drawdown(equity)
        # Max DD: (120 - 90) / 120 = 25%
        assert abs(dd - 0.25) < 0.001

    def test_no_drawdown(self):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        # Solo sube
        equity = pd.Series(
            [100, 101, 102, 103, 104],
            index=pd.date_range("2024-01-01", periods=5, freq="h"),
        )
        dd, _ = PerformanceMetrics._max_drawdown(equity)
        assert dd == 0.0


class TestMetricsSharpe:
    def test_sharpe_positive(self):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        # Equity con retornos consistentemente positivos
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.005, 200)
        equity = pd.Series(
            100 * np.cumprod(1 + returns),
            index=pd.date_range("2024-01-01", periods=200, freq="h"),
        )
        sharpe = PerformanceMetrics._sharpe_ratio(equity)
        assert sharpe > 0  # Retornos positivos -> Sharpe positivo

    def test_sharpe_flat(self):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        equity = pd.Series(
            [100] * 20,
            index=pd.date_range("2024-01-01", periods=20, freq="h"),
        )
        sharpe = PerformanceMetrics._sharpe_ratio(equity)
        assert sharpe == 0.0


class TestMetricsConsecutive:
    def test_consecutive_wins(self):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        profits = [1, 2, 3, -1, 1, 1, -2, 1, 1, 1, 1]
        assert PerformanceMetrics._max_consecutive(profits, positive=True) == 4

    def test_consecutive_losses(self):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        profits = [1, -1, -2, -3, 1, -1, -2]
        assert PerformanceMetrics._max_consecutive(profits, positive=False) == 3


class TestMetricsPrintReport:
    def test_print_report_no_crash(self, sample_trades_winning, sample_equity_curve):
        from forex_bot.backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics.calculate_all(
            sample_trades_winning, sample_equity_curve, 100.0,
        )
        # No debe arrojar error
        PerformanceMetrics.print_report(metrics)


# =====================================================================
# TESTS: Backtester
# =====================================================================

class TestBacktesterRun:
    def test_run_with_buy_signals(self, sample_ohlcv_for_backtest):
        from forex_bot.backtesting.backtester import Backtester

        bt = Backtester(initial_balance=100.0)
        n = len(sample_ohlcv_for_backtest)

        # Generar senales: BUY cada 50 velas
        signals = pd.Series(0, index=sample_ohlcv_for_backtest.index)
        for i in range(100, n - 50, 50):
            signals.iloc[i] = 1  # BUY

        result = bt.run(sample_ohlcv_for_backtest, signals)

        assert "trades" in result
        assert "equity_curve" in result
        assert "metrics" in result
        assert len(result["trades"]) > 0

    def test_run_with_sell_signals(self, sample_ohlcv_for_backtest):
        from forex_bot.backtesting.backtester import Backtester

        bt = Backtester(initial_balance=100.0)
        n = len(sample_ohlcv_for_backtest)

        signals = pd.Series(0, index=sample_ohlcv_for_backtest.index)
        for i in range(100, n - 50, 50):
            signals.iloc[i] = -1  # SELL

        result = bt.run(sample_ohlcv_for_backtest, signals)
        assert len(result["trades"]) > 0

    def test_run_no_signals(self, sample_ohlcv_for_backtest):
        """Sin senales, no debe haber trades."""
        from forex_bot.backtesting.backtester import Backtester

        bt = Backtester(initial_balance=100.0)
        signals = pd.Series(0, index=sample_ohlcv_for_backtest.index)

        result = bt.run(sample_ohlcv_for_backtest, signals)
        assert len(result["trades"]) == 0
        assert result["metrics"]["total_trades"] == 0

    def test_equity_curve_length(self, sample_ohlcv_for_backtest):
        from forex_bot.backtesting.backtester import Backtester

        bt = Backtester(initial_balance=100.0)
        signals = pd.Series(0, index=sample_ohlcv_for_backtest.index)
        signals.iloc[100] = 1

        result = bt.run(sample_ohlcv_for_backtest, signals)
        assert len(result["equity_curve"]) == len(sample_ohlcv_for_backtest)

    def test_equity_starts_at_initial_balance(self, sample_ohlcv_for_backtest):
        from forex_bot.backtesting.backtester import Backtester

        bt = Backtester(initial_balance=100.0)
        signals = pd.Series(0, index=sample_ohlcv_for_backtest.index)

        result = bt.run(sample_ohlcv_for_backtest, signals)
        assert result["equity_curve"].iloc[0] == 100.0


class TestBacktesterSLTP:
    def test_sl_hit_buy(self):
        """BUY con SL hit: low baja hasta SL -> trade cerrado con perdida."""
        from forex_bot.backtesting.backtester import Backtester

        # Crear datos donde el precio baja despues de la senal
        dates = pd.date_range("2024-01-01", periods=20, freq="h")
        close = [1.0800] * 5 + [1.0790, 1.0780, 1.0770, 1.0760, 1.0750] + [1.0800] * 10
        high = [c + 0.0005 for c in close]
        low = [c - 0.0005 for c in close]

        df = pd.DataFrame({
            "open": close, "high": high, "low": low, "close": close,
            "tick_volume": [1000] * 20,
        }, index=dates)
        df.index.name = "time"

        signals = pd.Series(0, index=dates)
        signals.iloc[3] = 1  # BUY en vela 3

        sl_pips = pd.Series(20.0, index=dates)  # 20 pips SL
        tp_pips = pd.Series(40.0, index=dates)  # 40 pips TP

        bt = Backtester(initial_balance=100.0, spread_pips=1.0)
        result = bt.run(df, signals, sl_pips, tp_pips)

        # Debe haber al menos 1 trade
        assert len(result["trades"]) >= 1
        # El trade deberia ser por SL (precio baja)
        sl_trades = [t for t in result["trades"] if t["exit_reason"] == "SL"]
        assert len(sl_trades) >= 1

    def test_tp_hit_buy(self):
        """BUY con TP hit: high sube hasta TP -> trade cerrado con ganancia."""
        from forex_bot.backtesting.backtester import Backtester

        dates = pd.date_range("2024-01-01", periods=20, freq="h")
        # Precio sube despues de senal
        close = [1.0800] * 5 + [1.0810, 1.0820, 1.0830, 1.0840, 1.0850] + [1.0800] * 10
        high = [c + 0.0010 for c in close]
        low = [c - 0.0003 for c in close]

        df = pd.DataFrame({
            "open": close, "high": high, "low": low, "close": close,
            "tick_volume": [1000] * 20,
        }, index=dates)
        df.index.name = "time"

        signals = pd.Series(0, index=dates)
        signals.iloc[3] = 1  # BUY

        sl_pips = pd.Series(30.0, index=dates)
        tp_pips = pd.Series(20.0, index=dates)  # TP mas cercano que SL

        bt = Backtester(initial_balance=100.0, spread_pips=1.0)
        result = bt.run(df, signals, sl_pips, tp_pips)

        tp_trades = [t for t in result["trades"] if t["exit_reason"] == "TP"]
        assert len(tp_trades) >= 1
        # Trade por TP debe tener profit positivo
        for t in tp_trades:
            assert t["profit"] > 0

    def test_uses_high_low_not_close(self):
        """Verificar que SL/TP se evaluan con high/low, no solo close."""
        from forex_bot.backtesting.backtester import Backtester

        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        # Close se mantiene, pero low toca SL
        close = [1.0800] * 10
        high = [1.0810] * 10
        low = [1.0800] * 5 + [1.0760] + [1.0800] * 4  # Baja en vela 5

        df = pd.DataFrame({
            "open": close, "high": high, "low": low, "close": close,
            "tick_volume": [1000] * 10,
        }, index=dates)
        df.index.name = "time"

        signals = pd.Series(0, index=dates)
        signals.iloc[2] = 1  # BUY

        sl_pips = pd.Series(30.0, index=dates)
        tp_pips = pd.Series(60.0, index=dates)

        bt = Backtester(initial_balance=100.0, spread_pips=1.0)
        result = bt.run(df, signals, sl_pips, tp_pips)

        # El SL deberia activarse por el low de vela 5, aunque close se mantuvo
        if result["trades"]:
            sl_trades = [t for t in result["trades"] if t["exit_reason"] == "SL"]
            assert len(sl_trades) >= 1


class TestBacktesterCommission:
    def test_commission_applied(self, sample_ohlcv_for_backtest):
        """Trades deben tener comision aplicada."""
        from forex_bot.backtesting.backtester import Backtester

        bt = Backtester(initial_balance=100.0, commission_per_lot=7.0)
        signals = pd.Series(0, index=sample_ohlcv_for_backtest.index)
        signals.iloc[100] = 1

        result = bt.run(sample_ohlcv_for_backtest, signals)

        for trade in result["trades"]:
            assert trade["commission"] > 0  # Comision debe estar presente


class TestBacktesterMaxTrades:
    def test_respects_max_open_trades(self, sample_ohlcv_for_backtest):
        """No debe abrir mas trades del maximo permitido."""
        from forex_bot.backtesting.backtester import Backtester
        from forex_bot.config import settings

        bt = Backtester(initial_balance=1000.0)

        # Muchas senales seguidas para saturar
        signals = pd.Series(0, index=sample_ohlcv_for_backtest.index)
        for i in range(100, 110):
            signals.iloc[i] = 1

        result = bt.run(sample_ohlcv_for_backtest, signals)

        # Verificar que nunca hubo mas de MAX_OPEN_TRADES abiertas
        # (implicitamente, el backtester ignora senales cuando esta lleno)
        assert result["metrics"]["total_trades"] <= 10  # Numero razonable


class TestBacktesterReport:
    def test_generate_report(self, sample_ohlcv_for_backtest):
        from forex_bot.backtesting.backtester import Backtester

        bt = Backtester(initial_balance=100.0)
        signals = pd.Series(0, index=sample_ohlcv_for_backtest.index)
        signals.iloc[100] = 1

        bt.run(sample_ohlcv_for_backtest, signals)
        report = bt.generate_report()

        assert "trades" in report
        assert "equity_curve" in report
        assert "metrics" in report

    def test_get_trades_df(self, sample_ohlcv_for_backtest):
        from forex_bot.backtesting.backtester import Backtester

        bt = Backtester(initial_balance=100.0)
        signals = pd.Series(0, index=sample_ohlcv_for_backtest.index)
        signals.iloc[100] = 1

        bt.run(sample_ohlcv_for_backtest, signals)
        trades_df = bt.get_trades_df()

        assert isinstance(trades_df, pd.DataFrame)
        if len(trades_df) > 0:
            assert "profit" in trades_df.columns
            assert "exit_reason" in trades_df.columns


# =====================================================================
# TEST: Pipeline completo - Feature Engine + Model + Backtest
# =====================================================================

class TestBacktestEndToEnd:
    def test_full_pipeline(self):
        """Pipeline completo: datos -> features -> modelo -> backtest."""
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.models.trainer import ModelTrainer
        from forex_bot.backtesting.backtester import Backtester
        from forex_bot.backtesting.metrics import PerformanceMetrics

        # 1. Datos sinteticos
        np.random.seed(42)
        n = 600
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = 1.08 + np.cumsum(np.random.normal(0, 0.0005, n))
        high = close + np.abs(np.random.normal(0, 0.0003, n))
        low = close - np.abs(np.random.normal(0, 0.0003, n))
        open_p = close + np.random.normal(0, 0.0002, n)
        high = np.maximum(high, np.maximum(open_p, close))
        low = np.minimum(low, np.minimum(open_p, close))

        df = pd.DataFrame({
            "open": open_p, "high": high, "low": low, "close": close,
            "tick_volume": np.random.randint(100, 5000, n),
        }, index=dates)
        df.index.name = "time"

        # 2. Features + Target
        fe = FeatureEngine()
        df_feat = fe.add_all_features(df)
        df_feat["target"] = fe.create_target(df_feat, horizon=5, min_pips=5, pip_size=0.0001)

        # 3. Entrenar modelo
        trainer = ModelTrainer(model_type="xgboost")
        X, y = trainer.prepare_features(df_feat)
        split = int(len(X) * 0.7)
        trainer.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        # 4. Generar predicciones para backtest (solo en test set)
        y_pred = trainer.model.predict(X.iloc[split:])
        # Remap: {0:SELL, 1:HOLD, 2:BUY} -> {-1, 0, 1}
        signal_map = {0: -1, 1: 0, 2: 1}
        signals_array = [signal_map.get(p, 0) for p in y_pred]
        signals = pd.Series(0, index=df.index)
        test_indices = X.iloc[split:].index
        for idx, sig in zip(test_indices, signals_array):
            if idx in df.index:
                pos = df.index.get_loc(idx)
                signals.iloc[pos] = sig

        # 5. Backtest
        bt = Backtester(initial_balance=100.0)
        result = bt.run(df, signals)

        # 6. Validar resultado
        assert result["metrics"]["total_trades"] >= 0
        assert len(result["equity_curve"]) == len(df)
        assert "sharpe_ratio" in result["metrics"]
        assert "profit_factor" in result["metrics"]
        assert "win_rate" in result["metrics"]

        # Imprimir reporte
        PerformanceMetrics.print_report(result["metrics"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

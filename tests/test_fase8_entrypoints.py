"""
VALIDACION FASE 8 - Entry Points (backtest_runner.py + main.py)
Ejecutar: python -m pytest tests/test_fase8_entrypoints.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def synthetic_csv(tmp_path):
    """Crear CSV sintetico para probar backtest_runner sin MT5."""
    np.random.seed(42)
    n = 800
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

    csv_path = tmp_path / "EURUSD_H1.csv"
    df.to_csv(csv_path)
    return str(csv_path)


# =====================================================================
# TESTS: backtest_runner imports
# =====================================================================

class TestBacktestRunnerImports:
    def test_import_module(self):
        from forex_bot.backtest_runner import run_backtest
        assert run_backtest is not None

    def test_import_evaluate_criteria(self):
        from forex_bot.backtest_runner import evaluate_criteria
        assert evaluate_criteria is not None


# =====================================================================
# TESTS: backtest_runner.run_backtest con datos sinteticos
# =====================================================================

class TestBacktestRunnerExecution:
    def test_run_backtest_from_csv(self, synthetic_csv):
        """Ejecutar backtest completo desde CSV sintetico."""
        from forex_bot.backtest_runner import run_backtest

        result = run_backtest(
            symbol="EURUSD",
            model_type="xgboost",
            from_csv=synthetic_csv,
            save_model=False,
        )

        # Verificar estructura de resultado
        assert "metrics" in result
        assert "trades" in result
        assert "equity_curve" in result
        assert "model_eval" in result
        assert "feature_importance" in result
        assert "passed_criteria" in result

    def test_backtest_produces_metrics(self, synthetic_csv):
        from forex_bot.backtest_runner import run_backtest

        result = run_backtest(symbol="EURUSD", from_csv=synthetic_csv)
        metrics = result["metrics"]

        required = [
            "total_trades", "total_profit", "win_rate", "profit_factor",
            "sharpe_ratio", "max_drawdown", "total_return_pct",
        ]
        for key in required:
            assert key in metrics, f"Falta metrica: {key}"

    def test_backtest_produces_trades(self, synthetic_csv):
        from forex_bot.backtest_runner import run_backtest

        result = run_backtest(symbol="EURUSD", from_csv=synthetic_csv)
        # Con datos sinteticos puede o no haber trades, pero la lista existe
        assert isinstance(result["trades"], list)

    def test_backtest_equity_curve(self, synthetic_csv):
        from forex_bot.backtest_runner import run_backtest

        result = run_backtest(symbol="EURUSD", from_csv=synthetic_csv)
        assert isinstance(result["equity_curve"], pd.Series)
        assert len(result["equity_curve"]) > 0

    def test_backtest_feature_importance(self, synthetic_csv):
        from forex_bot.backtest_runner import run_backtest

        result = run_backtest(symbol="EURUSD", from_csv=synthetic_csv)
        fi = result["feature_importance"]
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) > 0

    def test_backtest_with_lightgbm(self, synthetic_csv):
        """Probar con LightGBM en vez de XGBoost."""
        from forex_bot.backtest_runner import run_backtest

        result = run_backtest(
            symbol="EURUSD",
            model_type="lightgbm",
            from_csv=synthetic_csv,
        )
        assert "metrics" in result

    def test_backtest_saves_model(self, synthetic_csv, tmp_path):
        """Verificar que save_model funciona (aunque no apruebe criterios)."""
        from forex_bot.backtest_runner import run_backtest
        from forex_bot.config import settings

        # Bajar criterios para que pase con datos sinteticos
        original_min_trades = settings.BACKTEST_MIN_TRADES
        original_min_pf = settings.BACKTEST_MIN_PROFIT_FACTOR
        original_min_sharpe = settings.BACKTEST_MIN_SHARPE
        original_min_wr = settings.BACKTEST_MIN_WIN_RATE

        try:
            settings.BACKTEST_MIN_TRADES = 0
            settings.BACKTEST_MIN_PROFIT_FACTOR = 0
            settings.BACKTEST_MIN_SHARPE = -999
            settings.BACKTEST_MIN_WIN_RATE = 0

            result = run_backtest(
                symbol="EURUSD",
                from_csv=synthetic_csv,
                save_model=True,
            )
            # Si pasa o no, el test no debe crashear
            assert isinstance(result["passed_criteria"], bool)
        finally:
            settings.BACKTEST_MIN_TRADES = original_min_trades
            settings.BACKTEST_MIN_PROFIT_FACTOR = original_min_pf
            settings.BACKTEST_MIN_SHARPE = original_min_sharpe
            settings.BACKTEST_MIN_WIN_RATE = original_min_wr

            # Limpiar modelos guardados por el test
            for f in settings.MODELS_DIR.glob("*"):
                f.unlink()


# =====================================================================
# TESTS: evaluate_criteria
# =====================================================================

class TestEvaluateCriteria:
    def test_good_metrics_pass(self):
        from forex_bot.backtest_runner import evaluate_criteria

        metrics = {
            "profit_factor": 2.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.10,
            "win_rate": 0.55,
            "total_trades": 200,
        }
        assert evaluate_criteria(metrics) is True

    def test_bad_metrics_fail(self):
        from forex_bot.backtest_runner import evaluate_criteria

        metrics = {
            "profit_factor": 0.5,
            "sharpe_ratio": 0.1,
            "max_drawdown": 0.50,
            "win_rate": 0.20,
            "total_trades": 5,
        }
        assert evaluate_criteria(metrics) is False

    def test_partial_fail(self):
        from forex_bot.backtest_runner import evaluate_criteria

        metrics = {
            "profit_factor": 2.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.10,
            "win_rate": 0.55,
            "total_trades": 10,  # Muy pocos trades
        }
        assert evaluate_criteria(metrics) is False

    def test_handles_inf_profit_factor(self):
        from forex_bot.backtest_runner import evaluate_criteria

        metrics = {
            "profit_factor": "inf",
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.10,
            "win_rate": 0.55,
            "total_trades": 200,
        }
        assert evaluate_criteria(metrics) is True


# =====================================================================
# TESTS: main.py - ForexBot class
# =====================================================================

class TestForexBotClass:
    def test_import(self):
        from forex_bot.main import ForexBot
        assert ForexBot is not None

    def test_init_default_mode(self):
        from forex_bot.main import ForexBot
        bot = ForexBot()
        assert bot.mode == "demo"
        assert bot.running is False

    def test_init_live_mode(self):
        from forex_bot.main import ForexBot
        bot = ForexBot(mode="live")
        assert bot.mode == "live"

    def test_components_none_before_setup(self):
        from forex_bot.main import ForexBot
        bot = ForexBot()
        assert bot.collector is None
        assert bot.predictor is None
        assert bot.risk_manager is None
        assert bot.executor is None

    def test_setup_fails_without_model(self):
        """Setup debe fallar si no hay modelo entrenado guardado."""
        from forex_bot.main import ForexBot
        from forex_bot.config import settings

        # Asegurar que no hay modelos guardados
        import shutil
        saved_dir = settings.MODELS_DIR
        for f in saved_dir.glob("*"):
            f.unlink()

        bot = ForexBot()
        # Sin modelo guardado, setup debe retornar False
        # (puede o no conectar a MT5 dependiendo del entorno)
        success = bot.setup()
        assert success is False

    def test_shutdown_no_crash(self):
        """Shutdown no debe crashear aunque no se haya hecho setup."""
        from forex_bot.main import ForexBot

        bot = ForexBot()
        bot.risk_manager = MagicMock()
        bot.risk_manager.get_risk_report.return_value = {
            "current_balance": 100, "total_pnl": 0,
        }
        bot.collector = MagicMock()

        # No debe crashear
        bot._shutdown()


# =====================================================================
# TEST: Estructura de archivos
# =====================================================================

class TestFileStructure:
    def test_backtest_runner_exists(self):
        path = ROOT_DIR / "forex_bot" / "backtest_runner.py"
        assert path.is_file()

    def test_main_exists(self):
        path = ROOT_DIR / "forex_bot" / "main.py"
        assert path.is_file()

    def test_backtest_runner_has_main(self):
        from forex_bot.backtest_runner import main
        assert callable(main)

    def test_main_has_main(self):
        from forex_bot.main import main
        assert callable(main)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

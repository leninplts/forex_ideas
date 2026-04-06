"""
VALIDACION FASE 7 - Utilidades (Logger + Notifications)
Ejecutar: python -m pytest tests/test_fase7_utils.py -v
"""
import csv
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =====================================================================
# TESTS: TradingLogger
# =====================================================================

class TestTradingLoggerInit:
    def test_import(self):
        from forex_bot.utils.logger import TradingLogger
        assert TradingLogger is not None

    def test_creates_log_dir(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        log_dir = tmp_path / "test_logs"
        tl = TradingLogger(log_dir=str(log_dir))
        assert log_dir.is_dir()

    def test_creates_csv_files(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        log_dir = tmp_path / "test_logs"
        tl = TradingLogger(log_dir=str(log_dir))

        trades_csv = log_dir / "trades.csv"
        signals_csv = log_dir / "signals.csv"
        assert trades_csv.exists()
        assert signals_csv.exists()

    def test_csv_has_headers(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        log_dir = tmp_path / "test_logs"
        tl = TradingLogger(log_dir=str(log_dir))

        trades_csv = log_dir / "trades.csv"
        with open(trades_csv, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert "timestamp" in headers
            assert "symbol" in headers
            assert "profit" in headers


class TestTradingLoggerTrades:
    def test_log_trade(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        tl = TradingLogger(log_dir=str(tmp_path / "logs"))

        trade = {
            "symbol": "EURUSD", "type": "BUY", "lot": 0.01,
            "entry_price": 1.085, "exit_price": 1.088,
            "sl": 1.083, "tp": 1.089, "profit": 3.0,
            "pips": 30, "exit_reason": "TP",
            "commission": 0.07, "duration_bars": 5,
        }
        tl.log_trade(trade, balance_after=103.0)

        # Verificar que se escribio al CSV
        trades_csv = tmp_path / "logs" / "trades.csv"
        df = pd.read_csv(trades_csv)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "EURUSD"
        assert df.iloc[0]["profit"] == 3.0

    def test_log_multiple_trades(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        tl = TradingLogger(log_dir=str(tmp_path / "logs"))

        for i in range(5):
            trade = {"symbol": "EURUSD", "type": "BUY", "profit": i * 1.0}
            tl.log_trade(trade, balance_after=100.0 + i)

        df = pd.read_csv(tmp_path / "logs" / "trades.csv")
        assert len(df) == 5

    def test_get_trade_history(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        tl = TradingLogger(log_dir=str(tmp_path / "logs"))

        trade = {"symbol": "GBPUSD", "type": "SELL", "profit": -2.0}
        tl.log_trade(trade, balance_after=98.0)

        history = tl.get_trade_history()
        assert history is not None
        assert len(history) == 1
        assert history.iloc[0]["symbol"] == "GBPUSD"

    def test_get_trade_history_empty(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        tl = TradingLogger(log_dir=str(tmp_path / "logs"))
        history = tl.get_trade_history()
        # CSV existe pero solo tiene headers -> None
        assert history is None


class TestTradingLoggerSignals:
    def test_log_signal(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        tl = TradingLogger(log_dir=str(tmp_path / "logs"))

        signal = {
            "symbol": "EURUSD", "signal": "BUY", "confidence": 0.75,
            "entry_price": 1.085, "sl_price": 1.083, "tp_price": 1.089,
            "sl_pips": 20, "tp_pips": 40, "filters_passed": True,
            "reason": "ML prediction",
        }
        tl.log_signal(signal, executed=True)

        df = pd.read_csv(tmp_path / "logs" / "signals.csv")
        assert len(df) == 1
        assert df.iloc[0]["signal"] == "BUY"
        assert df.iloc[0]["executed"] == True

    def test_get_signal_history(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        tl = TradingLogger(log_dir=str(tmp_path / "logs"))

        signal = {"symbol": "EURUSD", "signal": "HOLD", "confidence": 0.4}
        tl.log_signal(signal, executed=False)

        history = tl.get_signal_history()
        assert history is not None
        assert len(history) == 1


class TestTradingLoggerOther:
    def test_log_error_no_crash(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        tl = TradingLogger(log_dir=str(tmp_path / "logs"))
        try:
            raise ValueError("test error")
        except Exception as e:
            tl.log_error(e, context="testing")
        # No debe arrojar excepcion

    def test_log_daily_summary_no_crash(self, tmp_path):
        from forex_bot.utils.logger import TradingLogger

        tl = TradingLogger(log_dir=str(tmp_path / "logs"))
        stats = {
            "current_balance": 105.0, "daily_pnl": 5.0,
            "daily_trades": 3, "daily_drawdown_pct": 1.5,
            "total_drawdown_pct": 3.0, "open_trades": 1,
        }
        tl.log_daily_summary(stats)
        # No debe arrojar excepcion


# =====================================================================
# TESTS: TelegramNotifier
# =====================================================================

class TestTelegramNotifierInit:
    def test_import(self):
        from forex_bot.utils.notifications import TelegramNotifier
        assert TelegramNotifier is not None

    def test_disabled_by_default(self):
        from forex_bot.utils.notifications import TelegramNotifier

        notifier = TelegramNotifier(bot_token="", chat_id="")
        assert notifier.is_enabled is False

    def test_disabled_without_token(self):
        from forex_bot.utils.notifications import TelegramNotifier

        notifier = TelegramNotifier(bot_token="", chat_id="12345")
        assert notifier.is_enabled is False

    def test_disabled_without_chat_id(self):
        from forex_bot.utils.notifications import TelegramNotifier

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="")
        assert notifier.is_enabled is False


class TestTelegramNotifierSend:
    def test_send_disabled_returns_false(self):
        from forex_bot.utils.notifications import TelegramNotifier

        notifier = TelegramNotifier(bot_token="", chat_id="")
        result = notifier.send_message("test")
        assert result is False

    @patch("forex_bot.utils.notifications.httpx.post")
    @patch("forex_bot.utils.notifications.settings")
    def test_send_success(self, mock_settings, mock_post):
        from forex_bot.utils.notifications import TelegramNotifier

        mock_settings.TELEGRAM_ENABLED = True
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="999")
        notifier._enabled = True  # Force enable

        result = notifier.send_message("Hello!")
        assert result is True
        mock_post.assert_called_once()

        # Verificar payload
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["chat_id"] == "999"
        assert payload["text"] == "Hello!"

    @patch("forex_bot.utils.notifications.httpx.post")
    def test_send_api_error(self, mock_post):
        from forex_bot.utils.notifications import TelegramNotifier

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="999")
        notifier._enabled = True

        result = notifier.send_message("test")
        assert result is False

    @patch("forex_bot.utils.notifications.httpx.post")
    def test_send_timeout(self, mock_post):
        from forex_bot.utils.notifications import TelegramNotifier
        import httpx

        mock_post.side_effect = httpx.TimeoutException("timeout")

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="999")
        notifier._enabled = True

        result = notifier.send_message("test")
        assert result is False


class TestTelegramNotifierFormats:
    """Verificar que los mensajes formateados no crashean."""

    def _make_notifier(self):
        from forex_bot.utils.notifications import TelegramNotifier
        notifier = TelegramNotifier(bot_token="fake", chat_id="fake")
        notifier._enabled = False  # No enviar realmente
        return notifier

    def test_notify_trade_opened_format(self):
        notifier = self._make_notifier()
        trade = {
            "symbol": "EURUSD", "type": "BUY", "lot": 0.01,
            "entry_price": 1.085, "sl": 1.083, "tp": 1.089, "confidence": 0.75,
        }
        # No debe crashear (retorna False porque esta disabled)
        result = notifier.notify_trade_opened(trade)
        assert result is False

    def test_notify_trade_closed_format(self):
        notifier = self._make_notifier()
        trade = {
            "symbol": "EURUSD", "type": "BUY", "profit": 3.5,
            "pips": 30, "exit_reason": "TP", "duration_bars": 5,
        }
        result = notifier.notify_trade_closed(trade)
        assert result is False

    def test_notify_daily_summary_format(self):
        notifier = self._make_notifier()
        stats = {
            "current_balance": 105.0, "daily_pnl": 5.0,
            "daily_trades": 3, "daily_drawdown_pct": 1.5,
            "total_drawdown_pct": 3.0, "open_trades": 1,
        }
        result = notifier.notify_daily_summary(stats)
        assert result is False

    def test_notify_error_format(self):
        notifier = self._make_notifier()
        result = notifier.notify_error("Connection lost")
        assert result is False

    def test_notify_emergency_stop_format(self):
        notifier = self._make_notifier()
        result = notifier.notify_emergency_stop("Max drawdown exceeded")
        assert result is False


# =====================================================================
# TESTS: Imports del paquete
# =====================================================================

class TestUtilsImports:
    def test_import_from_package(self):
        from forex_bot.utils import TradingLogger, TelegramNotifier
        assert TradingLogger is not None
        assert TelegramNotifier is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

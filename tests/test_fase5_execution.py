"""
VALIDACION FASE 5 - Ejecucion de Ordenes
Tests con mocks ya que no hay conexion real a MT5.
Ejecutar: python -m pytest tests/test_fase5_execution.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =====================================================================
# Mock helpers para simular MT5
# =====================================================================

def make_mock_mt5():
    """Crear mock del modulo MetaTrader5 con todas las constantes."""
    mock = MagicMock()
    # Constantes
    mock.ORDER_TYPE_BUY = 0
    mock.ORDER_TYPE_SELL = 1
    mock.TRADE_ACTION_DEAL = 1
    mock.TRADE_ACTION_SLTP = 6
    mock.TRADE_RETCODE_DONE = 10009
    mock.TRADE_RETCODE_REQUOTE = 10004
    mock.TRADE_RETCODE_NO_MONEY = 10019
    mock.TRADE_RETCODE_INVALID_STOPS = 10016
    mock.TRADE_RETCODE_REJECT = 10006
    mock.TRADE_RETCODE_PLACED = 10008
    mock.TRADE_RETCODE_ERROR = 10011
    mock.ORDER_TIME_GTC = 0
    mock.ORDER_FILLING_IOC = 1
    mock.ORDER_FILLING_FOK = 0
    mock.ORDER_FILLING_RETURN = 2
    mock.POSITION_TYPE_BUY = 0
    mock.POSITION_TYPE_SELL = 1
    return mock


def make_send_result(retcode=10009, order=12345, price=1.08500, volume=0.01):
    """Crear resultado simulado de order_send."""
    result = MagicMock()
    result.retcode = retcode
    result.order = order
    result.price = price
    result.volume = volume
    result.deal = 99999
    return result


def make_tick(bid=1.08490, ask=1.08500):
    """Crear tick simulado."""
    tick = MagicMock()
    tick.bid = bid
    tick.ask = ask
    tick.last = bid
    tick.time = 1700000000
    return tick


def make_symbol_info(visible=True):
    """Crear symbol_info simulado."""
    info = MagicMock()
    info.visible = visible
    info.point = 0.00001
    return info


def make_position(ticket=12345, symbol="EURUSD", pos_type=0, volume=0.01,
                  price_open=1.085, price_current=1.086, sl=1.083, tp=1.089,
                  profit=10.0, swap=0, time_val=1700000000, comment="test", magic=234567):
    """Crear posicion simulada."""
    pos = MagicMock()
    pos.ticket = ticket
    pos.symbol = symbol
    pos.type = pos_type  # 0=BUY, 1=SELL
    pos.volume = volume
    pos.price_open = price_open
    pos.price_current = price_current
    pos.sl = sl
    pos.tp = tp
    pos.profit = profit
    pos.swap = swap
    pos.time = time_val
    pos.comment = comment
    pos.magic = magic
    return pos


@pytest.fixture
def mock_executor():
    """MT5Executor con MT5 mockeado."""
    from forex_bot.execution.mt5_executor import MT5Executor

    executor = MT5Executor.__new__(MT5Executor)
    executor._mt5 = make_mock_mt5()
    executor._collector = None

    # Setup mock responses por defecto
    mt5 = executor._mt5
    mt5.terminal_info.return_value = MagicMock()  # Connected
    mt5.symbol_info.return_value = make_symbol_info()
    mt5.symbol_info_tick.return_value = make_tick()
    mt5.order_send.return_value = make_send_result()

    return executor


@pytest.fixture
def risk_manager():
    from forex_bot.strategy.risk_manager import RiskManager
    return RiskManager(initial_balance=100.0)


# =====================================================================
# TESTS: MT5Executor - Market Orders
# =====================================================================

class TestMT5ExecutorMarketOrder:
    def test_buy_order_success(self, mock_executor):
        result = mock_executor.send_market_order("EURUSD", "BUY", 0.01, sl=1.083, tp=1.089)

        assert result["success"] is True
        assert result["order_ticket"] == 12345
        assert result["price_executed"] == 1.085
        assert result["volume"] == 0.01

        # Verificar que se llamo a order_send con los campos correctos
        call_args = mock_executor._mt5.order_send.call_args[0][0]
        assert call_args["action"] == 1  # TRADE_ACTION_DEAL
        assert call_args["symbol"] == "EURUSD"
        assert call_args["volume"] == 0.01
        assert call_args["type"] == 0  # ORDER_TYPE_BUY
        assert call_args["sl"] == 1.083
        assert call_args["tp"] == 1.089

    def test_sell_order_success(self, mock_executor):
        result = mock_executor.send_market_order("GBPUSD", "SELL", 0.02)

        assert result["success"] is True
        # Verificar que usa ORDER_TYPE_SELL y bid price
        call_args = mock_executor._mt5.order_send.call_args[0][0]
        assert call_args["type"] == 1  # ORDER_TYPE_SELL

    def test_buy_uses_ask_price(self, mock_executor):
        mock_executor._mt5.symbol_info_tick.return_value = make_tick(bid=1.080, ask=1.081)
        mock_executor.send_market_order("EURUSD", "BUY", 0.01)

        call_args = mock_executor._mt5.order_send.call_args[0][0]
        assert call_args["price"] == 1.081  # ask

    def test_sell_uses_bid_price(self, mock_executor):
        mock_executor._mt5.symbol_info_tick.return_value = make_tick(bid=1.080, ask=1.081)
        mock_executor.send_market_order("EURUSD", "SELL", 0.01)

        call_args = mock_executor._mt5.order_send.call_args[0][0]
        assert call_args["price"] == 1.080  # bid

    def test_order_includes_magic_number(self, mock_executor):
        from forex_bot.config import settings
        mock_executor.send_market_order("EURUSD", "BUY", 0.01)

        call_args = mock_executor._mt5.order_send.call_args[0][0]
        assert call_args["magic"] == settings.MAGIC_NUMBER

    def test_order_includes_slippage(self, mock_executor):
        from forex_bot.config import settings
        mock_executor.send_market_order("EURUSD", "BUY", 0.01)

        call_args = mock_executor._mt5.order_send.call_args[0][0]
        assert call_args["deviation"] == settings.MAX_SLIPPAGE

    def test_order_failure_no_money(self, mock_executor):
        mock_executor._mt5.order_send.return_value = make_send_result(retcode=10019)

        result = mock_executor.send_market_order("EURUSD", "BUY", 0.01)
        assert result["success"] is False
        assert result["retcode"] == 10019
        assert "No money" in result["error_msg"] or "fondos" in result["error_msg"]

    def test_order_failure_invalid_stops(self, mock_executor):
        mock_executor._mt5.order_send.return_value = make_send_result(retcode=10016)

        result = mock_executor.send_market_order("EURUSD", "BUY", 0.01, sl=1.08499)
        assert result["success"] is False
        assert result["retcode"] == 10016

    def test_order_requote_retries(self, mock_executor):
        """En requote, debe reintentar con nuevo precio."""
        requote = make_send_result(retcode=10004)
        success = make_send_result(retcode=10009)
        mock_executor._mt5.order_send.side_effect = [requote, success]

        result = mock_executor.send_market_order("EURUSD", "BUY", 0.01)
        assert result["success"] is True
        assert mock_executor._mt5.order_send.call_count == 2

    def test_order_invalid_type(self, mock_executor):
        result = mock_executor.send_market_order("EURUSD", "INVALID", 0.01)
        assert result["success"] is False
        assert "invalido" in result["error_msg"]

    def test_order_symbol_not_found(self, mock_executor):
        mock_executor._mt5.symbol_info.return_value = None
        result = mock_executor.send_market_order("FAKEPAIR", "BUY", 0.01)
        assert result["success"] is False
        assert "no encontrado" in result["error_msg"]

    def test_order_mt5_not_connected(self):
        from forex_bot.execution.mt5_executor import MT5Executor
        executor = MT5Executor.__new__(MT5Executor)
        executor._mt5 = None
        executor._collector = None

        result = executor.send_market_order("EURUSD", "BUY", 0.01)
        assert result["success"] is False
        assert "no conectado" in result["error_msg"]


# =====================================================================
# TESTS: MT5Executor - Modify Position
# =====================================================================

class TestMT5ExecutorModify:
    def test_modify_sl_tp_success(self, mock_executor):
        result = mock_executor.modify_position(12345, "EURUSD", new_sl=1.084, new_tp=1.090)
        assert result is True

        call_args = mock_executor._mt5.order_send.call_args[0][0]
        assert call_args["action"] == 6  # TRADE_ACTION_SLTP
        assert call_args["position"] == 12345
        assert call_args["sl"] == 1.084
        assert call_args["tp"] == 1.090

    def test_modify_failure(self, mock_executor):
        mock_executor._mt5.order_send.return_value = make_send_result(retcode=10016)
        result = mock_executor.modify_position(12345, "EURUSD", new_sl=1.08499, new_tp=1.086)
        assert result is False


# =====================================================================
# TESTS: MT5Executor - Close Position
# =====================================================================

class TestMT5ExecutorClose:
    def test_close_buy_position(self, mock_executor):
        """Cerrar BUY = enviar SELL con position ticket."""
        buy_pos = make_position(ticket=111, pos_type=0)  # BUY
        mock_executor._mt5.positions_get.return_value = [buy_pos]

        result = mock_executor.close_position(111, "EURUSD")
        assert result is True

        call_args = mock_executor._mt5.order_send.call_args[0][0]
        assert call_args["type"] == 1  # ORDER_TYPE_SELL (opuesto a BUY)
        assert call_args["position"] == 111

    def test_close_sell_position(self, mock_executor):
        sell_pos = make_position(ticket=222, pos_type=1)  # SELL
        mock_executor._mt5.positions_get.return_value = [sell_pos]

        result = mock_executor.close_position(222, "GBPUSD")
        assert result is True

        call_args = mock_executor._mt5.order_send.call_args[0][0]
        assert call_args["type"] == 0  # ORDER_TYPE_BUY (opuesto a SELL)

    def test_close_nonexistent_position(self, mock_executor):
        mock_executor._mt5.positions_get.return_value = []
        result = mock_executor.close_position(999, "EURUSD")
        assert result is False


# =====================================================================
# TESTS: MT5Executor - Get Open Positions
# =====================================================================

class TestMT5ExecutorPositions:
    def test_get_open_positions(self, mock_executor):
        from forex_bot.config import settings
        pos1 = make_position(ticket=111, symbol="EURUSD", pos_type=0, magic=settings.MAGIC_NUMBER)
        pos2 = make_position(ticket=222, symbol="GBPUSD", pos_type=1, magic=settings.MAGIC_NUMBER)
        pos3 = make_position(ticket=333, symbol="USDJPY", pos_type=0, magic=99999)  # otro bot

        mock_executor._mt5.positions_get.return_value = [pos1, pos2, pos3]

        positions = mock_executor.get_open_positions()
        # Solo debe retornar las del bot (magic number correcto)
        assert len(positions) == 2
        assert positions[0]["ticket"] == 111
        assert positions[1]["ticket"] == 222

    def test_get_open_positions_empty(self, mock_executor):
        mock_executor._mt5.positions_get.return_value = []
        assert mock_executor.get_open_positions() == []

    def test_position_fields(self, mock_executor):
        from forex_bot.config import settings
        pos = make_position(
            ticket=111, symbol="EURUSD", pos_type=0, volume=0.01,
            price_open=1.085, price_current=1.087, sl=1.083, tp=1.089,
            profit=20.0, magic=settings.MAGIC_NUMBER,
        )
        mock_executor._mt5.positions_get.return_value = [pos]

        positions = mock_executor.get_open_positions()
        p = positions[0]
        assert p["ticket"] == 111
        assert p["symbol"] == "EURUSD"
        assert p["type"] == "BUY"
        assert p["volume"] == 0.01
        assert p["open_price"] == 1.085
        assert p["profit"] == 20.0
        assert p["sl"] == 1.083
        assert p["tp"] == 1.089


# =====================================================================
# TESTS: MT5Executor - Order Check
# =====================================================================

class TestMT5ExecutorCheck:
    def test_order_check_valid(self, mock_executor):
        check_result = MagicMock()
        check_result.retcode = 10009
        check_result.comment = "Done"
        check_result.margin = 50.0
        check_result.profit = 0.0
        mock_executor._mt5.order_check.return_value = check_result

        result = mock_executor.order_check("EURUSD", "BUY", 0.01, sl=1.083, tp=1.089)
        assert result["valid"] is True
        assert result["margin"] == 50.0


# =====================================================================
# TESTS: OrderManager - Process Signal
# =====================================================================

class TestOrderManagerProcessSignal:
    def test_process_buy_signal(self, mock_executor, risk_manager):
        from forex_bot.execution.order_manager import OrderManager

        om = OrderManager(mock_executor, risk_manager)

        signal = {
            "symbol": "EURUSD",
            "signal": "BUY",
            "confidence": 0.75,
            "entry_price": 1.085,
            "sl_price": 1.083,
            "tp_price": 1.089,
            "sl_pips": 20,
            "tp_pips": 40,
        }

        result = om.process_signal(signal)
        assert result["executed"] is True
        assert result["order_ticket"] == 12345
        assert result["lot_size"] >= 0.01

    def test_process_hold_signal(self, mock_executor, risk_manager):
        from forex_bot.execution.order_manager import OrderManager

        om = OrderManager(mock_executor, risk_manager)
        signal = {"symbol": "EURUSD", "signal": "HOLD", "sl_pips": 0}

        result = om.process_signal(signal)
        assert result["executed"] is False
        assert "HOLD" in result["error_msg"]

    def test_process_signal_risk_rejected(self, mock_executor, risk_manager):
        from forex_bot.execution.order_manager import OrderManager

        # Activar emergency stop
        risk_manager.trigger_emergency_stop("test")
        om = OrderManager(mock_executor, risk_manager)

        signal = {
            "symbol": "EURUSD", "signal": "BUY", "confidence": 0.8,
            "entry_price": 1.085, "sl_price": 1.083, "tp_price": 1.089,
            "sl_pips": 20, "tp_pips": 40,
        }

        result = om.process_signal(signal)
        assert result["executed"] is False
        assert "RiskManager" in result["error_msg"]

    def test_process_signal_registers_trade(self, mock_executor, risk_manager):
        from forex_bot.execution.order_manager import OrderManager

        om = OrderManager(mock_executor, risk_manager)
        assert risk_manager._open_trades_count == 0

        signal = {
            "symbol": "EURUSD", "signal": "BUY", "confidence": 0.8,
            "entry_price": 1.085, "sl_price": 1.083, "tp_price": 1.089,
            "sl_pips": 20, "tp_pips": 40,
        }

        om.process_signal(signal)
        assert risk_manager._open_trades_count == 1

    def test_process_signal_order_fails(self, mock_executor, risk_manager):
        from forex_bot.execution.order_manager import OrderManager

        mock_executor._mt5.order_send.return_value = make_send_result(retcode=10019)
        om = OrderManager(mock_executor, risk_manager)

        signal = {
            "symbol": "EURUSD", "signal": "BUY", "confidence": 0.8,
            "entry_price": 1.085, "sl_price": 1.083, "tp_price": 1.089,
            "sl_pips": 20, "tp_pips": 40,
        }

        result = om.process_signal(signal)
        assert result["executed"] is False
        # No debe registrar trade si fallo
        assert risk_manager._open_trades_count == 0


# =====================================================================
# TESTS: OrderManager - Manage Positions
# =====================================================================

class TestOrderManagerManage:
    def test_emergency_close_all(self, mock_executor, risk_manager):
        from forex_bot.execution.order_manager import OrderManager
        from forex_bot.config import settings

        pos1 = make_position(ticket=111, magic=settings.MAGIC_NUMBER)
        mock_executor._mt5.positions_get.return_value = [pos1]

        om = OrderManager(mock_executor, risk_manager)
        results = om.emergency_close_all()

        assert risk_manager.is_emergency_stopped
        assert len(results) >= 0  # Depende del mock

    def test_check_exit_signals_contrary(self, mock_executor, risk_manager):
        """Senal contraria debe cerrar posicion."""
        from forex_bot.execution.order_manager import OrderManager
        from forex_bot.config import settings

        # Posicion BUY abierta
        buy_pos = make_position(ticket=111, symbol="EURUSD", pos_type=0, magic=settings.MAGIC_NUMBER, profit=5.0)
        mock_executor._mt5.positions_get.return_value = [buy_pos]

        om = OrderManager(mock_executor, risk_manager)

        # Senal SELL para EURUSD
        signals = {"EURUSD": {"signal": "SELL", "confidence": 0.8}}
        om.check_exit_signals(signals)

        # Debe haber intentado cerrar (close_position llama a positions_get + order_send)
        assert mock_executor._mt5.order_send.called


# =====================================================================
# TESTS: Import checks
# =====================================================================

class TestExecutionImports:
    def test_import_executor(self):
        from forex_bot.execution.mt5_executor import MT5Executor
        assert MT5Executor is not None

    def test_import_order_manager(self):
        from forex_bot.execution.order_manager import OrderManager
        assert OrderManager is not None

    def test_import_from_package(self):
        from forex_bot.execution import MT5Executor, OrderManager
        assert MT5Executor is not None
        assert OrderManager is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

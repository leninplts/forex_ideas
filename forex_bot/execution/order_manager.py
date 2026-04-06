"""
Order Manager - Orquesta senales, risk management y ejecucion.
Coordina SignalGenerator -> RiskManager -> MT5Executor.
Gestiona trailing stops y senales de salida.
"""
import logging
from typing import Optional

import pandas_ta as ta

from forex_bot.config import settings
from forex_bot.execution.mt5_executor import MT5Executor
from forex_bot.strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Orquestador de ordenes.
    Recibe senales, verifica riesgo, ejecuta ordenes y gestiona posiciones abiertas.
    """

    def __init__(self, executor: MT5Executor, risk_manager: RiskManager):
        """
        Args:
            executor: MT5Executor para enviar ordenes
            risk_manager: RiskManager para verificar riesgo y position sizing
        """
        self.executor = executor
        self.risk_manager = risk_manager

        # Cache de posiciones abiertas por el bot
        self._open_positions = {}  # {ticket: position_info}

    # ------------------------------------------------------------------
    # Procesar senal
    # ------------------------------------------------------------------

    def process_signal(self, signal: dict) -> dict:
        """
        Procesar una senal del SignalGenerator y ejecutar si pasa todas las validaciones.

        Pipeline:
        1. Verificar que la senal es operable (no HOLD)
        2. Verificar con RiskManager (drawdown, max trades)
        3. Calcular position size
        4. Enviar orden via MT5Executor
        5. Registrar la operacion

        Args:
            signal: Dict del SignalGenerator con: symbol, signal, confidence,
                    entry_price, sl_price, tp_price, sl_pips, tp_pips

        Returns:
            Dict con resultado: executed, order_ticket, lot_size, error_msg
        """
        result = {
            "executed": False,
            "order_ticket": 0,
            "lot_size": 0.0,
            "error_msg": "",
        }

        symbol = signal.get("symbol", "")
        direction = signal.get("signal", "HOLD")
        sl_price = signal.get("sl_price", 0)
        tp_price = signal.get("tp_price", 0)
        sl_pips = signal.get("sl_pips", 0)

        # 1. Verificar senal operable
        if direction == "HOLD":
            result["error_msg"] = "Senal es HOLD, no operar"
            return result

        if sl_pips <= 0:
            result["error_msg"] = "SL en pips invalido"
            return result

        # 2. Verificar riesgo
        # Obtener margen libre si MT5 esta conectado
        free_margin = None
        account = None
        try:
            if self.executor._is_ready():
                mt5 = self.executor._mt5
                account = mt5.account_info()
                if account is not None:
                    margin_val = account.margin_free
                    # Verificar que es un numero real (no un mock u otro tipo)
                    if isinstance(margin_val, (int, float)):
                        free_margin = float(margin_val)
        except Exception:
            pass  # Si falla, free_margin queda None y no se verifica margen

        can_trade, reason = self.risk_manager.can_open_trade(free_margin)

        if not can_trade:
            result["error_msg"] = f"RiskManager rechazo: {reason}"
            logger.info("Trade rechazado por risk manager: %s", reason)
            return result

        # 3. Calcular position size
        balance = self.risk_manager.current_balance
        try:
            if account is not None and isinstance(account.balance, (int, float)):
                balance = float(account.balance)
        except Exception:
            pass
        lot_size = self.risk_manager.calculate_position_size(sl_pips, symbol, balance)
        result["lot_size"] = lot_size

        # 4. Enviar orden
        logger.info(
            "Enviando orden: %s %s %.2f lots | SL: %.5f | TP: %.5f",
            direction, symbol, lot_size, sl_price, tp_price,
        )

        order_result = self.executor.send_market_order(
            symbol=symbol,
            order_type=direction,
            lot=lot_size,
            sl=sl_price,
            tp=tp_price,
            comment=f"ForexBot {direction} conf={signal.get('confidence', 0):.0%}",
        )

        if order_result["success"]:
            result["executed"] = True
            result["order_ticket"] = order_result["order_ticket"]

            # 5. Registrar con risk manager
            self.risk_manager.register_trade_opened()

            # Guardar en cache
            self._open_positions[order_result["order_ticket"]] = {
                "ticket": order_result["order_ticket"],
                "symbol": symbol,
                "type": direction,
                "lot": lot_size,
                "entry_price": order_result["price_executed"],
                "sl": sl_price,
                "tp": tp_price,
                "sl_pips": sl_pips,
            }

            logger.info(
                "TRADE ABIERTO: %s %s %.2f lots @ %.5f | Ticket: %d",
                direction, symbol, lot_size,
                order_result["price_executed"], order_result["order_ticket"],
            )
        else:
            result["error_msg"] = f"Orden fallida: {order_result['error_msg']}"
            logger.error("Orden fallida: %s", order_result["error_msg"])

        return result

    # ------------------------------------------------------------------
    # Gestion de posiciones abiertas
    # ------------------------------------------------------------------

    def manage_open_positions(self, current_data: dict = None):
        """
        Revisar y gestionar posiciones abiertas.
        Actualizar trailing stops y verificar senales de salida.

        Args:
            current_data: Dict {symbol: DataFrame} con datos actuales para ATR
        """
        positions = self.executor.get_open_positions()

        if not positions:
            self._open_positions = {}
            return

        for pos in positions:
            ticket = pos["ticket"]
            symbol = pos["symbol"]

            # Actualizar cache
            self._open_positions[ticket] = pos

            # Trailing stop
            if settings.TRAILING_STOP_ENABLED and current_data:
                df = current_data.get(symbol)
                if df is not None:
                    self._update_trailing_stop(pos, df)

        # Limpiar cache: quitar posiciones que ya no existen
        active_tickets = {p["ticket"] for p in positions}
        closed = [t for t in self._open_positions if t not in active_tickets]
        for ticket in closed:
            del self._open_positions[ticket]

    def _update_trailing_stop(self, position: dict, df):
        """
        Actualizar trailing stop de una posicion.

        Reglas:
        - BUY: solo subir SL, nunca bajarlo
        - SELL: solo bajar SL, nunca subirlo
        - Nuevo SL basado en precio actual - ATR * multiplier

        Args:
            position: Dict con info de la posicion
            df: DataFrame con datos OHLCV para calcular ATR
        """
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=settings.ATR_PERIOD)
        if atr_series is None or atr_series.empty:
            return

        current_atr = atr_series.iloc[-1]
        if current_atr <= 0:
            return

        trailing_distance = current_atr * settings.TRAILING_STOP_ATR_MULT
        current_price = position["current_price"]
        current_sl = position["sl"]
        ticket = position["ticket"]
        symbol = position["symbol"]

        if position["type"] == "BUY":
            new_sl = current_price - trailing_distance
            # Solo mover SL hacia arriba
            if new_sl > current_sl and current_sl > 0:
                success = self.executor.modify_position(
                    ticket, symbol, new_sl, position["tp"],
                )
                if success:
                    logger.info(
                        "Trailing stop BUY %d: SL %.5f -> %.5f (precio: %.5f)",
                        ticket, current_sl, new_sl, current_price,
                    )

        elif position["type"] == "SELL":
            new_sl = current_price + trailing_distance
            # Solo mover SL hacia abajo
            if new_sl < current_sl or current_sl == 0:
                success = self.executor.modify_position(
                    ticket, symbol, new_sl, position["tp"],
                )
                if success:
                    logger.info(
                        "Trailing stop SELL %d: SL %.5f -> %.5f (precio: %.5f)",
                        ticket, current_sl, new_sl, current_price,
                    )

    # ------------------------------------------------------------------
    # Senales de salida
    # ------------------------------------------------------------------

    def check_exit_signals(self, current_signals: dict):
        """
        Verificar si hay senales contrarias a posiciones abiertas.
        Si hay senal contraria, cerrar la posicion.

        Args:
            current_signals: Dict {symbol: signal_dict} con senales actuales
        """
        positions = self.executor.get_open_positions()

        for pos in positions:
            symbol = pos["symbol"]
            signal = current_signals.get(symbol)

            if signal is None:
                continue

            direction = signal.get("signal", "HOLD")

            # Senal contraria -> cerrar
            if pos["type"] == "BUY" and direction == "SELL":
                logger.info(
                    "Senal contraria para %s: posicion BUY, senal SELL. Cerrando ticket %d",
                    symbol, pos["ticket"],
                )
                self.executor.close_position(pos["ticket"], symbol)
                self.risk_manager.register_trade_closed(pos["profit"])

            elif pos["type"] == "SELL" and direction == "BUY":
                logger.info(
                    "Senal contraria para %s: posicion SELL, senal BUY. Cerrando ticket %d",
                    symbol, pos["ticket"],
                )
                self.executor.close_position(pos["ticket"], symbol)
                self.risk_manager.register_trade_closed(pos["profit"])

    # ------------------------------------------------------------------
    # Emergency
    # ------------------------------------------------------------------

    def emergency_close_all(self) -> list:
        """
        Cerrar TODAS las posiciones inmediatamente.
        Se usa cuando se excede el drawdown maximo.

        Returns:
            Lista de resultados de cierre.
        """
        logger.critical("EMERGENCY: Cerrando todas las posiciones")
        results = self.executor.close_all_positions()

        # Registrar cierres en risk manager
        for r in results:
            if r["closed"]:
                # No sabemos el profit exacto aqui, se sincronizara despues
                pass

        self.risk_manager.trigger_emergency_stop("Emergency close all")
        return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def open_positions_count(self) -> int:
        return len(self._open_positions)

    def get_positions_summary(self) -> list:
        """Resumen de posiciones abiertas del cache."""
        return list(self._open_positions.values())

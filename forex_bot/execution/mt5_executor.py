"""
MT5 Executor - Envio de ordenes a MetaTrader 5.
Maneja market orders, modificaciones de SL/TP, cierre de posiciones.

API verificada contra documentacion oficial de MQL5:
  https://www.mql5.com/en/docs/python_metatrader5/mt5ordersend_py
  - order_send(request): enviar orden con dict de campos
  - TRADE_ACTION_DEAL: orden de mercado
  - TRADE_ACTION_SLTP: modificar SL/TP
  - TRADE_RETCODE_DONE (10009): ejecucion exitosa
  - positions_get(): obtener posiciones abiertas
  - history_deals_get(): historial de operaciones
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from forex_bot.config import settings

logger = logging.getLogger(__name__)


class MT5Executor:
    """
    Ejecutor de ordenes de trading via MetaTrader 5.
    Encapsula toda la interaccion con la API de MT5 para ordenes.
    """

    def __init__(self, collector=None):
        """
        Args:
            collector: DataCollector con conexion a MT5 (opcional).
                       Si None, se importa MT5 directamente.
        """
        self._mt5 = None
        self._collector = collector
        self._import_mt5()

    def _import_mt5(self):
        """
        Importar modulo MetaTrader5.
        Si hay un collector conectado, reutilizar su modulo MT5 (funciona en Docker y local).
        Si no, importar directamente.
        """
        # Primero: reutilizar el modulo del collector (ya maneja Docker vs local)
        if self._collector and self._collector._mt5 is not None:
            self._mt5 = self._collector._mt5
            return

        # Fallback: importar segun el entorno
        import os
        if os.environ.get("MT5_HOST"):
            try:
                from mt5linux import MetaTrader5
                host = os.environ.get("MT5_HOST", "localhost")
                port = int(os.environ.get("MT5_PORT", "8001"))
                self._mt5 = MetaTrader5(host=host, port=port)
                self._mt5.initialize()
            except ImportError:
                logger.error("mt5linux no instalado")
                self._mt5 = None
        else:
            try:
                import MetaTrader5 as mt5
                self._mt5 = mt5
            except ImportError:
                logger.error("MetaTrader5 no instalado")
                self._mt5 = None

    def _is_ready(self) -> bool:
        """Verificar que MT5 esta disponible y conectado."""
        if self._mt5 is None:
            return False
        if self._collector:
            return self._collector.is_connected
        # Si no hay collector, verificar terminal directamente
        info = self._mt5.terminal_info()
        return info is not None

    # ------------------------------------------------------------------
    # Market Orders
    # ------------------------------------------------------------------

    def send_market_order(
        self,
        symbol: str,
        order_type: str,
        lot: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "",
    ) -> dict:
        """
        Enviar orden de mercado (BUY o SELL) a MT5.

        Formato del request segun documentacion oficial de MQL5:
        {
            "action": TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": ORDER_TYPE_BUY / ORDER_TYPE_SELL,
            "price": ask (BUY) o bid (SELL),
            "sl": stop_loss_price,
            "tp": take_profit_price,
            "deviation": max_slippage,
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_time": ORDER_TIME_GTC,
            "type_filling": ORDER_FILLING_IOC,
        }

        Args:
            symbol: Par de divisas (ej: "EURUSD")
            order_type: "BUY" o "SELL"
            lot: Tamano de la posicion en lotes
            sl: Precio de Stop Loss (0 = sin SL)
            tp: Precio de Take Profit (0 = sin TP)
            comment: Comentario de la orden

        Returns:
            Dict con: success, order_ticket, price_executed, volume, error_msg, retcode
        """
        result = {
            "success": False,
            "order_ticket": 0,
            "price_executed": 0.0,
            "volume": 0.0,
            "error_msg": "",
            "retcode": 0,
        }

        if not self._is_ready():
            result["error_msg"] = "MT5 no conectado"
            return result

        mt5 = self._mt5

        # Verificar simbolo
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            result["error_msg"] = f"Simbolo no encontrado: {symbol}"
            return result
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                result["error_msg"] = f"No se pudo activar simbolo: {symbol}"
                return result

        # Obtener precio actual
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            result["error_msg"] = f"No se pudo obtener precio de {symbol}"
            return result

        # Determinar tipo de orden y precio
        if order_type.upper() == "BUY":
            mt5_order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif order_type.upper() == "SELL":
            mt5_order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            result["error_msg"] = f"Tipo de orden invalido: {order_type}"
            return result

        # Construir request segun API oficial de MT5
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": mt5_order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": settings.MAX_SLIPPAGE,
            "magic": settings.MAGIC_NUMBER,
            "comment": comment or f"ForexBot {order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Intentar enviar con reintentos
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            send_result = mt5.order_send(request)

            if send_result is None:
                error = mt5.last_error()
                result["error_msg"] = f"order_send retorno None: {error}"
                logger.error("order_send() retorno None (intento %d): %s", attempt, error)
                time.sleep(0.5)
                continue

            result["retcode"] = send_result.retcode

            if send_result.retcode == mt5.TRADE_RETCODE_DONE:
                # Orden ejecutada exitosamente
                result["success"] = True
                result["order_ticket"] = send_result.order
                result["price_executed"] = send_result.price
                result["volume"] = send_result.volume

                logger.info(
                    "ORDEN EJECUTADA: %s %s %.2f lots @ %.5f | Ticket: %d | SL: %.5f | TP: %.5f",
                    order_type, symbol, lot, send_result.price,
                    send_result.order, sl, tp,
                )
                return result

            elif send_result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                # Requote: actualizar precio y reintentar
                logger.warning("Requote en %s, reintentando (intento %d)...", symbol, attempt)
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    price = tick.ask if order_type.upper() == "BUY" else tick.bid
                    request["price"] = price
                time.sleep(0.2)
                continue

            else:
                # Otros errores
                error_msg = self._get_retcode_message(send_result.retcode)
                result["error_msg"] = f"retcode={send_result.retcode}: {error_msg}"
                logger.error(
                    "ORDEN FALLIDA: %s %s | retcode=%d: %s",
                    order_type, symbol, send_result.retcode, error_msg,
                )
                # No reintentar para errores no recuperables
                if send_result.retcode in (
                    mt5.TRADE_RETCODE_NO_MONEY,
                    mt5.TRADE_RETCODE_INVALID_STOPS,
                    mt5.TRADE_RETCODE_REJECT,
                ):
                    return result
                time.sleep(0.5)

        return result

    # ------------------------------------------------------------------
    # Modify Orders (SL/TP)
    # ------------------------------------------------------------------

    def modify_position(self, ticket: int, symbol: str, new_sl: float, new_tp: float) -> bool:
        """
        Modificar SL/TP de una posicion abierta.
        Usa TRADE_ACTION_SLTP segun API oficial de MT5.

        Args:
            ticket: Ticket de la posicion
            symbol: Simbolo del par
            new_sl: Nuevo precio de Stop Loss
            new_tp: Nuevo precio de Take Profit

        Returns:
            True si la modificacion fue exitosa.
        """
        if not self._is_ready():
            logger.error("MT5 no conectado para modificar posicion")
            return False

        mt5 = self._mt5

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": float(new_sl),
            "tp": float(new_tp),
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error("modify_position retorno None: %s", mt5.last_error())
            return False

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                "Posicion %d modificada: SL=%.5f, TP=%.5f",
                ticket, new_sl, new_tp,
            )
            return True
        else:
            logger.error(
                "Error modificando posicion %d: retcode=%d (%s)",
                ticket, result.retcode, self._get_retcode_message(result.retcode),
            )
            return False

    # ------------------------------------------------------------------
    # Close Orders
    # ------------------------------------------------------------------

    def close_position(self, ticket: int, symbol: str, lot: float = None) -> bool:
        """
        Cerrar una posicion abierta.
        Para cerrar, se envia una orden opuesta con TRADE_ACTION_DEAL
        y el campo "position" con el ticket.

        Args:
            ticket: Ticket de la posicion a cerrar
            symbol: Simbolo del par
            lot: Volumen a cerrar (None = cerrar todo)

        Returns:
            True si se cerro exitosamente.
        """
        if not self._is_ready():
            logger.error("MT5 no conectado para cerrar posicion")
            return False

        mt5 = self._mt5

        # Obtener info de la posicion para saber tipo y volumen
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            logger.error("Posicion %d no encontrada", ticket)
            return False

        position = positions[0]
        pos_type = position.type  # 0=BUY, 1=SELL
        volume = lot if lot else position.volume

        # Tipo opuesto para cerrar
        if pos_type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if tick else 0
        else:
            close_type = mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if tick else 0

        if price == 0:
            logger.error("No se pudo obtener precio para cerrar %s", symbol)
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": settings.MAX_SLIPPAGE,
            "magic": settings.MAGIC_NUMBER,
            "comment": "ForexBot close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error("close_position retorno None: %s", mt5.last_error())
            return False

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                "POSICION CERRADA: ticket=%d, %s %.2f lots @ %.5f",
                ticket, symbol, volume, result.price,
            )
            return True
        else:
            logger.error(
                "Error cerrando posicion %d: retcode=%d (%s)",
                ticket, result.retcode, self._get_retcode_message(result.retcode),
            )
            return False

    def close_all_positions(self, symbol: str = None) -> list:
        """
        Cerrar todas las posiciones abiertas del bot.

        Args:
            symbol: Si se especifica, solo cerrar posiciones de ese par.

        Returns:
            Lista de resultados (True/False) por cada posicion.
        """
        positions = self.get_open_positions()
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]

        results = []
        for pos in positions:
            success = self.close_position(pos["ticket"], pos["symbol"])
            results.append({"ticket": pos["ticket"], "symbol": pos["symbol"], "closed": success})

        logger.info(
            "Cerradas %d/%d posiciones",
            sum(1 for r in results if r["closed"]), len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Query Positions
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list:
        """
        Obtener posiciones abiertas del bot (filtradas por MAGIC_NUMBER).

        Returns:
            Lista de dicts con info de cada posicion:
            ticket, symbol, type, volume, open_price, sl, tp, profit, time, comment
        """
        if not self._is_ready():
            return []

        mt5 = self._mt5
        positions = mt5.positions_get()

        if positions is None or len(positions) == 0:
            return []

        result = []
        for pos in positions:
            # Filtrar solo posiciones del bot
            if pos.magic != settings.MAGIC_NUMBER:
                continue

            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "open_price": pos.price_open,
                "current_price": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "swap": pos.swap,
                "time": datetime.fromtimestamp(pos.time),
                "comment": pos.comment,
                "magic": pos.magic,
            })

        return result

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_order_history(self, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Obtener historial de operaciones cerradas de los ultimos N dias.

        Args:
            days: Numero de dias hacia atras

        Returns:
            DataFrame con historial o None si no hay datos.
        """
        if not self._is_ready():
            return None

        mt5 = self._mt5
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()

        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None or len(deals) == 0:
            return None

        # Filtrar solo deals del bot
        records = []
        for deal in deals:
            if deal.magic != settings.MAGIC_NUMBER:
                continue
            records.append({
                "ticket": deal.ticket,
                "order": deal.order,
                "time": datetime.fromtimestamp(deal.time),
                "symbol": deal.symbol,
                "type": "BUY" if deal.type == 0 else "SELL",
                "volume": deal.volume,
                "price": deal.price,
                "profit": deal.profit,
                "commission": deal.commission,
                "swap": deal.swap,
                "comment": deal.comment,
            })

        if not records:
            return None

        df = pd.DataFrame(records)
        df.set_index("time", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_retcode_message(self, retcode: int) -> str:
        """Traducir retcode a mensaje legible."""
        if self._mt5 is None:
            return f"Unknown ({retcode})"

        mt5 = self._mt5
        messages = {
            mt5.TRADE_RETCODE_DONE: "Ejecutada correctamente",
            mt5.TRADE_RETCODE_REQUOTE: "Requote - precio cambio",
            mt5.TRADE_RETCODE_REJECT: "Rechazada por el servidor",
            mt5.TRADE_RETCODE_NO_MONEY: "Sin fondos suficientes",
            mt5.TRADE_RETCODE_INVALID_STOPS: "SL/TP invalidos (muy cerca del precio)",
            mt5.TRADE_RETCODE_PLACED: "Orden colocada",
            mt5.TRADE_RETCODE_ERROR: "Error general",
            10010: "Solo parte de la orden fue ejecutada",
            10013: "Request invalido",
            10014: "Volumen invalido",
            10015: "Precio invalido",
            10017: "Trading deshabilitado",
            10018: "Mercado cerrado",
        }
        return messages.get(retcode, f"Codigo desconocido ({retcode})")

    def order_check(self, symbol: str, order_type: str, lot: float, sl: float = 0, tp: float = 0) -> dict:
        """
        Verificar una orden sin enviarla (pre-validacion).
        Usa mt5.order_check() para verificar margen, stops, etc.

        Returns:
            Dict con resultado de la verificacion.
        """
        if not self._is_ready():
            return {"valid": False, "error": "MT5 no conectado"}

        mt5 = self._mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"valid": False, "error": f"No se pudo obtener precio de {symbol}"}

        if order_type.upper() == "BUY":
            mt5_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            mt5_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": mt5_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": settings.MAX_SLIPPAGE,
            "magic": settings.MAGIC_NUMBER,
            "comment": "ForexBot check",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        check_result = mt5.order_check(request)
        if check_result is None:
            return {"valid": False, "error": str(mt5.last_error())}

        return {
            "valid": check_result.retcode == mt5.TRADE_RETCODE_DONE,
            "retcode": check_result.retcode,
            "comment": check_result.comment,
            "margin": check_result.margin,
            "profit": check_result.profit,
        }

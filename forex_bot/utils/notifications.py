"""
Telegram Notifier - Envio de alertas via Telegram Bot API.
Usa httpx (HTTP directo) en vez de python-telegram-bot async para
compatibilidad con el loop sincrono del bot.

Soporta comandos interactivos:
  /balance - consultar balance y estado actual
  /status  - ver posiciones abiertas

API de Telegram: https://core.telegram.org/bots/api#sendmessage
"""
import logging
import threading
import time
from typing import Optional, Callable

import httpx

from forex_bot.config import settings

logger = logging.getLogger(__name__)

# URLs de la API de Telegram
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_UPDATES_URL = "https://api.telegram.org/bot{token}/getUpdates"


class TelegramNotifier:
    """
    Notificador de Telegram para alertas de trading.
    Envia mensajes via HTTP POST a la API de Telegram Bot.
    Es completamente opcional - el bot funciona sin Telegram.
    """

    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Args:
            bot_token: Token del bot de Telegram (de @BotFather)
            chat_id: ID del chat donde enviar mensajes
        """
        import os

        # Prioridad: argumento explicito > env var > mt5_config.py
        # Si se paso argumento (incluso vacio), respetar esa decision
        if bot_token is None:
            bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            # Fallback: mt5_config.py (modo local Windows)
            if not bot_token:
                try:
                    from forex_bot.config import mt5_config
                    bot_token = getattr(mt5_config, "TELEGRAM_BOT_TOKEN", "")
                except ImportError:
                    pass

        if chat_id is None:
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if not chat_id:
                try:
                    from forex_bot.config import mt5_config
                    chat_id = getattr(mt5_config, "TELEGRAM_CHAT_ID", "")
                except ImportError:
                    pass

        self.bot_token = bot_token or ""
        self.chat_id = str(chat_id) if chat_id else ""
        self._enabled = bool(self.bot_token and self.chat_id and settings.TELEGRAM_ENABLED)

        # Callbacks para comandos interactivos (se registran desde main.py)
        self._command_handlers = {}
        self._polling_thread = None
        self._polling_active = False
        self._last_update_id = 0

        if self._enabled:
            logger.info("Telegram notificaciones habilitadas (chat_id: %s)", self.chat_id)
        else:
            logger.info("Telegram notificaciones deshabilitadas")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Comandos interactivos (polling)
    # ------------------------------------------------------------------

    def register_command(self, command: str, handler: Callable):
        """
        Registrar un handler para un comando de Telegram.
        El handler recibe 0 argumentos y debe retornar un string con la respuesta.

        Args:
            command: Comando sin / (ej: "balance", "status")
            handler: Funcion que retorna str con la respuesta
        """
        self._command_handlers[command.lower()] = handler
        logger.info("Comando Telegram registrado: /%s", command)

    def start_polling(self):
        """Iniciar thread de polling para escuchar comandos."""
        if not self._enabled or not self._command_handlers:
            return

        self._polling_active = True
        self._polling_thread = threading.Thread(
            target=self._polling_loop,
            daemon=True,
            name="telegram-polling",
        )
        self._polling_thread.start()
        logger.info("Telegram polling iniciado (comandos: %s)",
                     ", ".join(f"/{c}" for c in self._command_handlers))

    def stop_polling(self):
        """Detener thread de polling."""
        self._polling_active = False
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=5)

    def _polling_loop(self):
        """Loop de polling que escucha mensajes/comandos de Telegram."""
        url = TELEGRAM_UPDATES_URL.format(token=self.bot_token)

        while self._polling_active:
            try:
                params = {
                    "offset": self._last_update_id + 1,
                    "timeout": 10,
                    "allowed_updates": '["message"]',
                }
                response = httpx.get(url, params=params, timeout=15)

                if response.status_code != 200:
                    time.sleep(5)
                    continue

                data = response.json()
                if not data.get("ok"):
                    time.sleep(5)
                    continue

                for update in data.get("result", []):
                    self._last_update_id = update["update_id"]
                    self._process_update(update)

            except httpx.TimeoutException:
                continue
            except Exception as e:
                logger.error("Error en Telegram polling: %s", e)
                time.sleep(10)

    def _process_update(self, update: dict):
        """Procesar un update de Telegram y responder si es un comando conocido."""
        message = update.get("message", {})
        text = message.get("text", "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))

        # Solo responder a nuestro chat_id
        if chat_id != self.chat_id:
            return

        # Verificar si es un comando
        if not text.startswith("/"):
            return

        command = text.split()[0].lstrip("/").lower().split("@")[0]  # /balance@botname -> balance

        handler = self._command_handlers.get(command)
        if handler:
            try:
                response_text = handler()
                if response_text:
                    self.send_message(response_text)
            except Exception as e:
                logger.error("Error ejecutando comando /%s: %s", command, e)
                self.send_message(f"Error ejecutando /{command}: {str(e)[:200]}")
        else:
            available = ", ".join(f"/{c}" for c in self._command_handlers)
            self.send_message(f"Comando desconocido: {text}\nDisponibles: {available}")

    # ------------------------------------------------------------------
    # Enviar mensaje
    # ------------------------------------------------------------------

    def send_message(self, text: str) -> bool:
        """
        Enviar mensaje de texto a Telegram.

        Args:
            text: Texto del mensaje (soporta HTML basico)

        Returns:
            True si se envio correctamente, False si fallo.
        """
        if not self._enabled:
            return False

        url = TELEGRAM_API_URL.format(token=self.bot_token)
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            response = httpx.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                logger.error(
                    "Telegram API error: status=%d, body=%s",
                    response.status_code, response.text[:200],
                )
                return False
        except httpx.TimeoutException:
            logger.error("Telegram timeout al enviar mensaje")
            return False
        except Exception as e:
            logger.error("Error enviando mensaje Telegram: %s", e)
            return False

    # ------------------------------------------------------------------
    # Notificaciones especificas
    # ------------------------------------------------------------------

    def notify_trade_opened(self, trade_info: dict) -> bool:
        """
        Notificar apertura de trade.

        Args:
            trade_info: Dict con symbol, type, lot, entry_price, sl, tp, confidence
        """
        text = (
            f"<b>TRADE ABIERTO</b>\n"
            f"{trade_info.get('type', '?')} {trade_info.get('symbol', '?')}\n"
            f"Precio: {trade_info.get('entry_price', 0):.5f}\n"
            f"Lot: {trade_info.get('lot', 0):.2f}\n"
            f"SL: {trade_info.get('sl', 0):.5f}\n"
            f"TP: {trade_info.get('tp', 0):.5f}\n"
            f"Confianza: {trade_info.get('confidence', 0):.0%}"
        )
        return self.send_message(text)

    def notify_trade_closed(self, trade_info: dict) -> bool:
        """
        Notificar cierre de trade.

        Args:
            trade_info: Dict con symbol, type, profit, exit_reason, pips
        """
        profit = trade_info.get("profit", 0)
        emoji_result = "+" if profit >= 0 else ""

        text = (
            f"<b>TRADE CERRADO</b>\n"
            f"{trade_info.get('type', '?')} {trade_info.get('symbol', '?')}\n"
            f"P&L: {emoji_result}${profit:.2f} ({trade_info.get('pips', 0):.1f} pips)\n"
            f"Razon: {trade_info.get('exit_reason', '?')}\n"
            f"Duracion: {trade_info.get('duration_bars', 0)} barras"
        )
        return self.send_message(text)

    def notify_daily_summary(self, stats: dict) -> bool:
        """
        Enviar resumen diario.

        Args:
            stats: Dict del RiskManager.get_risk_report()
        """
        daily_pnl = stats.get("daily_pnl", 0)
        sign = "+" if daily_pnl >= 0 else ""

        text = (
            f"<b>RESUMEN DIARIO</b>\n"
            f"Balance: ${stats.get('current_balance', 0):.2f}\n"
            f"P&L dia: {sign}${daily_pnl:.2f}\n"
            f"Trades: {stats.get('daily_trades', 0)}\n"
            f"DD diario: {stats.get('daily_drawdown_pct', 0):.1f}%\n"
            f"DD total: {stats.get('total_drawdown_pct', 0):.1f}%\n"
            f"Trades abiertos: {stats.get('open_trades', 0)}"
        )
        return self.send_message(text)

    def notify_signal_decision(self, decisions: list) -> bool:
        """
        Notificar las decisiones del modelo para todos los pares.
        Se envia cada hora cuando el bot evalua los 3 pares.

        Args:
            decisions: Lista de dicts con {symbol, signal, confidence, reason,
                       entry_price, sl_pips, tp_pips, filters_passed}
        """
        from datetime import datetime
        now = datetime.now().strftime("%H:%M UTC")

        lines = [f"<b>ANALISIS H1 - {now}</b>\n"]

        for d in decisions:
            signal = d.get("signal", "HOLD")
            conf = d.get("confidence", 0)
            symbol = d.get("symbol", "?")
            reason = d.get("reason", "")
            filters = d.get("filters_passed", True)

            if signal == "BUY":
                icon = "BUY"
            elif signal == "SELL":
                icon = "SELL"
            else:
                icon = "HOLD"

            line = f"<b>{symbol}</b>: {icon} ({conf:.0%})"

            if signal != "HOLD":
                sl = d.get("sl_pips", 0)
                tp = d.get("tp_pips", 0)
                line += f" | SL:{sl:.0f} TP:{tp:.0f} pips"
                if not filters:
                    line += " [FILTRADO]"
            elif reason:
                line += f"\n  {reason}"

            lines.append(line)

        return self.send_message("\n".join(lines))

    def notify_trailing_update(self, updates: list) -> bool:
        """
        Notificar actualizaciones de trailing stop.

        Args:
            updates: Lista de dicts con {symbol, type, ticket, old_sl, new_sl, profit}
        """
        if not updates:
            return False

        lines = ["<b>TRAILING STOP UPDATE</b>\n"]
        for u in updates:
            lines.append(
                f"{u.get('type', '?')} {u.get('symbol', '?')} #{u.get('ticket', 0)}\n"
                f"  SL: {u.get('old_sl', 0):.5f} -> {u.get('new_sl', 0):.5f}\n"
                f"  P&L: ${u.get('profit', 0):.2f}"
            )

        return self.send_message("\n".join(lines))

    def notify_positions_status(self, positions: list, prev_positions: dict = None) -> bool:
        """
        Notificar estado detallado de posiciones abiertas (cada 5 min).
        Incluye: precio de entrada vs actual, distancia a SL/TP,
        si el trailing se movio, y un ID secuencial para cada posicion.

        Args:
            positions: Lista de dicts de get_open_positions()
            prev_positions: Dict {ticket: {sl, tp}} del check anterior
                            para detectar si el trailing movio el SL
        """
        if not positions:
            return False

        from datetime import datetime
        now = datetime.now().strftime("%H:%M UTC")

        lines = [f"<b>MONITOR - {now}</b>"]
        lines.append(f"{len(positions)} posicion(es) abierta(s)\n")
        total_profit = 0

        for i, p in enumerate(positions, 1):
            ticket = p.get("ticket", 0)
            symbol = p.get("symbol", "?")
            ptype = p.get("type", "?")
            volume = p.get("volume", 0)
            open_price = p.get("open_price", 0)
            current = p.get("current_price", 0)
            sl = p.get("sl", 0)
            tp = p.get("tp", 0)
            profit = p.get("profit", 0)
            swap = p.get("swap", 0)
            open_time = p.get("time", None)
            total_profit += profit

            # Calcular pips de ganancia/perdida
            pip_size = 0.01 if "JPY" in symbol else 0.0001
            if ptype == "BUY":
                pips_current = (current - open_price) / pip_size
            else:
                pips_current = (open_price - current) / pip_size

            # Distancia a SL y TP en pips
            if ptype == "BUY":
                pips_to_sl = (current - sl) / pip_size if sl > 0 else 0
                pips_to_tp = (tp - current) / pip_size if tp > 0 else 0
            else:
                pips_to_sl = (sl - current) / pip_size if sl > 0 else 0
                pips_to_tp = (current - tp) / pip_size if tp > 0 else 0

            # Duracion
            duration_str = ""
            if open_time:
                delta = datetime.now() - open_time
                hours = int(delta.total_seconds() // 3600)
                mins = int((delta.total_seconds() % 3600) // 60)
                duration_str = f"{hours}h{mins:02d}m"

            # Detectar si el trailing movio el SL
            trailing_info = ""
            if prev_positions and ticket in prev_positions:
                old_sl = prev_positions[ticket].get("sl", 0)
                if old_sl > 0 and sl > 0 and abs(sl - old_sl) > pip_size * 0.5:
                    sl_moved_pips = abs(sl - old_sl) / pip_size
                    trailing_info = f"\n  Trailing: SL movido {sl_moved_pips:.1f} pips ({old_sl:.5f} -> {sl:.5f})"
                else:
                    trailing_info = "\n  Trailing: sin cambio"

            # P&L con signo
            pnl_sign = "+" if profit >= 0 else ""
            pips_sign = "+" if pips_current >= 0 else ""

            lines.append(
                f"<b>#{i} {ptype} {symbol}</b> [ticket:{ticket}]\n"
                f"  Entrada: {open_price:.5f} | Actual: {current:.5f}\n"
                f"  P&L: {pnl_sign}${profit:.2f} ({pips_sign}{pips_current:.1f} pips)\n"
                f"  SL: {sl:.5f} ({pips_to_sl:.0f} pips) | TP: {tp:.5f} ({pips_to_tp:.0f} pips)\n"
                f"  Lot: {volume:.2f} | Swap: ${swap:.2f} | Duracion: {duration_str}"
                f"{trailing_info}"
            )

            # Linea separadora entre posiciones
            if i < len(positions):
                lines.append("")

        # Total
        total_sign = "+" if total_profit >= 0 else ""
        lines.append(f"\n<b>Total P&L: {total_sign}${total_profit:.2f}</b>")

        return self.send_message("\n".join(lines))

    def notify_bot_started(self, account: dict, model_info: str = "", mode: str = "demo") -> bool:
        """
        Notificar que el bot arranco correctamente.

        Args:
            account: Dict de get_account_info() con balance, login, server, etc.
            model_info: Nombre del modelo cargado
            mode: Modo de operacion (demo/live)
        """
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        login = account.get("login", "?")
        server = account.get("server", "?")
        balance = account.get("balance", 0)
        currency = account.get("currency", "USD")
        leverage = account.get("leverage", "?")

        text = (
            f"<b>BOT INICIADO</b>\n"
            f"Modo: {mode.upper()}\n"
            f"Fecha: {now}\n\n"
            f"<b>Cuenta</b>\n"
            f"  Login: {login}\n"
            f"  Servidor: {server}\n"
            f"  Balance: ${balance:.2f} {currency}\n"
            f"  Apalancamiento: 1:{leverage}\n\n"
            f"<b>Configuracion</b>\n"
            f"  Modelo: {model_info}\n"
            f"  Pares: {', '.join(settings.SYMBOLS)}\n"
            f"  Timeframe: {settings.TIMEFRAME_PRIMARY}\n"
            f"  Riesgo/trade: {settings.RISK_PER_TRADE:.0%}\n"
            f"  Max trades: {settings.MAX_OPEN_TRADES}\n"
            f"  Threshold: {settings.CONFIDENCE_THRESHOLD}"
        )
        return self.send_message(text)

    def notify_error(self, error_msg: str) -> bool:
        """
        Alertar error critico.

        Args:
            error_msg: Descripcion del error
        """
        text = (
            f"<b>ERROR CRITICO</b>\n"
            f"{error_msg[:500]}"
        )
        return self.send_message(text)

    def notify_emergency_stop(self, reason: str) -> bool:
        """Alertar emergency stop."""
        text = (
            f"<b>EMERGENCY STOP ACTIVADO</b>\n"
            f"Razon: {reason}\n"
            f"El bot ha detenido todas las operaciones.\n"
            f"Revision manual requerida."
        )
        return self.send_message(text)

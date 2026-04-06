"""
Telegram Notifier - Envio de alertas via Telegram Bot API.
Usa httpx (HTTP directo) en vez de python-telegram-bot async para
compatibilidad con el loop sincrono del bot.

API de Telegram: https://core.telegram.org/bots/api#sendmessage
"""
import logging
from typing import Optional

import httpx

from forex_bot.config import settings

logger = logging.getLogger(__name__)

# URL base de la API de Telegram
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


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
        # Intentar cargar de config si no se pasan directamente
        if bot_token is None or chat_id is None:
            try:
                from forex_bot.config import mt5_config
                bot_token = bot_token or getattr(mt5_config, "TELEGRAM_BOT_TOKEN", "")
                chat_id = chat_id or getattr(mt5_config, "TELEGRAM_CHAT_ID", "")
            except ImportError:
                pass

        self.bot_token = bot_token or ""
        self.chat_id = str(chat_id) if chat_id else ""
        self._enabled = bool(self.bot_token and self.chat_id and settings.TELEGRAM_ENABLED)

        if self._enabled:
            logger.info("Telegram notificaciones habilitadas (chat_id: %s)", self.chat_id)
        else:
            logger.info("Telegram notificaciones deshabilitadas")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

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

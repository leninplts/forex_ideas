"""
Trading Logger - Logging estructurado para el bot de Forex.
Maneja logs de consola, archivo rotativo diario, y CSV de trades.
"""
import csv
import logging
import logging.handlers
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from forex_bot.config import settings


class TradingLogger:
    """
    Logger especializado para trading.
    - Log general a consola + archivo con rotacion diaria
    - Log de trades a CSV separado para analisis
    - Log de senales generadas
    """

    def __init__(self, log_dir: str = None):
        """
        Args:
            log_dir: Directorio para archivos de log (default: settings.LOGS_DIR)
        """
        self.log_dir = Path(log_dir) if log_dir else settings.LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._trades_csv = self.log_dir / "trades.csv"
        self._signals_csv = self.log_dir / "signals.csv"

        # Configurar logging de Python
        self._setup_logging()

        # Crear headers de CSV si no existen
        self._init_csv(self._trades_csv, [
            "timestamp", "symbol", "type", "lot", "entry_price", "exit_price",
            "sl", "tp", "profit", "pips", "exit_reason", "commission", "duration_bars",
            "balance_after",
        ])
        self._init_csv(self._signals_csv, [
            "timestamp", "symbol", "signal", "confidence", "entry_price",
            "sl_price", "tp_price", "sl_pips", "tp_pips",
            "filters_passed", "executed", "reason",
        ])

        self._logger = logging.getLogger("forex_bot")

    def _setup_logging(self):
        """Configurar logging con consola + archivo rotativo."""
        root_logger = logging.getLogger("forex_bot")

        # Evitar duplicar handlers si se llama multiples veces
        if root_logger.handlers:
            return

        root_logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

        formatter = logging.Formatter(
            fmt=settings.LOG_FORMAT,
            datefmt=settings.LOG_DATE_FORMAT,
        )

        # Handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Handler de archivo con rotacion diaria
        log_file = self.log_dir / "forex_bot.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            interval=1,
            backupCount=30,  # Guardar 30 dias
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    def _init_csv(self, filepath: Path, headers: list):
        """Crear archivo CSV con headers si no existe."""
        if not filepath.exists():
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    # ------------------------------------------------------------------
    # Log de trades
    # ------------------------------------------------------------------

    def log_trade(self, trade_info: dict, balance_after: float = 0.0):
        """
        Registrar un trade ejecutado (apertura o cierre) en el CSV de trades.

        Args:
            trade_info: Dict con info del trade (symbol, type, profit, etc.)
            balance_after: Balance despues del trade
        """
        timestamp = datetime.now().isoformat()

        row = [
            timestamp,
            trade_info.get("symbol", ""),
            trade_info.get("type", ""),
            trade_info.get("lot", 0),
            trade_info.get("entry_price", 0),
            trade_info.get("exit_price", 0),
            trade_info.get("sl", 0),
            trade_info.get("tp", 0),
            trade_info.get("profit", 0),
            trade_info.get("pips", 0),
            trade_info.get("exit_reason", ""),
            trade_info.get("commission", 0),
            trade_info.get("duration_bars", 0),
            balance_after,
        ]

        try:
            with open(self._trades_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            self._logger.error("Error escribiendo trade CSV: %s", e)

        # Tambien loggear al log general
        profit = trade_info.get("profit", 0)
        sign = "+" if profit >= 0 else ""
        self._logger.info(
            "TRADE: %s %s %.2f lots | P&L: %s$%.2f | Reason: %s | Balance: $%.2f",
            trade_info.get("type", "?"),
            trade_info.get("symbol", "?"),
            trade_info.get("lot", 0),
            sign, profit,
            trade_info.get("exit_reason", "?"),
            balance_after,
        )

    # ------------------------------------------------------------------
    # Log de senales
    # ------------------------------------------------------------------

    def log_signal(self, signal_info: dict, executed: bool = False):
        """
        Registrar una senal generada (ejecutada o no) en el CSV de senales.

        Args:
            signal_info: Dict del SignalGenerator
            executed: Si la senal fue ejecutada o no
        """
        timestamp = datetime.now().isoformat()

        row = [
            timestamp,
            signal_info.get("symbol", ""),
            signal_info.get("signal", "HOLD"),
            round(signal_info.get("confidence", 0), 4),
            signal_info.get("entry_price", 0),
            signal_info.get("sl_price", 0),
            signal_info.get("tp_price", 0),
            round(signal_info.get("sl_pips", 0), 1),
            round(signal_info.get("tp_pips", 0), 1),
            signal_info.get("filters_passed", True),
            executed,
            signal_info.get("reason", ""),
        ]

        try:
            with open(self._signals_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            self._logger.error("Error escribiendo signal CSV: %s", e)

    # ------------------------------------------------------------------
    # Log de errores
    # ------------------------------------------------------------------

    def log_error(self, error: Exception, context: str = ""):
        """
        Registrar un error con traceback completo.

        Args:
            error: La excepcion
            context: Contexto adicional (ej: "al enviar orden")
        """
        tb = traceback.format_exc()
        self._logger.error(
            "ERROR %s: %s\n%s",
            context, str(error), tb,
        )

    # ------------------------------------------------------------------
    # Resumen diario
    # ------------------------------------------------------------------

    def log_daily_summary(self, stats: dict):
        """
        Registrar resumen diario de operaciones.

        Args:
            stats: Dict del RiskManager.get_risk_report()
        """
        self._logger.info(
            "\n=== RESUMEN DIARIO ===\n"
            "  Balance:        $%.2f\n"
            "  P&L del dia:    $%.2f\n"
            "  Trades del dia: %d\n"
            "  DD diario:      %.1f%%\n"
            "  DD total:       %.1f%%\n"
            "  Trades abiertos: %d\n"
            "========================",
            stats.get("current_balance", 0),
            stats.get("daily_pnl", 0),
            stats.get("daily_trades", 0),
            stats.get("daily_drawdown_pct", 0),
            stats.get("total_drawdown_pct", 0),
            stats.get("open_trades", 0),
        )

    # ------------------------------------------------------------------
    # Leer historial
    # ------------------------------------------------------------------

    def get_trade_history(self) -> Optional[pd.DataFrame]:
        """
        Leer historial de trades del CSV.

        Returns:
            DataFrame con todos los trades registrados, o None si no hay datos.
        """
        if not self._trades_csv.exists():
            return None

        try:
            df = pd.read_csv(self._trades_csv, parse_dates=["timestamp"])
            if df.empty:
                return None
            return df
        except Exception as e:
            self._logger.error("Error leyendo trades CSV: %s", e)
            return None

    def get_signal_history(self) -> Optional[pd.DataFrame]:
        """Leer historial de senales del CSV."""
        if not self._signals_csv.exists():
            return None

        try:
            df = pd.read_csv(self._signals_csv, parse_dates=["timestamp"])
            if df.empty:
                return None
            return df
        except Exception as e:
            self._logger.error("Error leyendo signals CSV: %s", e)
            return None

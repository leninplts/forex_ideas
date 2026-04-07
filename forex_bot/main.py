"""
Main - Bot principal de Forex con IA.

Loop principal:
  1. Conectar a MT5
  2. Cargar modelo entrenado
  3. Cada cierre de vela H1:
     - Descargar datos
     - Generar senales
     - Ejecutar ordenes
     - Gestionar posiciones
  4. Shutdown graceful con Ctrl+C

Uso:
  python forex_bot/main.py              (demo por defecto)
  python forex_bot/main.py --mode live  (PELIGRO: dinero real)
  python forex_bot/main.py --verbose    (logging detallado)
"""
import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Agregar directorio raiz al path para que funcione tanto con
# "python forex_bot/main.py" como con "python -m forex_bot.main"
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from forex_bot.config import settings
from forex_bot.data.collector import DataCollector
from forex_bot.data.feature_engine import FeatureEngine
from forex_bot.models.predictor import ModelPredictor
from forex_bot.strategy.signal_generator import SignalGenerator
from forex_bot.strategy.risk_manager import RiskManager
from forex_bot.execution.mt5_executor import MT5Executor
from forex_bot.execution.order_manager import OrderManager
from forex_bot.utils.logger import TradingLogger
from forex_bot.utils.notifications import TelegramNotifier

logger = logging.getLogger(__name__)

# Flag global para shutdown graceful
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handler para Ctrl+C."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\nShutdown solicitado... cerrando despues del ciclo actual.")


def parse_args():
    parser = argparse.ArgumentParser(description="Forex Bot - Trading en Vivo")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "live"],
                        help="Modo de operacion (default: demo)")
    parser.add_argument("--verbose", action="store_true", help="Logging detallado")
    return parser.parse_args()


class ForexBot:
    """
    Bot principal de Forex con IA.
    Orquesta todos los componentes del sistema.
    """

    def __init__(self, mode: str = "demo"):
        self.mode = mode
        self.running = False

        # Componentes (se inicializan en setup())
        self.collector = None
        self.feature_engine = None
        self.predictor = None
        self.signal_generator = None
        self.risk_manager = None
        self.executor = None
        self.order_manager = None
        self.trading_logger = None
        self.notifier = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """
        Inicializar todos los componentes.

        Returns:
            True si todo se inicializo correctamente.
        """
        print(f"\n{'='*60}")
        print(f"  FOREX BOT - Inicializando ({self.mode.upper()})")
        print(f"{'='*60}\n")

        # 1. Logger y notificador
        print("[1/7] Configurando logger y notificaciones...")
        self.trading_logger = TradingLogger()
        self.notifier = TelegramNotifier()

        # 2. Conexion a MT5
        print("[2/7] Conectando a MetaTrader 5...")
        self.collector = DataCollector()
        if not self.collector.connect():
            logger.error("No se pudo conectar a MT5")
            return False

        # Obtener info de cuenta
        account = self.collector.get_account_info()
        if account:
            print(f"       Cuenta: {account['login']} | Servidor: {account['server']}")
            print(f"       Balance: ${account['balance']:.2f} {account['currency']}")

        # 3. Feature engine
        print("[3/7] Inicializando feature engine...")
        self.feature_engine = FeatureEngine()

        # 4. Modelo ML
        print("[4/7] Cargando modelo ML...")
        self.predictor = ModelPredictor()
        if not self.predictor.load_latest():
            logger.error(
                "No hay modelo entrenado. Ejecuta primero: python -m forex_bot.backtest_runner --save-model"
            )
            self.collector.disconnect()
            return False

        if self.predictor.needs_retraining():
            logger.warning("El modelo tiene mas de %d dias. Se recomienda reentrenar.", settings.MODEL_RETRAIN_DAYS)

        # 5. Signal generator
        print("[5/7] Inicializando signal generator...")
        self.signal_generator = SignalGenerator(self.predictor, self.feature_engine)

        # 6. Risk manager
        print("[6/7] Inicializando risk manager...")
        balance = account["balance"] if account else settings.INITIAL_BALANCE
        self.risk_manager = RiskManager(initial_balance=balance)

        # 7. Executor y order manager
        print("[7/7] Inicializando executor y order manager...")
        self.executor = MT5Executor(self.collector)
        self.order_manager = OrderManager(self.executor, self.risk_manager)

        # Verificacion de seguridad
        if self.mode == "live":
            print("\n  *** ATENCION: MODO LIVE - SE USARA DINERO REAL ***")
            confirm = input("  Escriba 'CONFIRMO' para continuar: ")
            if confirm != "CONFIRMO":
                print("  Cancelado por el usuario.")
                self.collector.disconnect()
                return False

        print(f"\n  Bot inicializado correctamente.")
        print(f"  Symbols: {settings.SYMBOLS}")
        print(f"  Timeframe: {settings.TIMEFRAME_PRIMARY}")
        print(f"  Riesgo por trade: {settings.RISK_PER_TRADE:.0%}")
        print(f"  Max trades: {settings.MAX_OPEN_TRADES}")
        print(f"{'='*60}\n")

        # Registrar comandos de Telegram y notificar startup
        if self.notifier:
            model_name = self.predictor.model_name if hasattr(self.predictor, "model_name") else "?"

            # Registrar /balance
            self.notifier.register_command("balance", self._cmd_balance)
            # Registrar /status
            self.notifier.register_command("status", self._cmd_status)
            # Iniciar polling de comandos
            self.notifier.start_polling()

            # Notificar que el bot arranco
            self.notifier.notify_bot_started(
                account=account or {},
                model_info=model_name,
                mode=self.mode,
            )

        return True

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def run(self):
        """
        Loop principal del bot.
        Dos ciclos independientes:
          - Ciclo H1 (cada hora en punto): evaluar senales, abrir trades
          - Ciclo trailing (cada 5 min): actualizar trailing stops de posiciones abiertas
        """
        global _shutdown_requested

        self.running = True
        logger.info("Bot iniciado en modo %s", self.mode.upper())

        last_processed_hour = -1
        last_trailing_check = 0  # timestamp del ultimo check de trailing
        prev_positions_snapshot = {}  # {ticket: {sl, tp}} para detectar cambios de trailing

        # Intervalo de check del trailing stop (en segundos)
        trailing_interval = settings.TRAILING_STOP_CHECK_SECONDS  # 300 = 5 min

        while self.running and not _shutdown_requested:
            try:
                now = datetime.now()
                now_ts = time.time()

                # =============================================
                # CICLO H1: senales de entrada (cada hora)
                # =============================================
                if now.minute == 0 and now.hour != last_processed_hour:
                    last_processed_hour = now.hour
                    logger.info("--- Ciclo H1: %s ---", now.strftime("%Y-%m-%d %H:%M"))

                    # Procesar cada par y acumular decisiones
                    decisions = []
                    for symbol in settings.SYMBOLS:
                        decision = self._process_symbol(symbol)
                        if decision:
                            decisions.append(decision)

                    # Notificar decisiones del modelo por Telegram
                    if decisions and self.notifier:
                        self.notifier.notify_signal_decision(decisions)

                    # Gestionar posiciones (incluye trailing)
                    self._manage_positions()
                    last_trailing_check = now_ts

                    # Resumen diario (al cierre de cada dia)
                    if now.hour == 0:
                        self._daily_summary()

                # =============================================
                # CICLO TRAILING: cada 5 minutos
                # =============================================
                elif (now_ts - last_trailing_check) >= trailing_interval:
                    # Solo revisar trailing si hay posiciones abiertas
                    if settings.TRAILING_STOP_ENABLED:
                        positions = self.executor.get_open_positions()
                        if positions:
                            logger.debug(
                                "--- Trailing check: %s (%d posiciones) ---",
                                now.strftime("%H:%M"), len(positions),
                            )
                            self._manage_positions()

                            # Notificar estado detallado de posiciones
                            if self.notifier:
                                self.notifier.notify_positions_status(
                                    positions, prev_positions_snapshot,
                                )

                            # Guardar snapshot actual para comparar en el proximo check
                            prev_positions_snapshot = {
                                p["ticket"]: {"sl": p["sl"], "tp": p["tp"]}
                                for p in positions
                            }
                    last_trailing_check = now_ts

                # Dormir para no consumir CPU (revisar cada 10 segundos)
                time.sleep(min(settings.CHECK_INTERVAL_SECONDS, 10))

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Error en loop principal: %s", e, exc_info=True)
                self.trading_logger.log_error(e, "loop principal")
                if self.notifier:
                    self.notifier.notify_error(f"Error en loop: {str(e)[:200]}")

                # Esperar antes de reintentar
                time.sleep(30)

        self._shutdown()

    # ------------------------------------------------------------------
    # Comandos Telegram
    # ------------------------------------------------------------------

    def _cmd_balance(self) -> str:
        """Handler para /balance — consultar balance y estado de cuenta."""
        account = self.collector.get_account_info()
        if not account:
            return "<b>/balance</b>\nNo se pudo obtener info de cuenta"

        positions = self.executor.get_open_positions() if self.executor else []
        total_floating = sum(p.get("profit", 0) for p in positions)
        floating_sign = "+" if total_floating >= 0 else ""

        risk_report = self.risk_manager.get_risk_report()

        return (
            f"<b>BALANCE</b>\n\n"
            f"Balance: ${account['balance']:.2f} {account.get('currency', 'USD')}\n"
            f"Equity: ${account['equity']:.2f}\n"
            f"Margen usado: ${account['margin']:.2f}\n"
            f"Margen libre: ${account['free_margin']:.2f}\n"
            f"Apalancamiento: 1:{account['leverage']}\n\n"
            f"<b>Posiciones</b>\n"
            f"Abiertas: {len(positions)}\n"
            f"P&L flotante: {floating_sign}${total_floating:.2f}\n\n"
            f"<b>Riesgo</b>\n"
            f"DD diario: {risk_report.get('daily_drawdown_pct', 0):.1f}%\n"
            f"DD total: {risk_report.get('total_drawdown_pct', 0):.1f}%\n"
            f"Trades hoy: {risk_report.get('daily_trades', 0)}"
        )

    def _cmd_status(self) -> str:
        """Handler para /status — ver posiciones abiertas en detalle."""
        positions = self.executor.get_open_positions() if self.executor else []

        if not positions:
            return "<b>/status</b>\nNo hay posiciones abiertas"

        lines = [f"<b>POSICIONES ({len(positions)})</b>\n"]
        total_profit = 0

        for i, p in enumerate(positions, 1):
            symbol = p.get("symbol", "?")
            ptype = p.get("type", "?")
            open_price = p.get("open_price", 0)
            current = p.get("current_price", 0)
            profit = p.get("profit", 0)
            sl = p.get("sl", 0)
            tp = p.get("tp", 0)
            volume = p.get("volume", 0)
            open_time = p.get("time", None)
            total_profit += profit

            pip_size = 0.01 if "JPY" in symbol else 0.0001
            if ptype == "BUY":
                pips = (current - open_price) / pip_size
            else:
                pips = (open_price - current) / pip_size

            duration_str = ""
            if open_time:
                delta = datetime.now() - open_time
                hours = int(delta.total_seconds() // 3600)
                mins = int((delta.total_seconds() % 3600) // 60)
                duration_str = f"{hours}h{mins:02d}m"

            pnl_sign = "+" if profit >= 0 else ""
            pips_sign = "+" if pips >= 0 else ""

            lines.append(
                f"<b>#{i} {ptype} {symbol}</b>\n"
                f"  Entrada: {open_price:.5f} | Actual: {current:.5f}\n"
                f"  P&L: {pnl_sign}${profit:.2f} ({pips_sign}{pips:.1f} pips)\n"
                f"  SL: {sl:.5f} | TP: {tp:.5f}\n"
                f"  Lot: {volume:.2f} | {duration_str}"
            )

        total_sign = "+" if total_profit >= 0 else ""
        lines.append(f"\n<b>Total: {total_sign}${total_profit:.2f}</b>")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Procesamiento por simbolo
    # ------------------------------------------------------------------

    def _process_symbol(self, symbol: str) -> dict:
        """
        Procesar un par: descargar datos, generar senal, ejecutar.

        Returns:
            Dict con la decision del modelo para notificacion, o None si fallo.
        """
        try:
            # Descargar datos multi-timeframe
            data = self.collector.get_multiple_timeframes(symbol)
            if data is None:
                logger.warning("No se pudieron obtener datos de %s", symbol)
                return None

            df_h1 = data.get(settings.TIMEFRAME_PRIMARY)
            df_h4 = data.get(settings.TIMEFRAME_HIGHER)
            df_d1 = data.get(settings.TIMEFRAME_DAILY)

            if df_h1 is None or len(df_h1) < 200:
                logger.warning("Datos insuficientes para %s (%d barras)", symbol, len(df_h1) if df_h1 is not None else 0)
                return None

            # Obtener precio actual
            current_price = self.collector.get_current_price(symbol)

            # Generar senal
            signal = self.signal_generator.generate_signal(
                symbol, df_h1, df_h4, df_d1, current_price,
            )

            # Loggear senal
            self.trading_logger.log_signal(signal, executed=False)

            # Preparar decision para notificacion
            decision = {
                "symbol": symbol,
                "signal": signal["signal"],
                "confidence": signal["confidence"],
                "reason": signal.get("reason", ""),
                "entry_price": signal.get("entry_price", 0),
                "sl_pips": signal.get("sl_pips", 0),
                "tp_pips": signal.get("tp_pips", 0),
                "filters_passed": signal.get("filters_passed", True),
            }

            # Procesar si es operable
            if signal["signal"] != "HOLD":
                result = self.order_manager.process_signal(signal)

                if result["executed"]:
                    decision["executed"] = True
                    self.trading_logger.log_signal(signal, executed=True)
                    self.trading_logger.log_trade(signal, self.risk_manager.current_balance)
                    if self.notifier:
                        self.notifier.notify_trade_opened({
                            "symbol": symbol,
                            "type": signal["signal"],
                            "lot": result["lot_size"],
                            "entry_price": signal["entry_price"],
                            "sl": signal["sl_price"],
                            "tp": signal["tp_price"],
                            "confidence": signal["confidence"],
                        })
                else:
                    decision["executed"] = False
                    decision["reject_reason"] = result.get("error_msg", "")
                    logger.info("Senal %s %s no ejecutada: %s", signal["signal"], symbol, result["error_msg"])

            return decision

        except Exception as e:
            logger.error("Error procesando %s: %s", symbol, e, exc_info=True)
            self.trading_logger.log_error(e, f"procesando {symbol}")
            return None

    # ------------------------------------------------------------------
    # Gestion de posiciones
    # ------------------------------------------------------------------

    def _manage_positions(self):
        """Gestionar posiciones abiertas: trailing stops, senales de salida."""
        try:
            # Obtener datos actuales para trailing stops
            current_data = {}
            for symbol in settings.SYMBOLS:
                df = self.collector.get_historical_data(symbol, settings.TIMEFRAME_PRIMARY, bars=50)
                if df is not None:
                    current_data[symbol] = df

            self.order_manager.manage_open_positions(current_data)

            # Sincronizar balance con MT5
            account = self.collector.get_account_info()
            if account:
                self.risk_manager.update_balance(account["balance"])

            # Verificar emergency stop
            if self.risk_manager.is_emergency_stopped:
                logger.critical("EMERGENCY STOP activo - cerrando todas las posiciones")
                self.order_manager.emergency_close_all()
                if self.notifier:
                    self.notifier.notify_emergency_stop("Max drawdown excedido")

        except Exception as e:
            logger.error("Error gestionando posiciones: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Resumen diario
    # ------------------------------------------------------------------

    def _daily_summary(self):
        """Generar y enviar resumen diario."""
        report = self.risk_manager.get_risk_report()
        self.trading_logger.log_daily_summary(report)

        if self.notifier:
            self.notifier.notify_daily_summary(report)

        # Reset stats diarias
        self.risk_manager.reset_daily_stats()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self):
        """Cerrar todo de forma limpia."""
        print("\n--- Cerrando bot ---")
        self.running = False

        # Detener polling de Telegram
        if self.notifier:
            self.notifier.stop_polling()

        # Log final
        report = self.risk_manager.get_risk_report()
        logger.info("Shutdown | Balance: $%.2f | P&L total: $%.2f", report["current_balance"], report["total_pnl"])

        # Desconectar MT5 (NO cerrar posiciones - dejar SL/TP activos)
        if self.collector:
            self.collector.disconnect()

        print("Bot cerrado. Las posiciones abiertas mantienen su SL/TP en MT5.")


def main():
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format=settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)

    # Capturar Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Crear y ejecutar bot
    bot = ForexBot(mode=args.mode)

    if not bot.setup():
        sys.exit(1)

    bot.run()


if __name__ == "__main__":
    main()

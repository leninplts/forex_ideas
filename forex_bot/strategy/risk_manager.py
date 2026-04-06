"""
Risk Manager - Gestion de riesgo y position sizing.
Controla cuanto arriesgar por trade, drawdown limits y emergency stops.
Este es el componente MAS IMPORTANTE del bot.
"""
import logging
import math
from datetime import date, datetime
from typing import Optional

from forex_bot.config import settings

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Gestor de riesgo del bot de trading.
    Controla:
      - Position sizing basado en % de riesgo y SL en pips
      - Limites de operaciones simultaneas
      - Drawdown diario y total
      - Emergency stop cuando se exceden limites
    """

    def __init__(self, initial_balance: float = None):
        """
        Args:
            initial_balance: Balance inicial de la cuenta (default: settings)
        """
        self.initial_balance = initial_balance or settings.INITIAL_BALANCE
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance

        # Tracking diario
        self._daily_start_balance = self.initial_balance
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._current_date = date.today()

        # Tracking global
        self._total_pnl = 0.0
        self._total_trades = 0
        self._open_trades_count = 0

        # Estado
        self._emergency_stop = False

        logger.info(
            "RiskManager inicializado | Balance: $%.2f | Riesgo/trade: %.1f%% | Max DD diario: %.1f%%",
            self.initial_balance,
            settings.RISK_PER_TRADE * 100,
            settings.MAX_DAILY_DRAWDOWN * 100,
        )

    # ------------------------------------------------------------------
    # Position Sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        sl_pips: float,
        symbol: str,
        balance: float = None,
    ) -> float:
        """
        Calcular tamano de posicion basado en riesgo fijo por operacion.

        Formula:
          riesgo_$ = balance * RISK_PER_TRADE
          lot_size = riesgo_$ / (sl_pips * pip_value_per_lot)

        Ejemplo con $100, 2% riesgo, 30 pips SL en EURUSD:
          riesgo = $100 * 0.02 = $2.00
          pip_value = $0.10 por pip con 0.01 lot
          lot_size = $2.00 / (30 * $0.10) = 0.67 -> redondear a 0.01 step -> 0.01
          (Con $100 casi siempre sera micro-lote 0.01)

        Args:
            sl_pips: Distancia del stop loss en pips
            symbol: Par de divisas
            balance: Balance a usar (default: current_balance)

        Returns:
            Lot size redondeado al step mas cercano, entre MIN y MAX.
        """
        if balance is None:
            balance = self.current_balance

        if sl_pips <= 0:
            logger.error("SL en pips debe ser > 0, recibido: %.1f", sl_pips)
            return settings.LOT_SIZE_MIN

        # Riesgo en dolares
        risk_amount = balance * settings.RISK_PER_TRADE

        # Pip value por micro-lote (0.01)
        pip_value_per_micro = settings.PIP_VALUES.get(symbol, 0.10)

        # Lot size = riesgo / (sl_pips * pip_value_per_micro_lot)
        # pip_value_per_micro es para 0.01 lot, asi que lot = riesgo / (sl * pv) * 0.01
        lot_size = risk_amount / (sl_pips * pip_value_per_micro) * settings.LOT_SIZE_MIN

        # Redondear al step mas cercano (hacia abajo para no exceder riesgo)
        lot_size = math.floor(lot_size / settings.LOT_SIZE_STEP) * settings.LOT_SIZE_STEP

        # Clamp entre min y max
        lot_size = max(settings.LOT_SIZE_MIN, min(lot_size, settings.LOT_SIZE_MAX))

        # Redondear a 2 decimales
        lot_size = round(lot_size, 2)

        # Calcular riesgo real
        real_risk = lot_size / settings.LOT_SIZE_MIN * sl_pips * pip_value_per_micro
        real_risk_pct = real_risk / balance * 100

        logger.info(
            "Position size: %.2f lots | SL: %.1f pips | Riesgo: $%.2f (%.1f%% de $%.2f) [%s]",
            lot_size, sl_pips, real_risk, real_risk_pct, balance, symbol,
        )

        return lot_size

    # ------------------------------------------------------------------
    # Verificaciones pre-trade
    # ------------------------------------------------------------------

    def can_open_trade(self, free_margin: float = None) -> tuple:
        """
        Verificar si se puede abrir una nueva operacion.

        Checks:
          1. Emergency stop no activado
          2. Operaciones abiertas < MAX_OPEN_TRADES
          3. Drawdown diario < MAX_DAILY_DRAWDOWN
          4. Drawdown total < MAX_TOTAL_DRAWDOWN
          5. Margen libre suficiente (si se proporciona)

        Args:
            free_margin: Margen libre de la cuenta (opcional, de MT5)

        Returns:
            Tuple (can_trade: bool, reason: str)
        """
        # Verificar si cambio el dia
        self._check_daily_reset()

        # 1. Emergency stop
        if self._emergency_stop:
            return False, "EMERGENCY STOP activo. Revisar manualmente."

        # 2. Max operaciones abiertas
        if self._open_trades_count >= settings.MAX_OPEN_TRADES:
            return False, f"Max operaciones abiertas alcanzado ({self._open_trades_count}/{settings.MAX_OPEN_TRADES})"

        # 3. Drawdown diario
        daily_dd = self.get_daily_drawdown()
        if daily_dd >= settings.MAX_DAILY_DRAWDOWN:
            return False, f"Drawdown diario excedido: {daily_dd:.1%} >= {settings.MAX_DAILY_DRAWDOWN:.1%}"

        # 4. Drawdown total
        total_dd = self.get_total_drawdown()
        if total_dd >= settings.MAX_TOTAL_DRAWDOWN:
            self._emergency_stop = True
            return False, f"EMERGENCY: Drawdown total excedido: {total_dd:.1%} >= {settings.MAX_TOTAL_DRAWDOWN:.1%}"

        # 5. Margen libre
        if free_margin is not None and free_margin < 10:  # Minimo $10 de margen
            return False, f"Margen libre insuficiente: ${free_margin:.2f}"

        return True, "OK"

    # ------------------------------------------------------------------
    # Tracking de operaciones
    # ------------------------------------------------------------------

    def register_trade_opened(self):
        """Registrar que se abrio una nueva operacion."""
        self._open_trades_count += 1
        logger.info("Trade abierto | Operaciones abiertas: %d", self._open_trades_count)

    def register_trade_closed(self, pnl: float):
        """
        Registrar que se cerro una operacion.

        Args:
            pnl: Profit/Loss de la operacion en dolares (positivo = ganancia)
        """
        self._open_trades_count = max(0, self._open_trades_count - 1)

        # Actualizar stats
        self._daily_pnl += pnl
        self._daily_trades += 1
        self._total_pnl += pnl
        self._total_trades += 1

        # Actualizar balance
        self.current_balance += pnl
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        logger.info(
            "Trade cerrado | P&L: $%.2f | Balance: $%.2f | Diario: $%.2f | Open: %d",
            pnl, self.current_balance, self._daily_pnl, self._open_trades_count,
        )

        # Verificar drawdown despues del trade
        total_dd = self.get_total_drawdown()
        if total_dd >= settings.MAX_TOTAL_DRAWDOWN:
            self._emergency_stop = True
            logger.critical(
                "EMERGENCY STOP: Drawdown total %.1f%% excede limite %.1f%%",
                total_dd * 100, settings.MAX_TOTAL_DRAWDOWN * 100,
            )

    def update_balance(self, new_balance: float):
        """
        Actualizar balance desde datos reales de MT5.
        Util para sincronizar con el broker.

        Args:
            new_balance: Balance actual segun MT5
        """
        old_balance = self.current_balance
        self.current_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        if abs(old_balance - new_balance) > 0.01:
            logger.info("Balance sincronizado: $%.2f -> $%.2f", old_balance, new_balance)

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def get_daily_drawdown(self) -> float:
        """
        Calcular drawdown del dia como fraccion.
        Drawdown = perdida_hoy / balance_inicio_dia

        Returns:
            Drawdown como fraccion (ej: 0.05 = 5%)
        """
        self._check_daily_reset()

        if self._daily_start_balance <= 0:
            return 0.0

        if self._daily_pnl >= 0:
            return 0.0

        return abs(self._daily_pnl) / self._daily_start_balance

    def get_total_drawdown(self) -> float:
        """
        Calcular drawdown total desde el peak balance.
        Drawdown = (peak - actual) / peak

        Returns:
            Drawdown como fraccion (ej: 0.15 = 15%)
        """
        if self.peak_balance <= 0:
            return 0.0

        if self.current_balance >= self.peak_balance:
            return 0.0

        return (self.peak_balance - self.current_balance) / self.peak_balance

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _check_daily_reset(self):
        """Resetear stats diarias si cambio el dia."""
        today = date.today()
        if today != self._current_date:
            self.reset_daily_stats()
            self._current_date = today

    def reset_daily_stats(self):
        """Resetear estadisticas diarias. Se llama al inicio de cada dia."""
        logger.info(
            "Reset diario | P&L ayer: $%.2f | Trades: %d",
            self._daily_pnl, self._daily_trades,
        )
        self._daily_start_balance = self.current_balance
        self._daily_pnl = 0.0
        self._daily_trades = 0

    # ------------------------------------------------------------------
    # Emergency
    # ------------------------------------------------------------------

    def trigger_emergency_stop(self, reason: str = "Manual"):
        """Activar emergency stop manualmente."""
        self._emergency_stop = True
        logger.critical("EMERGENCY STOP activado: %s", reason)

    def reset_emergency_stop(self):
        """Desactivar emergency stop (requiere intervencion manual)."""
        self._emergency_stop = False
        logger.warning("Emergency stop desactivado manualmente")

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stop

    # ------------------------------------------------------------------
    # Reportes
    # ------------------------------------------------------------------

    def get_risk_report(self) -> dict:
        """
        Generar reporte completo del estado de riesgo.

        Returns:
            Dict con todas las metricas de riesgo actuales.
        """
        return {
            "current_balance": self.current_balance,
            "initial_balance": self.initial_balance,
            "peak_balance": self.peak_balance,
            "total_pnl": self._total_pnl,
            "total_pnl_pct": (self._total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0,
            "total_trades": self._total_trades,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "daily_drawdown": self.get_daily_drawdown(),
            "daily_drawdown_pct": self.get_daily_drawdown() * 100,
            "total_drawdown": self.get_total_drawdown(),
            "total_drawdown_pct": self.get_total_drawdown() * 100,
            "open_trades": self._open_trades_count,
            "max_open_trades": settings.MAX_OPEN_TRADES,
            "emergency_stop": self._emergency_stop,
            "risk_per_trade": settings.RISK_PER_TRADE,
            "max_daily_drawdown": settings.MAX_DAILY_DRAWDOWN,
            "max_total_drawdown": settings.MAX_TOTAL_DRAWDOWN,
        }

    def __repr__(self) -> str:
        return (
            f"RiskManager(balance=${self.current_balance:.2f}, "
            f"dd_daily={self.get_daily_drawdown():.1%}, "
            f"dd_total={self.get_total_drawdown():.1%}, "
            f"open={self._open_trades_count})"
        )

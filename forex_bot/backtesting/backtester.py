"""
Backtester - Motor de simulacion de estrategias con datos historicos.
Simula trades vela por vela con spread, comisiones, SL/TP realista.
CRITICO: nunca usa datos futuros (no lookahead bias).
"""
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from forex_bot.config import settings
from forex_bot.backtesting.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class Backtester:
    """
    Motor de backtesting para simular estrategias de trading.
    Itera vela por vela, verifica SL/TP con high/low (no solo close),
    aplica spread y comisiones.
    """

    def __init__(
        self,
        initial_balance: float = None,
        commission_per_lot: float = None,
        spread_pips: float = None,
    ):
        """
        Args:
            initial_balance: Balance inicial (default: settings)
            commission_per_lot: Comision por lote estandar (default: settings)
            spread_pips: Spread simulado en pips (default: settings)
        """
        self.initial_balance = initial_balance or settings.BACKTEST_INITIAL_BALANCE
        self.commission_per_lot = commission_per_lot or settings.BACKTEST_COMMISSION_PER_LOT
        self.spread_pips = spread_pips or settings.BACKTEST_SPREAD_PIPS

        # Estado de la simulacion
        self._balance = self.initial_balance
        self._equity_history = []
        self._trades = []
        self._open_positions = []

    # ------------------------------------------------------------------
    # Run principal
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        sl_pips_series: pd.Series = None,
        tp_pips_series: pd.Series = None,
        symbol: str = "EURUSD",
    ) -> dict:
        """
        Ejecutar backtest completo vela por vela.

        Args:
            df: DataFrame OHLCV con DatetimeIndex
            signals: Series con senales {1: BUY, -1: SELL, 0: HOLD} alineada con df
            sl_pips_series: Series con SL en pips por vela (opcional, usa ATR si None)
            tp_pips_series: Series con TP en pips por vela (opcional, calcula por R:R si None)
            symbol: Par de divisas para calcular pip_value

        Returns:
            Dict con: trades, equity_curve, metrics
        """
        # Reset estado
        self._balance = self.initial_balance
        self._equity_history = []
        self._trades = []
        self._open_positions = []

        pip_size = settings.PIP_SIZE.get(symbol, 0.0001)
        pip_value = settings.PIP_VALUES.get(symbol, 0.10)  # por micro-lote

        # Calcular SL/TP por defecto si no se proporcionan
        if sl_pips_series is None or tp_pips_series is None:
            sl_pips_series, tp_pips_series = self._default_sl_tp(df, pip_size)

        logger.info(
            "Iniciando backtest: %d velas | Balance: $%.2f | Spread: %.1f pips | Comision: $%.2f/lot",
            len(df), self.initial_balance, self.spread_pips, self.commission_per_lot,
        )

        # Iterar vela por vela
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]

            # 1. Verificar SL/TP de posiciones abiertas usando high/low de ESTA vela
            self._check_sl_tp(row, pip_size, pip_value, timestamp)

            # 2. Procesar senal de ESTA vela (al cierre)
            if i < len(signals):
                signal = signals.iloc[i]
                sl_pips = sl_pips_series.iloc[i] if i < len(sl_pips_series) else 0
                tp_pips = tp_pips_series.iloc[i] if i < len(tp_pips_series) else 0

                if signal in (1, -1) and sl_pips > 0 and tp_pips > 0:
                    self._process_signal(
                        signal, row, sl_pips, tp_pips,
                        pip_size, pip_value, symbol, timestamp,
                    )

            # 3. Registrar equity (balance + profit no realizado)
            unrealized = sum(
                self._calc_unrealized_pnl(pos, row["close"], pip_size, pip_value)
                for pos in self._open_positions
            )
            self._equity_history.append({
                "time": timestamp,
                "balance": self._balance,
                "equity": self._balance + unrealized,
            })

        # Cerrar posiciones abiertas al final
        self._close_all_at_end(df, pip_size, pip_value)

        # Generar equity curve
        equity_df = pd.DataFrame(self._equity_history)
        if not equity_df.empty:
            equity_df.set_index("time", inplace=True)
            equity_curve = equity_df["equity"]
        else:
            equity_curve = pd.Series(dtype=float)

        # Calcular metricas
        metrics = PerformanceMetrics.calculate_all(
            self._trades, equity_curve, self.initial_balance,
        )

        logger.info(
            "Backtest completado: %d trades | P&L: $%.2f (%.1f%%) | Win rate: %.1f%% | Sharpe: %.2f",
            metrics["total_trades"],
            metrics["total_profit"],
            metrics["total_return_pct"],
            metrics["win_rate_pct"],
            metrics["sharpe_ratio"],
        )

        return {
            "trades": self._trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Procesamiento de senales
    # ------------------------------------------------------------------

    def _process_signal(
        self,
        signal: int,
        row,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        pip_value: float,
        symbol: str,
        timestamp,
    ):
        """Procesar una senal y abrir posicion si pasa validaciones."""
        # Verificar max trades abiertos
        if len(self._open_positions) >= settings.MAX_OPEN_TRADES:
            return

        # No abrir en la misma direccion si ya hay posicion abierta en ese par
        for pos in self._open_positions:
            if pos["symbol"] == symbol and pos["type"] == signal:
                return

        # Calcular position size
        risk_amount = self._balance * settings.RISK_PER_TRADE
        lot = risk_amount / (sl_pips * pip_value) * settings.LOT_SIZE_MIN
        lot = max(settings.LOT_SIZE_MIN, min(lot, settings.LOT_SIZE_MAX))
        lot = round(lot, 2)

        # Precio de entrada (con spread simulado)
        spread_cost = self.spread_pips * pip_size
        if signal == 1:  # BUY
            entry_price = row["close"] + spread_cost / 2  # ask = close + spread/2
            sl_price = entry_price - sl_pips * pip_size
            tp_price = entry_price + tp_pips * pip_size
        else:  # SELL
            entry_price = row["close"] - spread_cost / 2  # bid = close - spread/2
            sl_price = entry_price + sl_pips * pip_size
            tp_price = entry_price - tp_pips * pip_size

        # Comision
        commission = self.commission_per_lot * (lot / 1.0)  # Proporcional al lot

        # Abrir posicion
        position = {
            "symbol": symbol,
            "type": signal,  # 1=BUY, -1=SELL
            "entry_price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "lot": lot,
            "entry_time": timestamp,
            "commission": commission,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
        }
        self._open_positions.append(position)

    # ------------------------------------------------------------------
    # Check SL/TP
    # ------------------------------------------------------------------

    def _check_sl_tp(self, row, pip_size: float, pip_value: float, timestamp):
        """
        Verificar si alguna posicion abierta alcanzo SL o TP.
        USA HIGH/LOW de la vela, no solo close (mas realista).

        Regla de prioridad: si AMBOS SL y TP son alcanzados en la misma vela,
        asumimos que SL se alcanzo primero (peor caso, conservador).
        """
        closed = []

        for pos in self._open_positions:
            exit_price = None
            exit_reason = None

            if pos["type"] == 1:  # BUY
                # SL: low llega al SL
                if row["low"] <= pos["sl"]:
                    exit_price = pos["sl"]
                    exit_reason = "SL"
                # TP: high llega al TP
                elif row["high"] >= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TP"

            elif pos["type"] == -1:  # SELL
                # SL: high llega al SL
                if row["high"] >= pos["sl"]:
                    exit_price = pos["sl"]
                    exit_reason = "SL"
                # TP: low llega al TP
                elif row["low"] <= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TP"

            if exit_price is not None:
                self._close_position(pos, exit_price, exit_reason, pip_size, pip_value, timestamp)
                closed.append(pos)

        # Remover posiciones cerradas
        for pos in closed:
            self._open_positions.remove(pos)

    def _close_position(
        self, pos: dict, exit_price: float, exit_reason: str,
        pip_size: float, pip_value: float, timestamp,
    ):
        """Cerrar una posicion y registrar el trade."""
        # Calcular P&L
        if pos["type"] == 1:  # BUY
            pips_gained = (exit_price - pos["entry_price"]) / pip_size
        else:  # SELL
            pips_gained = (pos["entry_price"] - exit_price) / pip_size

        # P&L en dolares: pips * pip_value * (lot / 0.01)
        lot_multiplier = pos["lot"] / settings.LOT_SIZE_MIN
        profit = pips_gained * pip_value * lot_multiplier

        # Restar comision
        profit -= pos["commission"]

        # Actualizar balance
        self._balance += profit

        # Calcular duracion
        duration_bars = 0
        if hasattr(timestamp, '__sub__') and hasattr(pos["entry_time"], '__sub__'):
            try:
                delta = timestamp - pos["entry_time"]
                duration_bars = int(delta.total_seconds() / 3600)  # Asumiendo H1
            except Exception:
                pass

        # Registrar trade
        trade = {
            "entry_time": pos["entry_time"],
            "exit_time": timestamp,
            "symbol": pos["symbol"],
            "type": "BUY" if pos["type"] == 1 else "SELL",
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "lot": pos["lot"],
            "sl": pos["sl"],
            "tp": pos["tp"],
            "profit": round(profit, 2),
            "pips": round(pips_gained, 1),
            "exit_reason": exit_reason,
            "commission": round(pos["commission"], 4),
            "duration_bars": duration_bars,
        }
        self._trades.append(trade)

    def _close_all_at_end(self, df: pd.DataFrame, pip_size: float, pip_value: float):
        """Cerrar todas las posiciones al final del backtest al precio de cierre."""
        if not self._open_positions:
            return

        last_close = df["close"].iloc[-1]
        last_time = df.index[-1]

        for pos in list(self._open_positions):
            self._close_position(pos, last_close, "END", pip_size, pip_value, last_time)

        self._open_positions.clear()
        logger.info("Cerradas %d posiciones al final del backtest", len(self._open_positions))

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _calc_unrealized_pnl(self, pos: dict, current_price: float, pip_size: float, pip_value: float) -> float:
        """Calcular P&L no realizado de una posicion abierta."""
        if pos["type"] == 1:  # BUY
            pips = (current_price - pos["entry_price"]) / pip_size
        else:  # SELL
            pips = (pos["entry_price"] - current_price) / pip_size

        lot_multiplier = pos["lot"] / settings.LOT_SIZE_MIN
        return pips * pip_value * lot_multiplier

    def _default_sl_tp(self, df: pd.DataFrame, pip_size: float) -> tuple:
        """
        Calcular SL/TP por defecto basados en ATR cuando no se proporcionan.

        Returns:
            Tuple (sl_pips_series, tp_pips_series)
        """
        import pandas_ta as ta

        atr = ta.atr(df["high"], df["low"], df["close"], length=settings.ATR_PERIOD)
        if atr is None:
            # Fallback: rango promedio
            atr = (df["high"] - df["low"]).rolling(window=settings.ATR_PERIOD).mean()

        sl_pips = (atr * settings.SL_ATR_MULTIPLIER) / pip_size
        tp_pips = (atr * settings.TP_ATR_MULTIPLIER) / pip_size

        # Minimo 10 pips de SL
        sl_pips = sl_pips.clip(lower=10)
        tp_pips = tp_pips.clip(lower=sl_pips * settings.MIN_RISK_REWARD)

        # Fill NaN con valores razonables
        sl_pips = sl_pips.ffill().fillna(30)
        tp_pips = tp_pips.ffill().fillna(45)

        return sl_pips, tp_pips

    # ------------------------------------------------------------------
    # Generar reporte
    # ------------------------------------------------------------------

    def generate_report(self) -> dict:
        """
        Generar reporte completo del ultimo backtest ejecutado.

        Returns:
            Dict con trades, equity_curve, metrics.
        """
        if not self._equity_history:
            logger.warning("No hay datos de backtest. Ejecutar run() primero.")
            return {}

        equity_df = pd.DataFrame(self._equity_history)
        equity_df.set_index("time", inplace=True)
        equity_curve = equity_df["equity"]

        metrics = PerformanceMetrics.calculate_all(
            self._trades, equity_curve, self.initial_balance,
        )

        return {
            "trades": self._trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
        }

    def get_trades_df(self) -> pd.DataFrame:
        """Retornar trades como DataFrame para analisis."""
        if not self._trades:
            return pd.DataFrame()

        df = pd.DataFrame(self._trades)
        if "entry_time" in df.columns:
            df.set_index("entry_time", inplace=True)
        return df

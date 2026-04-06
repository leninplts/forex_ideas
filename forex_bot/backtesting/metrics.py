"""
Performance Metrics - Calculo de metricas de rendimiento para backtesting.
Todas las metricas financieras estandar: Sharpe, Sortino, drawdown, profit factor, etc.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculadora de metricas de rendimiento para trading.
    Todos los metodos son estaticos para poder usarse sin instanciar.
    """

    # ------------------------------------------------------------------
    # Metodo principal
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_all(trades: list, equity_curve: pd.Series, initial_balance: float = 100.0) -> dict:
        """
        Calcular TODAS las metricas a partir de la lista de trades y curva de equity.

        Args:
            trades: Lista de dicts con trades cerrados. Cada trade tiene:
                    {entry_time, exit_time, symbol, type, entry_price, exit_price,
                     lot, sl, tp, profit, exit_reason, duration_bars}
            equity_curve: Series con el balance a lo largo del tiempo (index=datetime)
            initial_balance: Balance inicial

        Returns:
            Dict con todas las metricas organizadas por categoria.
        """
        if not trades:
            return PerformanceMetrics._empty_metrics()

        profits = [t["profit"] for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        # --- RENDIMIENTO ---
        total_profit = sum(profits)
        total_return_pct = (total_profit / initial_balance) * 100

        # Retorno anualizado
        if len(equity_curve) > 1:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            years = max(days / 365.25, 0.01)
            final_balance = equity_curve.iloc[-1]
            annual_return = ((final_balance / initial_balance) ** (1 / years) - 1) * 100
        else:
            annual_return = 0.0

        # Retornos mensuales
        monthly_returns = PerformanceMetrics._monthly_returns(equity_curve)

        # --- RIESGO ---
        max_dd, max_dd_duration = PerformanceMetrics._max_drawdown(equity_curve)
        sharpe = PerformanceMetrics._sharpe_ratio(equity_curve)
        sortino = PerformanceMetrics._sortino_ratio(equity_curve)
        calmar = abs(annual_return / (max_dd * 100)) if max_dd > 0 else 0.0

        # --- OPERACIONES ---
        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 0.0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        # R:R realizado promedio
        rr_list = []
        for t in trades:
            if t.get("sl") and t.get("entry_price"):
                risk = abs(t["entry_price"] - t["sl"])
                if risk > 0:
                    reward = abs(t["profit"]) / (t.get("lot", 0.01) * 100000 * risk) if t["profit"] > 0 else 0
                    rr_list.append(reward)

        max_consec_wins = PerformanceMetrics._max_consecutive(profits, positive=True)
        max_consec_losses = PerformanceMetrics._max_consecutive(profits, positive=False)

        # Duracion promedio
        durations = [t.get("duration_bars", 0) for t in trades if t.get("duration_bars")]
        avg_duration = np.mean(durations) if durations else 0.0

        # --- ESTABILIDAD ---
        recovery_factor = total_profit / (max_dd * initial_balance) if max_dd > 0 else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            # Rendimiento
            "total_profit": round(total_profit, 2),
            "total_return_pct": round(total_return_pct, 2),
            "annual_return_pct": round(annual_return, 2),
            "monthly_returns": monthly_returns,

            # Riesgo
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "max_drawdown_duration_days": max_dd_duration,
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),

            # Operaciones
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_rate, 4),
            "win_rate_pct": round(win_rate * 100, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio != float("inf") else "inf",
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "avg_trade_duration_bars": round(avg_duration, 1),

            # Estabilidad
            "recovery_factor": round(recovery_factor, 2),
            "expectancy": round(expectancy, 2),

            # Balance
            "initial_balance": initial_balance,
            "final_balance": round(initial_balance + total_profit, 2),
        }

    # ------------------------------------------------------------------
    # Metricas individuales
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> tuple:
        """
        Calcular maximo drawdown y su duracion.

        Returns:
            Tuple (max_drawdown_fraction, duration_days)
        """
        if len(equity_curve) < 2:
            return 0.0, 0

        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max

        max_dd = abs(drawdown.min())

        # Duracion del max drawdown
        dd_duration = 0
        if max_dd > 0:
            in_drawdown = drawdown < 0
            if in_drawdown.any():
                # Encontrar el drawdown mas largo
                groups = (~in_drawdown).cumsum()
                dd_lengths = in_drawdown.groupby(groups).sum()
                if len(dd_lengths) > 0:
                    max_dd_bars = dd_lengths.max()
                    # Estimar dias (asumiendo H1 = 24 barras/dia)
                    dd_duration = int(max_dd_bars / 24) if max_dd_bars > 0 else 0

        return max_dd, dd_duration

    @staticmethod
    def _sharpe_ratio(equity_curve: pd.Series, risk_free: float = 0.0) -> float:
        """
        Calcular Sharpe Ratio anualizado.
        Sharpe = (mean_return - risk_free) / std_return * sqrt(252)

        Nota: Usamos retornos porcentuales de la equity curve.
        252 = dias de trading por anho.
        """
        if len(equity_curve) < 10:
            return 0.0

        returns = equity_curve.pct_change().dropna()
        if returns.std() == 0:
            return 0.0

        sharpe = (returns.mean() - risk_free) / returns.std() * np.sqrt(252)
        return sharpe

    @staticmethod
    def _sortino_ratio(equity_curve: pd.Series, risk_free: float = 0.0) -> float:
        """
        Calcular Sortino Ratio (como Sharpe pero solo con volatilidad negativa).
        Sortino = (mean_return - risk_free) / downside_std * sqrt(252)
        """
        if len(equity_curve) < 10:
            return 0.0

        returns = equity_curve.pct_change().dropna()
        downside = returns[returns < 0]

        if len(downside) == 0 or downside.std() == 0:
            return 0.0

        sortino = (returns.mean() - risk_free) / downside.std() * np.sqrt(252)
        return sortino

    @staticmethod
    def _monthly_returns(equity_curve: pd.Series) -> dict:
        """Calcular retornos mensuales como dict {YYYY-MM: return_pct}."""
        if len(equity_curve) < 2:
            return {}

        monthly = equity_curve.resample("ME").last()
        returns = monthly.pct_change().dropna() * 100

        return {dt.strftime("%Y-%m"): round(r, 2) for dt, r in returns.items()}

    @staticmethod
    def _max_consecutive(profits: list, positive: bool = True) -> int:
        """Calcular racha maxima de wins o losses consecutivos."""
        max_streak = 0
        current = 0

        for p in profits:
            if (positive and p > 0) or (not positive and p <= 0):
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0

        return max_streak

    @staticmethod
    def _empty_metrics() -> dict:
        """Retornar metricas vacias cuando no hay trades."""
        return {
            "total_profit": 0, "total_return_pct": 0, "annual_return_pct": 0,
            "monthly_returns": {},
            "max_drawdown": 0, "max_drawdown_pct": 0, "max_drawdown_duration_days": 0,
            "sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0,
            "total_trades": 0, "win_count": 0, "loss_count": 0,
            "win_rate": 0, "win_rate_pct": 0,
            "gross_profit": 0, "gross_loss": 0, "profit_factor": 0,
            "avg_win": 0, "avg_loss": 0, "payoff_ratio": 0,
            "max_consecutive_wins": 0, "max_consecutive_losses": 0,
            "avg_trade_duration_bars": 0,
            "recovery_factor": 0, "expectancy": 0,
            "initial_balance": 0, "final_balance": 0,
        }

    # ------------------------------------------------------------------
    # Reportes
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(metrics: dict):
        """Imprimir reporte de rendimiento formateado en consola."""
        print("\n" + "=" * 60)
        print("         REPORTE DE RENDIMIENTO - BACKTEST")
        print("=" * 60)

        print(f"\n--- RENDIMIENTO ---")
        print(f"  Balance inicial:    ${metrics['initial_balance']:.2f}")
        print(f"  Balance final:      ${metrics['final_balance']:.2f}")
        print(f"  Profit total:       ${metrics['total_profit']:.2f} ({metrics['total_return_pct']:.1f}%)")
        print(f"  Retorno anualizado: {metrics['annual_return_pct']:.1f}%")

        print(f"\n--- RIESGO ---")
        print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:       {metrics['calmar_ratio']:.2f}")

        print(f"\n--- OPERACIONES ---")
        print(f"  Total trades:       {metrics['total_trades']}")
        print(f"  Win rate:           {metrics['win_rate_pct']:.1f}%")
        print(f"  Profit Factor:      {metrics['profit_factor']}")
        print(f"  Avg Win:            ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss:           ${metrics['avg_loss']:.2f}")
        print(f"  Payoff Ratio:       {metrics['payoff_ratio']}")
        print(f"  Max Wins seguidos:  {metrics['max_consecutive_wins']}")
        print(f"  Max Losses seguidos:{metrics['max_consecutive_losses']}")

        print(f"\n--- ESTABILIDAD ---")
        print(f"  Recovery Factor:    {metrics['recovery_factor']:.2f}")
        print(f"  Expectancy:         ${metrics['expectancy']:.2f}")
        print("=" * 60 + "\n")

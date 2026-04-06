"""
Optimize Prediction Horizon - Prueba diferentes horizontes de prediccion
y selecciona el que da mejores resultados en walk-forward.

El horizonte actual es 3 velas H1 (3 horas), que puede ser demasiado
ruidoso. Probamos 3, 5, 8, 12 velas para encontrar el optimo.

Uso:
  python scripts/optimize_horizon.py
  python scripts/optimize_horizon.py --symbol EURUSDm
"""
import argparse
import logging
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from forex_bot.config import settings
from forex_bot.data.preprocessor import DataPreprocessor
from forex_bot.data.feature_engine import FeatureEngine
from forex_bot.models.trainer import ModelTrainer
from forex_bot.backtesting.backtester import Backtester
from forex_bot.backtesting.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


def test_horizon(
    df: pd.DataFrame,
    df_h4: pd.DataFrame,
    df_d1: pd.DataFrame,
    horizon: int,
    symbol: str,
    min_pips: int = None,
) -> dict:
    """Entrenar y evaluar con un horizonte especifico usando walk-forward."""
    min_pips = min_pips if min_pips is not None else settings.MIN_MOVEMENT_PIPS
    pip_size = settings.PIP_SIZE.get(symbol, 0.0001)

    fe = FeatureEngine()
    df_feat = fe.add_all_features(df.copy())
    if df_h4 is not None:
        df_feat = fe.add_higher_timeframe_features(df_feat, df_h4, suffix="h4")
    if df_d1 is not None:
        df_feat = fe.add_higher_timeframe_features(df_feat, df_d1, suffix="d1")

    df_feat["target"] = fe.create_target(df_feat, horizon=horizon, min_pips=min_pips, pip_size=pip_size)

    pp = DataPreprocessor()
    splits = pp.create_walk_forward_splits(df_feat, 120, 30, 15)
    if not splits:
        return {"horizon": horizon, "error": "No splits"}

    # Acumular senales de todas las ventanas
    all_signals = pd.Series(0, index=df_feat.index, dtype=int)
    test_accs = []

    for train_df, val_df in splits:
        base_type = "xgboost"
        trainer = ModelTrainer(model_type=base_type)
        try:
            X_train_full, y_train_full = trainer.prepare_features(train_df)
            X_val, y_val = trainer.prepare_features(val_df)
        except Exception:
            continue

        if len(X_train_full) < 100 or len(X_val) < 20:
            continue

        es_split = int(len(X_train_full) * 0.75)
        cal_split = int(len(X_train_full) * 0.85)
        X_train = X_train_full.iloc[:es_split]
        y_train = y_train_full.iloc[:es_split]
        X_es = X_train_full.iloc[es_split:cal_split]
        y_es = y_train_full.iloc[es_split:cal_split]

        # Entrenar ensemble
        trainer.train_ensemble(X_train, y_train, X_es, y_es)
        test_eval = trainer.evaluate(X_val, y_val)
        test_accs.append(test_eval["accuracy"])

        # Generar senales
        y_proba = trainer.model.predict_proba(X_val)
        threshold = settings.CONFIDENCE_THRESHOLD
        is_binary = y_proba.shape[1] == 2

        for j, idx in enumerate(X_val.index):
            if idx not in df_feat.index:
                continue
            pos = df_feat.index.get_loc(idx)
            if is_binary:
                prob_up = y_proba[j, 1]
                if prob_up > threshold:
                    all_signals.iloc[pos] = 1
                elif prob_up < (1 - threshold):
                    all_signals.iloc[pos] = -1
            else:
                pred = int(np.argmax(y_proba[j]))
                conf = y_proba[j, pred]
                if conf > threshold:
                    signal_map = {0: -1, 1: 0, 2: 1}
                    all_signals.iloc[pos] = signal_map.get(pred, 0)

    # Backtest
    bt = Backtester(initial_balance=settings.BACKTEST_INITIAL_BALANCE)
    result = bt.run(df_feat, all_signals, symbol=symbol)
    metrics = result["metrics"]

    return {
        "horizon": horizon,
        "avg_test_acc": np.mean(test_accs) if test_accs else 0,
        "win_rate_pct": metrics["win_rate_pct"],
        "profit_factor": metrics["profit_factor"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "total_return_pct": metrics["total_return_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "total_trades": metrics["total_trades"],
        "expectancy": metrics["expectancy"],
        "final_balance": metrics["final_balance"],
    }


def main():
    parser = argparse.ArgumentParser(description="Optimizar horizonte de prediccion")
    parser.add_argument("--symbol", type=str, default="EURUSDm")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format=settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)

    symbol = args.symbol
    data_dir = settings.DATA_DIR
    pp = DataPreprocessor()

    print(f"\n{'='*70}")
    print(f"  OPTIMIZACION DE HORIZONTE DE PREDICCION - {symbol}")
    print(f"{'='*70}\n")

    # Cargar datos
    df = pd.read_csv(data_dir / f"{symbol}_H1.csv", index_col="time", parse_dates=True)
    df = pp.clean_data(df)

    df_h4, df_d1 = None, None
    h4_path = data_dir / f"{symbol}_H4.csv"
    d1_path = data_dir / f"{symbol}_D1.csv"
    if h4_path.exists():
        df_h4 = pd.read_csv(h4_path, index_col="time", parse_dates=True)
        df_h4 = pp.clean_data(df_h4)
    if d1_path.exists():
        df_d1 = pd.read_csv(d1_path, index_col="time", parse_dates=True)
        df_d1 = pp.clean_data(df_d1)

    horizons = [3, 5, 8, 12]
    results = []

    for h in horizons:
        print(f"  Probando horizon={h} velas H1 ({h} horas)...")
        res = test_horizon(df, df_h4, df_d1, h, symbol)
        results.append(res)
        pf = res['profit_factor']
        pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) and pf != float('inf') else str(pf)
        print(f"    WR={res['win_rate_pct']:.1f}% | PF={pf_str} | "
              f"Sharpe={res['sharpe_ratio']:.2f} | Return={res['total_return_pct']:.1f}% | "
              f"Trades={res['total_trades']}")

    # Tabla comparativa
    print(f"\n{'='*90}")
    print(f"  COMPARACION DE HORIZONTES")
    print(f"{'='*90}")
    header = f"{'Horizon':>8} {'TestAcc':>8} {'WinRate':>8} {'PF':>7} {'Sharpe':>7} {'Return%':>9} {'MaxDD%':>8} {'Trades':>7} {'Expect$':>8}"
    print(header)
    print("-" * 90)

    best = None
    best_sharpe = -999

    for r in results:
        pf = r['profit_factor']
        pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) and pf != float('inf') else str(pf)
        sharpe = r['sharpe_ratio']
        marker = ""
        if isinstance(sharpe, (int, float)) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best = r
            marker = " <-- BEST"
        print(
            f"{r['horizon']:>7}h "
            f"{r['avg_test_acc']:>7.1%} "
            f"{r['win_rate_pct']:>7.1f}% "
            f"{pf_str:>7} "
            f"{sharpe:>7.2f} "
            f"{r['total_return_pct']:>8.1f}% "
            f"{r['max_drawdown_pct']:>7.1f}% "
            f"{r['total_trades']:>7} "
            f"${r['expectancy']:>7.2f}"
            f"{marker}"
        )

    print("=" * 90)

    if best:
        print(f"\n  Mejor horizonte por Sharpe: {best['horizon']} velas H1")
        if best['horizon'] != settings.PREDICTION_HORIZON:
            print(f"  RECOMENDACION: Cambiar PREDICTION_HORIZON de {settings.PREDICTION_HORIZON} a {best['horizon']}")
        else:
            print(f"  El horizonte actual ({settings.PREDICTION_HORIZON}) ya es optimo.")
    print()


if __name__ == "__main__":
    main()

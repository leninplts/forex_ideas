"""
Compara modelo actual (complejo) vs modelo simplificado,
con datos cortos (~1-2 anos) vs datos largos (5 anos).

Objetivo: encontrar la combinacion optima de complejidad + cantidad de datos
que minimice el gap de overfitting y maximice el Sharpe.

Ejecutar:
  python scripts/compare_complexity.py
  python scripts/compare_complexity.py --symbol EURUSDm
"""
import argparse
import logging
import sys
from pathlib import Path
from copy import deepcopy

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

logger = logging.getLogger(__name__)

# =====================================================================
# Configuraciones de modelo a probar
# =====================================================================

# Modelo actual (ya tiene algo de regularizacion)
PARAMS_CURRENT = {
    "xgboost": deepcopy(settings.XGBOOST_PARAMS),
    "lightgbm": deepcopy(settings.LIGHTGBM_PARAMS),
    "label": "actual (max_depth=4)",
}

# Modelo simplificado: mucha mas regularizacion
PARAMS_SIMPLE = {
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 3,                # 4 -> 3 (arboles menos profundos)
        "learning_rate": 0.03,          # 0.05 -> 0.03 (aprender mas lento)
        "subsample": 0.7,              # 0.8 -> 0.7 (menos datos por arbol)
        "colsample_bytree": 0.5,       # 0.7 -> 0.5 (menos features por arbol)
        "min_child_weight": 20,         # 10 -> 20 (hojas con mas muestras)
        "gamma": 0.5,                  # 0.2 -> 0.5 (mas costo de split)
        "reg_alpha": 1.0,             # 0.5 -> 1.0 (mas L1)
        "reg_lambda": 5.0,            # 2.0 -> 5.0 (mas L2)
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "early_stopping_rounds": 30,
        "random_state": 42,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 3,                # 4 -> 3
        "learning_rate": 0.03,         # 0.05 -> 0.03
        "subsample": 0.7,             # 0.8 -> 0.7
        "colsample_bytree": 0.5,      # 0.7 -> 0.5
        "min_child_weight": 20,        # 10 -> 20
        "num_leaves": 15,             # 31 -> 15 (mucho menos hojas)
        "min_data_in_leaf": 50,        # Nuevo: minimo 50 muestras por hoja
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "label": "simplificado (max_depth=3, high_reg)",
}


def run_experiment(
    symbol: str,
    h1_csv: str,
    h4_csv: str,
    d1_csv: str,
    params_config: dict,
    data_label: str,
    wf_train_days: int = 365,
    wf_val_days: int = 60,
    wf_step_days: int = 30,
) -> dict:
    """Ejecutar walk-forward con una configuracion especifica."""
    pp = DataPreprocessor()
    fe = FeatureEngine()
    data_dir = settings.DATA_DIR

    # Cargar datos
    df = pd.read_csv(data_dir / h1_csv, index_col="time", parse_dates=True)
    df = pp.clean_data(df)
    df = fe.add_all_features(df)

    h4_path = data_dir / h4_csv
    d1_path = data_dir / d1_csv
    if h4_path.exists():
        df_h4 = pd.read_csv(h4_path, index_col="time", parse_dates=True)
        df_h4 = pp.clean_data(df_h4)
        df = fe.add_higher_timeframe_features(df, df_h4, suffix="h4")
    if d1_path.exists():
        df_d1 = pd.read_csv(d1_path, index_col="time", parse_dates=True)
        df_d1 = pp.clean_data(df_d1)
        df = fe.add_higher_timeframe_features(df, df_d1, suffix="d1")

    pip_size = settings.PIP_SIZE.get(symbol, 0.0001)
    df["target"] = fe.create_target(df, pip_size=pip_size)

    days = (df.index[-1] - df.index[0]).days
    bars = len(df)

    # Walk-forward
    splits = pp.create_walk_forward_splits(df, wf_train_days, wf_val_days, wf_step_days)
    if not splits:
        return {"error": "No splits", "data_label": data_label}

    # Temporalmente cambiar los params en settings
    orig_xgb = deepcopy(settings.XGBOOST_PARAMS)
    orig_lgb = deepcopy(settings.LIGHTGBM_PARAMS)

    if "xgboost" in params_config:
        for k, v in params_config["xgboost"].items():
            settings.XGBOOST_PARAMS[k] = v
    if "lightgbm" in params_config:
        for k, v in params_config["lightgbm"].items():
            settings.LIGHTGBM_PARAMS[k] = v

    all_signals = pd.Series(0, index=df.index, dtype=int)
    train_accs, test_accs = [], []

    for train_df, val_df in splits:
        trainer = ModelTrainer(model_type="xgboost")
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

        trainer.train_ensemble(X_train, y_train, X_es, y_es)

        train_eval = trainer.evaluate(X_train, y_train)
        test_eval = trainer.evaluate(X_val, y_val)
        train_accs.append(train_eval["accuracy"])
        test_accs.append(test_eval["accuracy"])

        y_proba = trainer.model.predict_proba(X_val)
        threshold = settings.CONFIDENCE_THRESHOLD
        is_binary = y_proba.shape[1] == 2

        for j, idx in enumerate(X_val.index):
            if idx not in df.index:
                continue
            pos = df.index.get_loc(idx)
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

    # Restaurar settings originales
    settings.XGBOOST_PARAMS.clear()
    settings.XGBOOST_PARAMS.update(orig_xgb)
    settings.LIGHTGBM_PARAMS.clear()
    settings.LIGHTGBM_PARAMS.update(orig_lgb)

    # Backtest
    bt = Backtester(
        initial_balance=settings.BACKTEST_INITIAL_BALANCE,
        apply_session_filter=True,
    )
    result = bt.run(df, all_signals, symbol=symbol)
    metrics = result["metrics"]

    avg_train = np.mean(train_accs) if train_accs else 0
    avg_test = np.mean(test_accs) if test_accs else 0

    return {
        "data_label": data_label,
        "model_label": params_config.get("label", "?"),
        "bars": bars,
        "days": days,
        "n_splits": len(splits),
        "avg_train_acc": avg_train,
        "avg_test_acc": avg_test,
        "overfit_gap": avg_train - avg_test,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EURUSDm")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format=settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)

    symbol = args.symbol

    print(f"\n{'='*90}")
    print(f"  COMPARACION: COMPLEJIDAD DEL MODELO vs CANTIDAD DE DATOS - {symbol}")
    print(f"{'='*90}\n")

    # Definir las 4 combinaciones a probar
    experiments = []

    # Datos cortos disponibles
    short_h1 = f"{symbol}_H1.csv"
    short_h4 = f"{symbol}_H4.csv"
    short_d1 = f"{symbol}_D1.csv"

    # Datos 5 anos
    long_h1 = f"{symbol}_H1_5y.csv"
    long_h4 = f"{symbol}_H4_5y.csv"
    long_d1 = f"{symbol}_D1_5y.csv"

    # Verificar que existen
    data_dir = settings.DATA_DIR
    has_short = (data_dir / short_h1).exists()
    has_long = (data_dir / long_h1).exists()

    if not has_short:
        print(f"ERROR: No se encontraron datos cortos: {short_h1}")
        return
    if not has_long:
        print(f"ERROR: No se encontraron datos de 5 anos: {long_h1}")
        return

    # --- Experimento 1: Datos cortos + modelo actual ---
    print("[1/4] Datos cortos + modelo actual...")
    r1 = run_experiment(symbol, short_h1, short_h4, short_d1,
                        PARAMS_CURRENT, "cortos (~1-2y)",
                        wf_train_days=120, wf_val_days=30, wf_step_days=15)
    experiments.append(r1)
    print(f"      WR={r1['win_rate_pct']:.1f}% | Overfit={r1['overfit_gap']:.1%} | Sharpe={r1['sharpe_ratio']:.2f}")

    # --- Experimento 2: Datos cortos + modelo simplificado ---
    print("[2/4] Datos cortos + modelo simplificado...")
    r2 = run_experiment(symbol, short_h1, short_h4, short_d1,
                        PARAMS_SIMPLE, "cortos (~1-2y)",
                        wf_train_days=120, wf_val_days=30, wf_step_days=15)
    experiments.append(r2)
    print(f"      WR={r2['win_rate_pct']:.1f}% | Overfit={r2['overfit_gap']:.1%} | Sharpe={r2['sharpe_ratio']:.2f}")

    # --- Experimento 3: Datos 5y + modelo actual ---
    print("[3/4] Datos 5 anos + modelo actual...")
    r3 = run_experiment(symbol, long_h1, long_h4, long_d1,
                        PARAMS_CURRENT, "largos (5y)",
                        wf_train_days=365, wf_val_days=60, wf_step_days=30)
    experiments.append(r3)
    print(f"      WR={r3['win_rate_pct']:.1f}% | Overfit={r3['overfit_gap']:.1%} | Sharpe={r3['sharpe_ratio']:.2f}")

    # --- Experimento 4: Datos 5y + modelo simplificado ---
    print("[4/4] Datos 5 anos + modelo simplificado...")
    r4 = run_experiment(symbol, long_h1, long_h4, long_d1,
                        PARAMS_SIMPLE, "largos (5y)",
                        wf_train_days=365, wf_val_days=60, wf_step_days=30)
    experiments.append(r4)
    print(f"      WR={r4['win_rate_pct']:.1f}% | Overfit={r4['overfit_gap']:.1%} | Sharpe={r4['sharpe_ratio']:.2f}")

    # --- Tabla comparativa ---
    print(f"\n{'='*110}")
    print(f"  RESULTADOS COMPARATIVOS - {symbol}")
    print(f"{'='*110}")
    header = (
        f"{'Datos':<16} {'Modelo':<38} {'Splits':>6} "
        f"{'TrainAcc':>9} {'TestAcc':>8} {'Overfit':>8} "
        f"{'WR%':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'Trades':>7} {'Return%':>9}"
    )
    print(header)
    print("-" * 110)

    best = None
    best_score = -999

    for r in experiments:
        if "error" in r:
            print(f"  {r['data_label']:<16} ERROR: {r['error']}")
            continue

        pf = r['profit_factor']
        pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) and pf != float('inf') else str(pf)

        # Score compuesto: Sharpe penalizado por overfitting
        score = r['sharpe_ratio'] - r['overfit_gap'] * 2  # Penalizar overfitting fuerte
        marker = ""
        if score > best_score:
            best_score = score
            best = r
            marker = " <-- BEST"

        print(
            f"{r['data_label']:<16} {r['model_label']:<38} {r['n_splits']:>6} "
            f"{r['avg_train_acc']:>8.1%} {r['avg_test_acc']:>7.1%} {r['overfit_gap']:>7.1%} "
            f"{r['win_rate_pct']:>5.1f}% {pf_str:>6} {r['sharpe_ratio']:>7.2f} "
            f"{r['max_drawdown_pct']:>6.1f}% {r['total_trades']:>7} {r['total_return_pct']:>8.1f}%"
            f"{marker}"
        )

    print("=" * 110)

    # --- Analisis ---
    print(f"\n{'='*70}")
    print(f"  ANALISIS")
    print(f"{'='*70}")

    if len(experiments) >= 4:
        # Comparar efecto de datos
        ov_short_avg = np.mean([experiments[0]['overfit_gap'], experiments[1]['overfit_gap']])
        ov_long_avg = np.mean([experiments[2]['overfit_gap'], experiments[3]['overfit_gap']])
        print(f"\n  Efecto de MAS DATOS:")
        print(f"    Overfit gap promedio datos cortos: {ov_short_avg:.1%}")
        print(f"    Overfit gap promedio datos largos:  {ov_long_avg:.1%}")
        print(f"    Reduccion: {(ov_short_avg - ov_long_avg):.1%}")

        # Comparar efecto de simplificacion
        ov_complex_avg = np.mean([experiments[0]['overfit_gap'], experiments[2]['overfit_gap']])
        ov_simple_avg = np.mean([experiments[1]['overfit_gap'], experiments[3]['overfit_gap']])
        print(f"\n  Efecto de SIMPLIFICAR modelo:")
        print(f"    Overfit gap promedio modelo complejo:      {ov_complex_avg:.1%}")
        print(f"    Overfit gap promedio modelo simplificado:  {ov_simple_avg:.1%}")
        print(f"    Reduccion: {(ov_complex_avg - ov_simple_avg):.1%}")

    if best:
        print(f"\n  MEJOR COMBINACION (score ajustado por overfitting):")
        print(f"    Datos: {best['data_label']} | Modelo: {best['model_label']}")
        print(f"    WR={best['win_rate_pct']:.1f}% | PF={best['profit_factor']:.2f} | "
              f"Sharpe={best['sharpe_ratio']:.2f} | Overfit={best['overfit_gap']:.1%}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

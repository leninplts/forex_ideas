"""
Backtest Runner - Script para ejecutar backtests completos.

Flujo:
  1. Cargar datos (de MT5 o CSV)
  2. Calcular features e indicadores
  3. Entrenar modelo con walk-forward validation
  4. Generar predicciones
  5. Ejecutar backtest simulando trades
  6. Generar reporte de rendimiento
  7. Guardar modelo si aprueba criterios minimos

Uso:
  python -m forex_bot.backtest_runner
  python -m forex_bot.backtest_runner --symbol EURUSD --model xgboost
  python -m forex_bot.backtest_runner --from-csv data_cache/EURUSD_H1.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forex_bot.config import settings
from forex_bot.data.collector import DataCollector
from forex_bot.data.preprocessor import DataPreprocessor
from forex_bot.data.feature_engine import FeatureEngine
from forex_bot.models.trainer import ModelTrainer
from forex_bot.backtesting.backtester import Backtester
from forex_bot.backtesting.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Forex Bot - Backtest Runner")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Par de divisas (default: EURUSDm)")
    parser.add_argument("--model", type=str, default=None, help="Tipo de modelo: xgboost, lightgbm (default: settings)")
    parser.add_argument("--from-csv", type=str, default=None, help="Cargar datos de archivo CSV en vez de MT5")
    parser.add_argument("--bars", type=int, default=None, help="Numero de barras historicas (default: settings)")
    parser.add_argument("--save-model", action="store_true", help="Guardar modelo si aprueba criterios")
    parser.add_argument("--verbose", action="store_true", help="Logging detallado")
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT,
    )


def load_data(symbol: str, from_csv: str = None, bars: int = None) -> pd.DataFrame:
    """Cargar datos historicos de MT5 o desde CSV."""
    if from_csv:
        filepath = Path(from_csv)
        if not filepath.exists():
            logger.error("Archivo CSV no encontrado: %s", filepath)
            sys.exit(1)

        df = pd.read_csv(filepath, index_col="time", parse_dates=True)
        logger.info("Datos cargados de CSV: %s (%d barras)", filepath, len(df))
        return df

    # Intentar descargar de MT5
    collector = DataCollector()
    if not collector.connect():
        logger.error("No se pudo conectar a MT5. Usa --from-csv para cargar datos desde archivo.")
        sys.exit(1)

    try:
        df = collector.get_historical_data(symbol, settings.TIMEFRAME_PRIMARY, bars)
        if df is None or df.empty:
            logger.error("No se obtuvieron datos de MT5 para %s", symbol)
            sys.exit(1)

        # Guardar a CSV para futuro uso
        collector.save_data(df, f"{symbol}_{settings.TIMEFRAME_PRIMARY}")
        return df
    finally:
        collector.disconnect()


def load_htf_data(symbol: str, pp: DataPreprocessor) -> tuple:
    """
    Cargar datos H4 y D1 para features multi-timeframe.

    Args:
        symbol: Par de divisas (ej: "EURUSDm")
        pp: DataPreprocessor para limpieza

    Returns:
        Tuple (df_h4, df_d1) o (None, None) si no se encuentran.
    """
    data_dir = settings.DATA_DIR
    df_h4, df_d1 = None, None

    h4_path = data_dir / f"{symbol}_{settings.TIMEFRAME_HIGHER}.csv"
    d1_path = data_dir / f"{symbol}_{settings.TIMEFRAME_DAILY}.csv"

    if h4_path.exists():
        df_h4 = pd.read_csv(h4_path, index_col="time", parse_dates=True)
        df_h4 = pp.clean_data(df_h4)
        logger.info("Datos H4 cargados: %d barras", len(df_h4))
    else:
        logger.warning("No se encontro %s, sin features H4", h4_path)

    if d1_path.exists():
        df_d1 = pd.read_csv(d1_path, index_col="time", parse_dates=True)
        df_d1 = pp.clean_data(df_d1)
        logger.info("Datos D1 cargados: %d barras", len(df_d1))
    else:
        logger.warning("No se encontro %s, sin features D1", d1_path)

    return df_h4, df_d1


def run_backtest(
    symbol: str = "EURUSDm",
    model_type: str = None,
    from_csv: str = None,
    bars: int = None,
    save_model: bool = False,
) -> dict:
    """
    Ejecutar backtest completo con multi-timeframe y target binario.

    Args:
        symbol: Par de divisas
        model_type: Tipo de modelo ML
        from_csv: Ruta a CSV con datos
        bars: Numero de barras
        save_model: Guardar modelo si aprueba

    Returns:
        Dict con resultados del backtest.
    """
    model_type = model_type or settings.MODEL_TYPE

    print(f"\n{'='*60}")
    print(f"  FOREX BOT - BACKTEST RUNNER")
    print(f"  Symbol: {symbol} | Model: {model_type}")
    print(f"{'='*60}\n")

    # 1. Cargar datos
    print("[1/7] Cargando datos...")
    df = load_data(symbol, from_csv, bars)

    # 2. Limpiar datos
    print("[2/7] Limpiando y preparando datos...")
    pp = DataPreprocessor()
    df = pp.clean_data(df)

    # 3. Calcular features H1
    print("[3/7] Calculando indicadores y features...")
    fe = FeatureEngine()
    df = fe.add_all_features(df)

    # 4. Agregar features multi-timeframe (H4 + D1)
    print("[4/7] Agregando features multi-timeframe (H4 + D1)...")
    df_h4, df_d1 = load_htf_data(symbol, pp)

    if df_h4 is not None:
        df = fe.add_higher_timeframe_features(df, df_h4, suffix="h4")
        print(f"       +features H4 agregadas")
    if df_d1 is not None:
        df = fe.add_higher_timeframe_features(df, df_d1, suffix="d1")
        print(f"       +features D1 agregadas")

    # Crear target
    pip_size = settings.PIP_SIZE.get(symbol, 0.0001)
    df["target"] = fe.create_target(df, pip_size=pip_size)

    n_features = len(fe.get_feature_columns(df))
    print(f"       {n_features} features totales")

    # 5. Entrenar modelo
    print("[5/7] Entrenando modelo...")
    trainer = ModelTrainer(model_type=model_type)
    X, y = trainer.prepare_features(df)

    # Split: 70% train+val, 30% test (para backtest)
    split_idx = int(len(X) * 0.7)
    X_train_full = X.iloc[:split_idx]
    y_train_full = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Dentro del train, split en train/val para early stopping
    val_split = int(len(X_train_full) * 0.85)
    X_train = X_train_full.iloc[:val_split]
    y_train = y_train_full.iloc[:val_split]
    X_val = X_train_full.iloc[val_split:]
    y_val = y_train_full.iloc[val_split:]

    trainer.train(X_train, y_train, X_val, y_val)

    # Evaluar en test set
    eval_metrics = trainer.evaluate(X_test, y_test)
    print(f"       Model accuracy: {eval_metrics['accuracy']:.2%}")
    print(f"       Model F1:       {eval_metrics['f1_weighted']:.2%}")
    print(f"       Model AUC:      {eval_metrics['auc']:.4f}")
    print(f"       Model Brier:    {eval_metrics['brier']:.4f}")
    print(f"       Precision:      {eval_metrics['precision']:.2%}")
    print(f"       Recall:         {eval_metrics['recall']:.2%}")

    # Feature importance
    fi = trainer.get_feature_importance()
    print(f"       Top 5 features:")
    for _, row in fi.head(5).iterrows():
        print(f"         - {row['feature']}: {row['importance']:.4f}")

    # 6. Generar senales para backtest
    print("[6/7] Ejecutando backtest...")

    # Usar probabilidades para generar senales con umbral de confianza
    y_proba = trainer.model.predict_proba(X_test)
    threshold = settings.CONFIDENCE_THRESHOLD

    # Crear Series de senales alineada con df completo
    signals = pd.Series(0, index=df.index, dtype=int)

    is_binary = y_proba.shape[1] == 2

    for i, idx in enumerate(X_test.index):
        if idx not in df.index:
            continue
        pos = df.index.get_loc(idx)

        if is_binary:
            # Binario: proba[0]=DOWN, proba[1]=UP
            prob_up = y_proba[i, 1]
            if prob_up > threshold:
                signals.iloc[pos] = 1     # BUY
            elif prob_up < (1 - threshold):
                signals.iloc[pos] = -1    # SELL
            # else: 0 = HOLD
        else:
            # Ternario: proba[0]=SELL, proba[1]=HOLD, proba[2]=BUY
            pred = int(np.argmax(y_proba[i]))
            conf = y_proba[i, pred]
            if conf > threshold:
                signal_map = {0: -1, 1: 0, 2: 1}
                signals.iloc[pos] = signal_map.get(pred, 0)

    n_buy = (signals == 1).sum()
    n_sell = (signals == -1).sum()
    n_hold = (signals == 0).sum()
    print(f"       Senales: BUY={n_buy} | SELL={n_sell} | HOLD={n_hold}")

    # Ejecutar backtest
    bt = Backtester(initial_balance=settings.BACKTEST_INITIAL_BALANCE)
    result = bt.run(df, signals, symbol=symbol)
    metrics = result["metrics"]

    # 7. Reporte
    print("[7/7] Generando reporte...\n")
    PerformanceMetrics.print_report(metrics)

    # Evaluar criterios minimos
    passed = evaluate_criteria(metrics)

    # Guardar modelo si aprueba
    if save_model and passed:
        model_path = trainer.save_model()
        print(f"\nModelo guardado en: {model_path}")
    elif save_model and not passed:
        print("\nModelo NO guardado: no aprobo criterios minimos.")

    return {
        "metrics": metrics,
        "trades": result["trades"],
        "equity_curve": result["equity_curve"],
        "model_eval": eval_metrics,
        "feature_importance": fi,
        "passed_criteria": passed,
    }


def evaluate_criteria(metrics: dict) -> bool:
    """
    Evaluar si el backtest pasa los criterios minimos.

    Criterios (de settings):
      - Profit Factor > 1.3
      - Sharpe Ratio > 0.8
      - Max Drawdown < 25%
      - Win Rate > 45%
      - Total trades > 100
    """
    print("\n--- EVALUACION DE CRITERIOS ---")
    all_passed = True

    checks = [
        (
            "Profit Factor",
            metrics.get("profit_factor", 0),
            settings.BACKTEST_MIN_PROFIT_FACTOR,
            ">=",
        ),
        (
            "Sharpe Ratio",
            metrics.get("sharpe_ratio", 0),
            settings.BACKTEST_MIN_SHARPE,
            ">=",
        ),
        (
            "Max Drawdown",
            metrics.get("max_drawdown", 0),
            settings.BACKTEST_MAX_DRAWDOWN,
            "<=",
        ),
        (
            "Win Rate",
            metrics.get("win_rate", 0),
            settings.BACKTEST_MIN_WIN_RATE,
            ">=",
        ),
        (
            "Total Trades",
            metrics.get("total_trades", 0),
            settings.BACKTEST_MIN_TRADES,
            ">=",
        ),
    ]

    for name, value, threshold, op in checks:
        # Manejar "inf" como string
        if isinstance(value, str):
            value = float("inf") if value == "inf" else 0

        if op == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] {name}: {value:.4f} (min: {threshold})")

    if all_passed:
        print("\n  RESULTADO: APROBADO - Listo para demo trading")
    else:
        print("\n  RESULTADO: NO APROBADO - Ajustar parametros y repetir")

    return all_passed


def main():
    args = parse_args()
    setup_logging(args.verbose)

    run_backtest(
        symbol=args.symbol,
        model_type=args.model,
        from_csv=args.from_csv,
        bars=args.bars,
        save_model=args.save_model,
    )


if __name__ == "__main__":
    main()

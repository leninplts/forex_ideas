"""
Benchmark Strategies - Compara el modelo ML contra estrategias ingenuas.

Objetivo: Establecer una linea base real. Si el modelo ML no supera a una
estrategia simple de MA crossover, entonces el ML no esta aportando valor.

Estrategias benchmark:
  1. Random: senales aleatorias BUY/SELL con la misma frecuencia que el ML
  2. MA Crossover: EMA(9) cruza EMA(21) - trend following basico
  3. RSI Mean Reversion: Comprar en oversold (<30), vender en overbought (>70)
  4. Always Buy: comprar en cada vela (baseline de mercado alcista)
  5. ML Model: el modelo actual de XGBoost para comparar

Uso:
  python scripts/benchmark_strategies.py
  python scripts/benchmark_strategies.py --symbol EURUSDm
  python scripts/benchmark_strategies.py --symbol GBPUSDm --runs 50
"""
import argparse
import logging
import sys
from pathlib import Path

# Agregar directorio raiz al path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import pandas_ta as ta

from forex_bot.config import settings
from forex_bot.data.preprocessor import DataPreprocessor
from forex_bot.data.feature_engine import FeatureEngine
from forex_bot.models.trainer import ModelTrainer
from forex_bot.backtesting.backtester import Backtester
from forex_bot.backtesting.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


# =====================================================================
# Estrategias Benchmark
# =====================================================================

def strategy_random(df: pd.DataFrame, n_signals: int = None, seed: int = 42) -> pd.Series:
    """
    Estrategia aleatoria: genera senales BUY/SELL al azar.
    Simula la misma frecuencia de senales que el modelo ML.
    
    Args:
        df: DataFrame OHLCV
        n_signals: Numero de senales a generar (si None, usa ~10% de las velas)
        seed: Semilla para reproducibilidad
        
    Returns:
        Series con senales {1: BUY, -1: SELL, 0: HOLD}
    """
    rng = np.random.RandomState(seed)
    signals = pd.Series(0, index=df.index, dtype=int)
    
    if n_signals is None:
        n_signals = max(int(len(df) * 0.10), 50)
    
    # Seleccionar indices aleatorios para senales
    signal_indices = rng.choice(len(df), size=min(n_signals, len(df)), replace=False)
    
    for idx in signal_indices:
        signals.iloc[idx] = rng.choice([1, -1])
    
    return signals


def strategy_ma_crossover(
    df: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21,
) -> pd.Series:
    """
    Estrategia de cruce de medias moviles exponenciales.
    BUY cuando EMA rapida cruza por encima de EMA lenta.
    SELL cuando EMA rapida cruza por debajo de EMA lenta.
    
    Args:
        df: DataFrame OHLCV
        fast_period: Periodo de EMA rapida
        slow_period: Periodo de EMA lenta
        
    Returns:
        Series con senales {1: BUY, -1: SELL, 0: HOLD}
    """
    ema_fast = ta.ema(df["close"], length=fast_period)
    ema_slow = ta.ema(df["close"], length=slow_period)
    
    signals = pd.Series(0, index=df.index, dtype=int)
    
    if ema_fast is None or ema_slow is None:
        return signals
    
    # Cruce: detectar cuando fast cruza slow
    prev_fast = ema_fast.shift(1)
    prev_slow = ema_slow.shift(1)
    
    # BUY: fast cruza por encima de slow
    buy_cross = (prev_fast <= prev_slow) & (ema_fast > ema_slow)
    signals[buy_cross] = 1
    
    # SELL: fast cruza por debajo de slow
    sell_cross = (prev_fast >= prev_slow) & (ema_fast < ema_slow)
    signals[sell_cross] = -1
    
    return signals


def strategy_rsi_reversion(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
) -> pd.Series:
    """
    Estrategia de mean reversion con RSI.
    BUY cuando RSI sale de zona oversold (<30 -> >30).
    SELL cuando RSI sale de zona overbought (>70 -> <70).
    
    Args:
        df: DataFrame OHLCV
        period: Periodo RSI
        oversold: Nivel oversold
        overbought: Nivel overbought
        
    Returns:
        Series con senales {1: BUY, -1: SELL, 0: HOLD}
    """
    rsi = ta.rsi(df["close"], length=period)
    signals = pd.Series(0, index=df.index, dtype=int)
    
    if rsi is None:
        return signals
    
    prev_rsi = rsi.shift(1)
    
    # BUY: RSI cruza de abajo a arriba del nivel oversold
    buy_signal = (prev_rsi < oversold) & (rsi >= oversold)
    signals[buy_signal] = 1
    
    # SELL: RSI cruza de arriba a abajo del nivel overbought
    sell_signal = (prev_rsi > overbought) & (rsi <= overbought)
    signals[sell_signal] = -1
    
    return signals


def strategy_always_buy(df: pd.DataFrame, interval: int = 24) -> pd.Series:
    """
    Estrategia naive: comprar cada N velas (simula DCA / mercado alcista).
    
    Args:
        df: DataFrame OHLCV
        interval: Intervalo entre senales de compra (velas H1)
        
    Returns:
        Series con senales {1: BUY, 0: HOLD}
    """
    signals = pd.Series(0, index=df.index, dtype=int)
    
    for i in range(0, len(df), interval):
        signals.iloc[i] = 1
    
    return signals


def strategy_ml_model(
    df: pd.DataFrame,
    df_h4: pd.DataFrame = None,
    df_d1: pd.DataFrame = None,
    symbol: str = "EURUSDm",
    model_type: str = None,
) -> pd.Series:
    """
    Estrategia del modelo ML actual (XGBoost/LightGBM).
    Replica el pipeline del backtest_runner.
    
    Args:
        df: DataFrame OHLCV H1
        df_h4: DataFrame H4 (multi-timeframe)
        df_d1: DataFrame D1 (multi-timeframe)
        symbol: Par de divisas
        model_type: Tipo de modelo
        
    Returns:
        Series con senales {1: BUY, -1: SELL, 0: HOLD}
    """
    model_type = model_type or settings.MODEL_TYPE
    
    # Calcular features
    fe = FeatureEngine()
    df_feat = fe.add_all_features(df)
    
    if df_h4 is not None:
        df_feat = fe.add_higher_timeframe_features(df_feat, df_h4, suffix="h4")
    if df_d1 is not None:
        df_feat = fe.add_higher_timeframe_features(df_feat, df_d1, suffix="d1")
    
    # Crear target
    pip_size = settings.PIP_SIZE.get(symbol, 0.0001)
    df_feat["target"] = fe.create_target(df_feat, pip_size=pip_size)
    
    # Entrenar modelo
    trainer = ModelTrainer(model_type=model_type)
    X, y = trainer.prepare_features(df_feat)
    
    # Split 70/30
    split_idx = int(len(X) * 0.7)
    X_train_full = X.iloc[:split_idx]
    y_train_full = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    
    # Dentro del train, split en train/val para early stopping
    val_split = int(len(X_train_full) * 0.85)
    X_train = X_train_full.iloc[:val_split]
    y_train = y_train_full.iloc[:val_split]
    X_val = X_train_full.iloc[val_split:]
    y_val = y_train_full.iloc[val_split:]
    
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Generar senales
    y_proba = trainer.model.predict_proba(X_test)
    threshold = settings.CONFIDENCE_THRESHOLD
    
    signals = pd.Series(0, index=df.index, dtype=int)
    is_binary = y_proba.shape[1] == 2
    
    for i, idx in enumerate(X_test.index):
        if idx not in df.index:
            continue
        pos = df.index.get_loc(idx)
        
        if is_binary:
            prob_up = y_proba[i, 1]
            if prob_up > threshold:
                signals.iloc[pos] = 1
            elif prob_up < (1 - threshold):
                signals.iloc[pos] = -1
        else:
            pred = int(np.argmax(y_proba[i]))
            conf = y_proba[i, pred]
            if conf > threshold:
                signal_map = {0: -1, 1: 0, 2: 1}
                signals.iloc[pos] = signal_map.get(pred, 0)
    
    return signals


# =====================================================================
# Runner principal
# =====================================================================

def load_data(symbol: str) -> tuple:
    """Cargar datos H1, H4, D1 desde CSV cache."""
    data_dir = settings.DATA_DIR
    pp = DataPreprocessor()
    
    # H1 (obligatorio)
    h1_path = data_dir / f"{symbol}_H1.csv"
    if not h1_path.exists():
        print(f"ERROR: No se encontro {h1_path}")
        sys.exit(1)
    
    df_h1 = pd.read_csv(h1_path, index_col="time", parse_dates=True)
    df_h1 = pp.clean_data(df_h1)
    
    # H4 (opcional)
    h4_path = data_dir / f"{symbol}_H4.csv"
    df_h4 = None
    if h4_path.exists():
        df_h4 = pd.read_csv(h4_path, index_col="time", parse_dates=True)
        df_h4 = pp.clean_data(df_h4)
    
    # D1 (opcional)
    d1_path = data_dir / f"{symbol}_D1.csv"
    df_d1 = None
    if d1_path.exists():
        df_d1 = pd.read_csv(d1_path, index_col="time", parse_dates=True)
        df_d1 = pp.clean_data(df_d1)
    
    return df_h1, df_h4, df_d1


def run_benchmark(
    df: pd.DataFrame,
    signals: pd.Series,
    strategy_name: str,
    symbol: str = "EURUSDm",
) -> dict:
    """Ejecutar backtest para una estrategia y retornar metricas."""
    bt = Backtester(initial_balance=settings.BACKTEST_INITIAL_BALANCE)
    result = bt.run(df, signals, symbol=symbol)
    metrics = result["metrics"]
    
    n_buy = (signals == 1).sum()
    n_sell = (signals == -1).sum()
    
    return {
        "strategy": strategy_name,
        "total_trades": metrics["total_trades"],
        "signals_buy": int(n_buy),
        "signals_sell": int(n_sell),
        "win_rate_pct": metrics["win_rate_pct"],
        "profit_factor": metrics["profit_factor"],
        "total_return_pct": metrics["total_return_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "final_balance": metrics["final_balance"],
        "expectancy": metrics["expectancy"],
        "avg_trade_duration_bars": metrics["avg_trade_duration_bars"],
    }


def run_all_benchmarks(
    symbol: str = "EURUSDm",
    n_random_runs: int = 20,
    model_type: str = None,
):
    """
    Ejecutar todas las estrategias benchmark y comparar.
    
    Args:
        symbol: Par de divisas
        n_random_runs: Numero de ejecuciones random (para promediar)
        model_type: Tipo de modelo ML
    """
    print(f"\n{'='*70}")
    print(f"  BENCHMARK DE ESTRATEGIAS - {symbol}")
    print(f"{'='*70}\n")
    
    # Cargar datos
    print("Cargando datos...")
    df_h1, df_h4, df_d1 = load_data(symbol)
    print(f"  H1: {len(df_h1)} barras ({df_h1.index[0]} -> {df_h1.index[-1]})")
    if df_h4 is not None:
        print(f"  H4: {len(df_h4)} barras")
    if df_d1 is not None:
        print(f"  D1: {len(df_d1)} barras")
    
    results = []
    
    # --- 1. Random (promedio de N ejecuciones) ---
    print(f"\n[1/5] Estrategia Random ({n_random_runs} ejecuciones)...")
    random_results = []
    for seed in range(n_random_runs):
        signals = strategy_random(df_h1, seed=seed)
        res = run_benchmark(df_h1, signals, "Random", symbol)
        random_results.append(res)
    
    # Promediar resultados random
    avg_random = {
        "strategy": f"Random (avg {n_random_runs} runs)",
        "total_trades": np.mean([r["total_trades"] for r in random_results]),
        "signals_buy": np.mean([r["signals_buy"] for r in random_results]),
        "signals_sell": np.mean([r["signals_sell"] for r in random_results]),
        "win_rate_pct": np.mean([r["win_rate_pct"] for r in random_results]),
        "profit_factor": np.mean([r["profit_factor"] for r in random_results]),
        "total_return_pct": np.mean([r["total_return_pct"] for r in random_results]),
        "max_drawdown_pct": np.mean([r["max_drawdown_pct"] for r in random_results]),
        "sharpe_ratio": np.mean([r["sharpe_ratio"] for r in random_results]),
        "final_balance": np.mean([r["final_balance"] for r in random_results]),
        "expectancy": np.mean([r["expectancy"] for r in random_results]),
        "avg_trade_duration_bars": np.mean([r["avg_trade_duration_bars"] for r in random_results]),
    }
    results.append(avg_random)
    print(f"  Win rate promedio: {avg_random['win_rate_pct']:.1f}% | "
          f"Return: {avg_random['total_return_pct']:.1f}% | "
          f"Sharpe: {avg_random['sharpe_ratio']:.2f}")
    
    # --- 2. MA Crossover ---
    print("\n[2/5] Estrategia MA Crossover (EMA 9/21)...")
    signals_ma = strategy_ma_crossover(df_h1, fast_period=9, slow_period=21)
    res_ma = run_benchmark(df_h1, signals_ma, "MA Crossover (9/21)", symbol)
    results.append(res_ma)
    print(f"  Trades: {res_ma['total_trades']} | Win rate: {res_ma['win_rate_pct']:.1f}% | "
          f"Return: {res_ma['total_return_pct']:.1f}% | Sharpe: {res_ma['sharpe_ratio']:.2f}")
    
    # --- 3. RSI Mean Reversion ---
    print("\n[3/5] Estrategia RSI Mean Reversion (30/70)...")
    signals_rsi = strategy_rsi_reversion(df_h1)
    res_rsi = run_benchmark(df_h1, signals_rsi, "RSI Reversion (30/70)", symbol)
    results.append(res_rsi)
    print(f"  Trades: {res_rsi['total_trades']} | Win rate: {res_rsi['win_rate_pct']:.1f}% | "
          f"Return: {res_rsi['total_return_pct']:.1f}% | Sharpe: {res_rsi['sharpe_ratio']:.2f}")
    
    # --- 4. Always Buy ---
    print("\n[4/5] Estrategia Always Buy (cada 24 velas)...")
    signals_buy = strategy_always_buy(df_h1, interval=24)
    res_buy = run_benchmark(df_h1, signals_buy, "Always Buy (24h)", symbol)
    results.append(res_buy)
    print(f"  Trades: {res_buy['total_trades']} | Win rate: {res_buy['win_rate_pct']:.1f}% | "
          f"Return: {res_buy['total_return_pct']:.1f}% | Sharpe: {res_buy['sharpe_ratio']:.2f}")
    
    # --- 5. ML Model ---
    print(f"\n[5/5] Estrategia ML ({model_type or settings.MODEL_TYPE})...")
    signals_ml = strategy_ml_model(df_h1, df_h4, df_d1, symbol, model_type)
    res_ml = run_benchmark(df_h1, signals_ml, f"ML ({model_type or settings.MODEL_TYPE})", symbol)
    results.append(res_ml)
    print(f"  Trades: {res_ml['total_trades']} | Win rate: {res_ml['win_rate_pct']:.1f}% | "
          f"Return: {res_ml['total_return_pct']:.1f}% | Sharpe: {res_ml['sharpe_ratio']:.2f}")
    
    # --- Tabla comparativa ---
    print_comparison_table(results)
    
    # --- Analisis ---
    print_analysis(results)
    
    return results


def print_comparison_table(results: list):
    """Imprimir tabla comparativa de todas las estrategias."""
    print(f"\n{'='*90}")
    print(f"  TABLA COMPARATIVA")
    print(f"{'='*90}")
    
    header = f"{'Estrategia':<30} {'Trades':>7} {'WinRate':>8} {'PF':>7} {'Return%':>9} {'MaxDD%':>8} {'Sharpe':>7} {'Final$':>8}"
    print(header)
    print("-" * 90)
    
    for r in results:
        pf = r['profit_factor']
        pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) and pf != float('inf') else "inf"
        
        print(
            f"{r['strategy']:<30} "
            f"{r['total_trades']:>7.0f} "
            f"{r['win_rate_pct']:>7.1f}% "
            f"{pf_str:>7} "
            f"{r['total_return_pct']:>8.1f}% "
            f"{r['max_drawdown_pct']:>7.1f}% "
            f"{r['sharpe_ratio']:>7.2f} "
            f"${r['final_balance']:>7.2f}"
        )
    
    print("=" * 90)


def print_analysis(results: list):
    """Analizar resultados y dar conclusiones."""
    print(f"\n{'='*70}")
    print(f"  ANALISIS")
    print(f"{'='*70}")
    
    ml_result = [r for r in results if "ML" in r["strategy"]]
    if not ml_result:
        print("  No se encontro resultado del modelo ML.")
        return
    
    ml = ml_result[0]
    benchmarks = [r for r in results if "ML" not in r["strategy"]]
    
    # Comparar ML vs cada benchmark
    for bench in benchmarks:
        wr_diff = ml["win_rate_pct"] - bench["win_rate_pct"]
        ret_diff = ml["total_return_pct"] - bench["total_return_pct"]
        sharpe_diff = ml["sharpe_ratio"] - bench["sharpe_ratio"]
        
        status = "MEJOR" if ret_diff > 0 else "PEOR"
        print(f"\n  ML vs {bench['strategy']}:")
        print(f"    Win Rate: {wr_diff:+.1f}% | Return: {ret_diff:+.1f}% | Sharpe: {sharpe_diff:+.2f} -> {status}")
    
    # Conclusion general
    best_by_sharpe = max(results, key=lambda r: r["sharpe_ratio"])
    best_by_return = max(results, key=lambda r: r["total_return_pct"])
    
    print(f"\n  Mejor estrategia por Sharpe: {best_by_sharpe['strategy']} ({best_by_sharpe['sharpe_ratio']:.2f})")
    print(f"  Mejor estrategia por Return: {best_by_return['strategy']} ({best_by_return['total_return_pct']:.1f}%)")
    
    if "ML" not in best_by_sharpe["strategy"]:
        print(f"\n  ALERTA: El modelo ML NO es la mejor estrategia por Sharpe.")
        print(f"  El ML necesita mejoras significativas para justificar su complejidad.")
    else:
        print(f"\n  OK: El modelo ML supera a los benchmarks por Sharpe.")
    
    if ml["win_rate_pct"] < 50:
        print(f"\n  ALERTA: El ML tiene win rate < 50% ({ml['win_rate_pct']:.1f}%).")
        print(f"  La rentabilidad depende unicamente del ratio R:R favorable.")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark de estrategias de trading")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Par de divisas")
    parser.add_argument("--runs", type=int, default=20, help="Numero de ejecuciones random")
    parser.add_argument("--model", type=str, default=None, help="Tipo de modelo ML")
    parser.add_argument("--all-symbols", action="store_true", help="Correr para todos los pares")
    parser.add_argument("--verbose", action="store_true", help="Logging detallado")
    args = parser.parse_args()
    
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format=settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)
    
    if args.all_symbols:
        all_results = {}
        for symbol in settings.SYMBOLS:
            try:
                all_results[symbol] = run_all_benchmarks(symbol, args.runs, args.model)
            except Exception as e:
                print(f"\nERROR en {symbol}: {e}")
                continue
        
        # Resumen global
        print(f"\n{'='*70}")
        print(f"  RESUMEN GLOBAL - TODOS LOS PARES")
        print(f"{'='*70}")
        for symbol, results in all_results.items():
            ml_res = [r for r in results if "ML" in r["strategy"]]
            if ml_res:
                ml = ml_res[0]
                print(f"  {symbol}: WR={ml['win_rate_pct']:.1f}% | Return={ml['total_return_pct']:.1f}% | Sharpe={ml['sharpe_ratio']:.2f}")
    else:
        run_all_benchmarks(args.symbol, args.runs, args.model)


if __name__ == "__main__":
    main()

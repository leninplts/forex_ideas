"""
Entrenar y guardar el modelo de produccion.
Este modelo es el que usa el bot en vivo/demo.

Uso:
  python scripts/train_production_model.py
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import os
import logging
import pandas as pd
from forex_bot.config import settings
from forex_bot.data.preprocessor import DataPreprocessor
from forex_bot.data.feature_engine import FeatureEngine
from forex_bot.models.trainer import ModelTrainer

logging.basicConfig(level=logging.WARNING)


def main():
    pp = DataPreprocessor()
    fe = FeatureEngine()
    data_dir = settings.DATA_DIR

    print("Entrenando modelo de produccion...")
    print(f"MODELS_DIR: {settings.MODELS_DIR}")

    # Buscar datos: preferir 5y, fallback a normales
    symbols_data = []
    for symbol in settings.SYMBOLS:
        h1_5y = data_dir / f"{symbol}_H1_5y.csv"
        h1 = data_dir / f"{symbol}_H1.csv"
        h1_path = h1_5y if h1_5y.exists() else h1 if h1.exists() else None

        if h1_path is None:
            print(f"  SKIP {symbol}: no hay datos H1")
            continue

        h4_5y = data_dir / f"{symbol}_H4_5y.csv"
        h4 = data_dir / f"{symbol}_H4.csv"
        h4_path = h4_5y if h4_5y.exists() else h4 if h4.exists() else None

        d1_5y = data_dir / f"{symbol}_D1_5y.csv"
        d1 = data_dir / f"{symbol}_D1.csv"
        d1_path = d1_5y if d1_5y.exists() else d1 if d1.exists() else None

        symbols_data.append((symbol, h1_path, h4_path, d1_path))
        print(f"  {symbol}: {h1_path.name}")

    if not symbols_data:
        print("ERROR: No hay datos para entrenar")
        sys.exit(1)

    # Usar el primer simbolo con datos para entrenar
    symbol, h1_path, h4_path, d1_path = symbols_data[0]
    print(f"\nEntrenando con {symbol} ({h1_path.name})...")

    df = pd.read_csv(h1_path, index_col="time", parse_dates=True)
    df = pp.clean_data(df)
    df = fe.add_all_features(df)

    if h4_path and h4_path.exists():
        h4 = pd.read_csv(h4_path, index_col="time", parse_dates=True)
        h4 = pp.clean_data(h4)
        df = fe.add_higher_timeframe_features(df, h4, suffix="h4")

    if d1_path and d1_path.exists():
        d1 = pd.read_csv(d1_path, index_col="time", parse_dates=True)
        d1 = pp.clean_data(d1)
        df = fe.add_higher_timeframe_features(df, d1, suffix="d1")

    pip_size = settings.PIP_SIZE.get(symbol, 0.0001)
    df["target"] = fe.create_target(df, pip_size=pip_size)

    trainer = ModelTrainer(model_type="xgboost")
    X, y = trainer.prepare_features(df)

    split = int(len(X) * 0.85)
    es = int(split * 0.85)
    X_tr, y_tr = X.iloc[:es], y.iloc[:es]
    X_es, y_es = X.iloc[es:split], y.iloc[es:split]

    print(f"  Train: {len(X_tr)} | Val: {len(X_es)} | Features: {X_tr.shape[1]}")

    trainer.train_ensemble(X_tr, y_tr, X_es, y_es)

    path = trainer.save_model()
    size_kb = os.path.getsize(path) / 1024
    meta_path = path.with_suffix(".meta.json")

    print(f"\nModelo guardado: {path} ({size_kb:.1f} KB)")
    print(f"Metadata: {meta_path}")
    print("LISTO - modelo de produccion generado.")


if __name__ == "__main__":
    main()

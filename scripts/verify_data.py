"""
Verificar los datos descargados en data_cache/.
Ejecutar: python scripts/verify_data.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from forex_bot.config import settings


def main():
    data_dir = settings.DATA_DIR
    files = sorted(data_dir.glob("*.csv"))

    if not files:
        print("No hay archivos en data_cache/")
        return

    print("RESUMEN DE DATOS DESCARGADOS")
    print("=" * 80)
    header = f"{'Archivo':<22} {'Barras':>7} {'Desde':>12} {'Hasta':>12} {'Dias':>6} {'Size':>8}"
    print(header)
    print("-" * 80)

    total_bars = 0
    total_size = 0

    for f in files:
        df = pd.read_csv(f, index_col="time", parse_dates=True)
        dias = (df.index[-1] - df.index[0]).days
        size_kb = f.stat().st_size / 1024
        d1 = df.index[0].strftime("%Y-%m-%d")
        d2 = df.index[-1].strftime("%Y-%m-%d")

        print(f"{f.name:<22} {len(df):>7,} {d1:>12} {d2:>12} {dias:>6} {size_kb:>6.0f} KB")
        total_bars += len(df)
        total_size += size_kb

    print("-" * 80)
    print(f"{'TOTAL':<22} {total_bars:>7,} {'':>12} {'':>12} {'':>6} {total_size:>6.0f} KB")
    print()

    # Verificar integridad
    print("VERIFICACION DE INTEGRIDAD")
    print("-" * 40)
    issues = 0

    for f in files:
        df = pd.read_csv(f, index_col="time", parse_dates=True)
        name = f.stem

        # NaN en OHLC
        ohlc_nan = df[["open", "high", "low", "close"]].isna().sum().sum()
        if ohlc_nan > 0:
            print(f"  [WARN] {name}: {ohlc_nan} NaN en OHLC")
            issues += 1

        # Duplicados
        dups = df.index.duplicated().sum()
        if dups > 0:
            print(f"  [WARN] {name}: {dups} timestamps duplicados")
            issues += 1

        # Orden cronologico
        if not df.index.is_monotonic_increasing:
            print(f"  [WARN] {name}: datos no estan en orden cronologico")
            issues += 1

        # OHLC consistencia
        bad_ohlc = (
            (df["high"] < df[["open", "close"]].max(axis=1)) |
            (df["low"] > df[["open", "close"]].min(axis=1))
        ).sum()
        if bad_ohlc > 0:
            print(f"  [WARN] {name}: {bad_ohlc} velas con OHLC inconsistente")
            issues += 1

    if issues == 0:
        print("  [OK] Todos los archivos pasaron las verificaciones")
    else:
        print(f"\n  {issues} problemas encontrados")


if __name__ == "__main__":
    main()

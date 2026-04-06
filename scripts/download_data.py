"""
Descarga datos historicos reales de MT5 y los guarda en CSV.
Ejecutar: python scripts/download_data.py

Descarga 2 anios de datos para los 3 pares configurados,
en 3 timeframes (H1, H4, D1). Limpia los datos con el preprocessor
antes de guardarlos.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from forex_bot.data.collector import DataCollector
from forex_bot.data.preprocessor import DataPreprocessor
from forex_bot.config import settings


def main():
    print("=" * 60)
    print("DESCARGA DE DATOS HISTORICOS - MT5")
    print("=" * 60)

    collector = DataCollector()
    preprocessor = DataPreprocessor()

    if not collector.connect():
        print("ERROR: No se pudo conectar a MT5")
        print("Verificar que MetaTrader 5 esta abierto y mt5_config.py tiene las credenciales correctas")
        return

    # Info de cuenta
    info = collector.get_account_info()
    print(f"Conectado: {info['login']} @ {info['server']} | Balance: ${info['balance']:.2f}")
    print()

    # Barras a descargar por timeframe (~5 anios)
    bars_per_tf = {
        "H1": 30000,    # ~5 anios de H1 (24 * 260 dias trading * 5 ~ 31200)
        "H4": 8000,     # ~5 anios de H4
        "D1": 1500,     # ~5 anios de D1 + extra para lookback SMA200
    }

    timeframes = [
        settings.TIMEFRAME_PRIMARY,   # H1
        settings.TIMEFRAME_HIGHER,    # H4
        settings.TIMEFRAME_DAILY,     # D1
    ]

    total_files = 0

    for symbol in settings.SYMBOLS:
        print(f"--- {symbol} ---")

        for tf in timeframes:
            bars = bars_per_tf.get(tf, 5000)

            # Descargar
            df = collector.get_historical_data(symbol, tf, bars=bars)
            if df is None:
                print(f"  {tf}: ERROR al descargar")
                continue

            # Limpiar
            df = preprocessor.clean_data(df)

            # Guardar
            filename = f"{symbol}_{tf}"
            filepath = collector.save_data(df, filename)

            days = (df.index[-1] - df.index[0]).days
            print(f"  {tf}: {len(df)} barras | {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')} | ~{days} dias | -> {filepath.name}")
            total_files += 1

        print()

    collector.disconnect()

    print("=" * 60)
    print(f"COMPLETADO: {total_files} archivos guardados en {settings.DATA_DIR}")
    print("=" * 60)

    # Listar archivos
    print()
    print("Archivos generados:")
    for f in sorted(settings.DATA_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()

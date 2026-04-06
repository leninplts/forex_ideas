"""
VALIDACION - Conexion a MetaTrader 5
Ejecutar: python validation/validate_mt5_connection.py

Verifica:
  1. MT5 esta instalado y corriendo
  2. Credenciales de mt5_config.py son correctas
  3. Se puede obtener info de cuenta
  4. Se pueden descargar precios en tiempo real
  5. Se pueden descargar datos historicos (H1, H4, D1)
  6. Symbol info disponible para los pares configurados
"""
import sys
from pathlib import Path

# Agregar raiz del proyecto al path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def validate():
    from forex_bot.data.collector import DataCollector
    from forex_bot.config import settings

    passed = 0
    failed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" -> {detail}"
        print(msg)

    print("=" * 60)
    print("VALIDACION DE CONEXION MT5")
    print("=" * 60)
    print()

    # 1. Conexion
    print("1. CONEXION")
    collector = DataCollector()
    connected = collector.connect()
    check("Conexion a MT5", connected)

    if not connected:
        print("\n  No se pudo conectar a MT5. Verificar:")
        print("  - MetaTrader 5 esta abierto y corriendo")
        print("  - Las credenciales en config/mt5_config.py son correctas")
        print("  - El servidor del broker esta accesible")
        return

    # 2. Info de cuenta
    print("\n2. CUENTA")
    info = collector.get_account_info()
    check("Obtener info de cuenta", info is not None)
    if info:
        check("Login correcto", info["login"] > 0, f"Login: {info['login']}")
        check("Server conectado", len(info["server"]) > 0, f"Server: {info['server']}")
        check("Balance disponible", info["balance"] > 0, f"Balance: ${info['balance']:.2f} {info['currency']}")
        check("Leverage configurado", info["leverage"] > 0, f"Leverage: 1:{info['leverage']}")
        print(f"\n  Equity: ${info['equity']:.2f} | Margen libre: ${info['free_margin']:.2f}")

    # 3. Precios en tiempo real
    print("\n3. PRECIOS EN TIEMPO REAL")
    for sym in settings.SYMBOLS:
        price = collector.get_current_price(sym)
        if price:
            check(
                f"Precio {sym}",
                True,
                f"bid={price['bid']:.5f} ask={price['ask']:.5f} spread={price['spread_pips']} pips",
            )
        else:
            check(f"Precio {sym}", False, "No se pudo obtener")

    # 4. Datos historicos
    print("\n4. DATOS HISTORICOS")
    for sym in settings.SYMBOLS:
        for tf in [settings.TIMEFRAME_PRIMARY, settings.TIMEFRAME_HIGHER, settings.TIMEFRAME_DAILY]:
            bars = 100
            df = collector.get_historical_data(sym, tf, bars=bars)
            if df is not None:
                check(
                    f"{sym} {tf}",
                    len(df) > 0,
                    f"{len(df)} barras | {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d %H:%M')}",
                )
            else:
                check(f"{sym} {tf}", False, "No se pudieron obtener datos")

    # 5. Descarga grande (solo EURUSD H1)
    print("\n5. DESCARGA MASIVA")
    main_sym = settings.SYMBOLS[0]
    df_big = collector.get_historical_data(main_sym, "H1", bars=5000)
    if df_big is not None:
        check(
            f"{main_sym} H1 (5000 barras)",
            len(df_big) >= 4000,
            f"{len(df_big)} barras descargadas",
        )
    else:
        check(f"{main_sym} H1 (5000 barras)", False)

    # 6. Symbol info
    print("\n6. INFO DE SIMBOLOS")
    for sym in settings.SYMBOLS:
        sym_info = collector.get_symbol_info(sym)
        if sym_info:
            check(
                f"Info {sym}",
                True,
                f"digits={sym_info['digits']} spread={sym_info['spread']} lot_min={sym_info['volume_min']}",
            )
        else:
            check(f"Info {sym}", False)

    # Desconectar
    collector.disconnect()

    # Resumen
    print()
    print("=" * 60)
    print(f"RESULTADO: {passed}/{total} checks pasados", end="")
    if failed > 0:
        print(f" ({failed} fallidos)")
    else:
        print(" - TODO OK")
    print("=" * 60)


if __name__ == "__main__":
    validate()

"""Test rapido de conexion a MT5."""
import sys
sys.path.insert(0, ".")

import MetaTrader5 as mt5

print("=" * 60)
print("TEST DE CONEXION A METATRADER 5")
print("=" * 60)

# Intento 1: conectar a la sesion ya abierta (sin credenciales)
print("\n[Intento 1] Conectar a sesion MT5 ya abierta...")
if mt5.initialize():
    print("CONECTADO (sesion existente)")
else:
    error = mt5.last_error()
    print(f"Fallo: {error}")
    
    # Intento 2: conectar con path solamente
    print("\n[Intento 2] Conectar solo con path...")
    if mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        print("CONECTADO (con path)")
    else:
        error = mt5.last_error()
        print(f"Fallo: {error}")
        
        # Intento 3: con credenciales completas
        print("\n[Intento 3] Conectar con credenciales...")
        from forex_bot.config import mt5_config
        if mt5.initialize(
            path=mt5_config.MT5_PATH,
            login=mt5_config.MT5_LOGIN,
            password=mt5_config.MT5_PASSWORD,
            server=mt5_config.MT5_SERVER,
        ):
            print("CONECTADO (con credenciales)")
        else:
            error = mt5.last_error()
            print(f"Fallo: {error}")
            print("\nNo se pudo conectar. Verifica que:")
            print("  1. MetaTrader 5 esta abierto")
            print("  2. Estas logueado en la cuenta correcta")
            print("  3. El terminal muestra 'connected' abajo a la derecha")
            mt5.shutdown()
            sys.exit(1)

# Si llegamos aqui, estamos conectados
info = mt5.account_info()
if info:
    trade_mode = "Demo" if info.trade_mode == 0 else "Live" if info.trade_mode == 2 else "Contest"
    print(f"\n--- INFO DE CUENTA ---")
    print(f"  Cuenta:     {info.login}")
    print(f"  Servidor:   {info.server}")
    print(f"  Nombre:     {info.name}")
    print(f"  Balance:    ${info.balance:.2f} {info.currency}")
    print(f"  Equity:     ${info.equity:.2f}")
    print(f"  Leverage:   1:{info.leverage}")
    print(f"  Tipo:       {trade_mode}")
else:
    print("No se pudo obtener info de cuenta")

# Terminal info
term = mt5.terminal_info()
if term:
    print(f"\n--- TERMINAL ---")
    print(f"  MT5 Build:  {term.build}")
    print(f"  Conectado:  {term.connected}")

# Precios en vivo
print(f"\n--- PRECIOS EN VIVO ---")
symbols = ["EURUSD", "GBPUSD", "USDJPY"]
for sym in symbols:
    # Primero activar el simbolo
    mt5.symbol_select(sym, True)
    tick = mt5.symbol_info_tick(sym)
    if tick:
        pip_size = 0.01 if "JPY" in sym else 0.0001
        spread = (tick.ask - tick.bid) / pip_size
        print(f"  {sym}: bid={tick.bid:.5f}  ask={tick.ask:.5f}  spread={spread:.1f} pips")
    else:
        print(f"  {sym}: NO DISPONIBLE - {mt5.last_error()}")

# Descarga de datos
print(f"\n--- TEST DESCARGA DATOS ---")
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 10)
if rates is not None and len(rates) > 0:
    import pandas as pd
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(f"  Descargadas {len(rates)} velas H1 de EURUSD")
    print(f"  Ultima vela: {df['time'].iloc[-1]} | close={df['close'].iloc[-1]:.5f}")
else:
    print(f"  ERROR descarga: {mt5.last_error()}")

mt5.shutdown()
print("\n" + "=" * 60)
print("TEST COMPLETADO EXITOSAMENTE")
print("=" * 60)

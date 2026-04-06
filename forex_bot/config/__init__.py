"""
Modulo de configuracion.
Importa settings y mt5_config para uso facil.
"""
from forex_bot.config.settings import *

# Intentar importar credenciales MT5 (puede no existir)
try:
    from forex_bot.config.mt5_config import (
        MT5_PATH,
        MT5_LOGIN,
        MT5_PASSWORD,
        MT5_SERVER,
        TELEGRAM_BOT_TOKEN,
        TELEGRAM_CHAT_ID,
    )
    MT5_CONFIG_LOADED = True
except ImportError:
    MT5_CONFIG_LOADED = False
    MT5_PATH = ""
    MT5_LOGIN = 0
    MT5_PASSWORD = ""
    MT5_SERVER = ""
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

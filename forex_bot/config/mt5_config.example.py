"""
Template de configuracion de MetaTrader 5.
INSTRUCCIONES:
  1. Copia este archivo como 'mt5_config.py'
  2. Llena tus credenciales reales
  3. NUNCA subas mt5_config.py al repositorio
"""

# =============================================================================
# METATRADER 5 - CREDENCIALES
# =============================================================================

# Ruta al terminal de MetaTrader 5
# Ejemplo: r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Numero de cuenta (login)
MT5_LOGIN = 12345678

# Contrasena de la cuenta
MT5_PASSWORD = "tu_password_aqui"

# Servidor del broker
# Ejemplo: "MetaQuotes-Demo", "ICMarketsSC-Demo", "Pepperstone-Demo"
MT5_SERVER = "NombreBroker-Demo"

# =============================================================================
# TELEGRAM - NOTIFICACIONES (OPCIONAL)
# =============================================================================

# Crear bot en @BotFather y obtener token
TELEGRAM_BOT_TOKEN = ""

# Obtener chat_id enviando mensaje al bot y consultando:
# https://api.telegram.org/bot<TOKEN>/getUpdates
TELEGRAM_CHAT_ID = ""

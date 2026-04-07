"""
Data Collector - Recoleccion de datos de MetaTrader 5.
Maneja conexion, descarga de datos historicos y en tiempo real.

Soporta 2 modos de conexion:
  - Local (Windows): usa MetaTrader5 nativo (import MetaTrader5)
  - Docker/Linux:    usa mt5linux via RPyC (se conecta a MT5 corriendo en otro container)

El modo se detecta automaticamente por la variable de entorno MT5_HOST.
"""
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from forex_bot.config import settings

logger = logging.getLogger(__name__)

# Detectar modo de conexion: Docker (mt5linux) o Local (MetaTrader5 nativo)
_MT5_HOST = os.environ.get("MT5_HOST")
_MT5_PORT = int(os.environ.get("MT5_PORT", "8001"))
_IS_DOCKER = _MT5_HOST is not None

if _IS_DOCKER:
    logger.info("Modo Docker detectado: conectando via mt5linux a %s:%d", _MT5_HOST, _MT5_PORT)
else:
    logger.info("Modo local detectado: usando MetaTrader5 nativo")


class DataCollector:
    """
    Recolector de datos de mercado via MetaTrader 5.
    Maneja conexion, descarga historica, precios en tiempo real y persistencia.

    Soporta 2 backends:
      - MetaTrader5 nativo (Windows, tu PC local)
      - mt5linux via RPyC (Docker, VPS Linux)
    """

    def __init__(self):
        self._mt5 = None
        self._connected = False

    # ------------------------------------------------------------------
    # Conexion
    # ------------------------------------------------------------------

    def _load_mt5_module(self):
        """
        Cargar el modulo MT5 segun el entorno.
        - Docker: usa mt5linux (RPyC al container MT5)
        - Local:  usa MetaTrader5 nativo (Windows)
        """
        if _IS_DOCKER:
            try:
                from mt5linux import MetaTrader5
                self._mt5 = MetaTrader5(host=_MT5_HOST, port=_MT5_PORT)
                logger.info("mt5linux conectado a %s:%d", _MT5_HOST, _MT5_PORT)
                return True
            except ImportError as e:
                logger.error("mt5linux no esta instalado: %s. pip install mt5linux", e)
                return False
            except Exception as e:
                logger.error("Error conectando mt5linux a %s:%d: %s", _MT5_HOST, _MT5_PORT, e)
                return False
        else:
            try:
                import MetaTrader5 as mt5
                self._mt5 = mt5
                return True
            except ImportError:
                logger.error("MetaTrader5 no esta instalado. pip install MetaTrader5")
                return False

    def connect(self) -> bool:
        """Conectar a MetaTrader 5 y verificar cuenta."""
        if not self._load_mt5_module():
            return False

        # Obtener credenciales: primero de env vars (Docker), luego de mt5_config (local)
        init_kwargs = {}

        if _IS_DOCKER:
            # En Docker las credenciales vienen de variables de entorno
            mt5_login = os.environ.get("MT5_LOGIN")
            mt5_password = os.environ.get("MT5_PASSWORD")
            mt5_server = os.environ.get("MT5_SERVER")

            if mt5_login:
                init_kwargs["login"] = int(mt5_login)
            if mt5_password:
                init_kwargs["password"] = mt5_password
            if mt5_server:
                init_kwargs["server"] = mt5_server
            # No se pasa path en Docker (MT5 ya esta corriendo en el otro container)
        else:
            # En local, usar mt5_config.py
            from forex_bot.config import mt5_config
            if mt5_config.MT5_PATH:
                init_kwargs["path"] = mt5_config.MT5_PATH
            if mt5_config.MT5_LOGIN:
                init_kwargs["login"] = mt5_config.MT5_LOGIN
            if mt5_config.MT5_PASSWORD:
                init_kwargs["password"] = mt5_config.MT5_PASSWORD
            if mt5_config.MT5_SERVER:
                init_kwargs["server"] = mt5_config.MT5_SERVER

        # En Docker, reintentar la conexion ya que el RPyC server puede tardar
        import time
        max_retries = 10 if _IS_DOCKER else 1
        retry_delay = 15  # segundos

        for attempt in range(1, max_retries + 1):
            try:
                if self._mt5.initialize(**init_kwargs):
                    break
                error = self._mt5.last_error()
                if attempt < max_retries:
                    logger.warning(
                        "MT5 initialize() intento %d/%d fallo: %s. Reintentando en %ds...",
                        attempt, max_retries, error, retry_delay,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error("MT5 initialize() fallo tras %d intentos: %s", max_retries, error)
                    return False
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        "MT5 conexion intento %d/%d error: %s. Reintentando en %ds...",
                        attempt, max_retries, e, retry_delay,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error("MT5 conexion fallo tras %d intentos: %s", max_retries, e)
                    return False

        # Verificar conexion a cuenta
        account = self._mt5.account_info()
        if account is None:
            logger.error("No se pudo obtener info de cuenta: %s", self._mt5.last_error())
            self._mt5.shutdown()
            return False

        self._connected = True
        logger.info(
            "Conectado a MT5 | Cuenta: %s | Servidor: %s | Balance: %.2f %s",
            account.login,
            account.server,
            account.balance,
            account.currency,
        )
        return True

    def disconnect(self):
        """Cerrar conexion con MT5."""
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("Desconectado de MT5")

    def ensure_connected(self) -> bool:
        """Verificar conexion y reconectar si es necesario."""
        if self._connected and self._mt5:
            # Verificar que sigue activa
            info = self._mt5.terminal_info()
            if info is not None:
                return True
            logger.warning("Conexion MT5 perdida, intentando reconectar...")

        self._connected = False
        for attempt in range(1, settings.MAX_RECONNECT_ATTEMPTS + 1):
            logger.info("Intento de reconexion %d/%d", attempt, settings.MAX_RECONNECT_ATTEMPTS)
            if self.connect():
                return True
            time.sleep(settings.RECONNECT_DELAY_SECONDS)

        logger.error("No se pudo reconectar a MT5 despues de %d intentos", settings.MAX_RECONNECT_ATTEMPTS)
        return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Datos historicos
    # ------------------------------------------------------------------

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = None,
    ) -> Optional[pd.DataFrame]:
        """
        Descargar datos historicos OHLCV de MT5.

        Args:
            symbol: Par de divisas (ej: "EURUSD")
            timeframe: Clave del timeframe (ej: "H1", "H4", "D1")
            bars: Cantidad de velas a descargar (default: HISTORY_BARS de settings)

        Returns:
            DataFrame con columnas: open, high, low, close, tick_volume, spread
            Index: DatetimeIndex (time)
            None si hay error.
        """
        if not self.ensure_connected():
            return None

        if bars is None:
            bars = settings.HISTORY_BARS

        tf_value = settings.TIMEFRAME_MAP.get(timeframe)
        if tf_value is None:
            logger.error("Timeframe invalido: %s. Opciones: %s", timeframe, list(settings.TIMEFRAME_MAP.keys()))
            return None

        # Verificar que el simbolo existe
        symbol_info = self._mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error("Simbolo no encontrado: %s", symbol)
            return None
        if not symbol_info.visible:
            if not self._mt5.symbol_select(symbol, True):
                logger.error("No se pudo activar simbolo: %s", symbol)
                return None

        # Descargar datos
        rates = self._mt5.copy_rates_from_pos(symbol, tf_value, 0, bars)
        if rates is None or len(rates) == 0:
            logger.error(
                "No se obtuvieron datos para %s %s: %s",
                symbol, timeframe, self._mt5.last_error(),
            )
            return None

        # Convertir a DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.index.name = "time"

        # Renombrar real_volume si existe, eliminar columnas innecesarias
        columns_keep = ["open", "high", "low", "close", "tick_volume", "spread"]
        df = df[[c for c in columns_keep if c in df.columns]]

        logger.info(
            "Descargados %d barras de %s %s | Rango: %s -> %s",
            len(df), symbol, timeframe,
            df.index[0].strftime("%Y-%m-%d %H:%M"),
            df.index[-1].strftime("%Y-%m-%d %H:%M"),
        )
        return df

    def get_multiple_timeframes(self, symbol: str) -> Optional[dict]:
        """
        Descargar datos de multiples timeframes para un par.

        Returns:
            Dict {timeframe_str: DataFrame} con H1, H4 y D1.
            None si falla alguno critico.
        """
        result = {}
        timeframes = [
            settings.TIMEFRAME_PRIMARY,
            settings.TIMEFRAME_HIGHER,
            settings.TIMEFRAME_DAILY,
        ]

        for tf in timeframes:
            df = self.get_historical_data(symbol, tf)
            if df is None:
                logger.error("Fallo descarga de %s %s", symbol, tf)
                return None
            result[tf] = df

        logger.info(
            "Descargados %d timeframes para %s: %s",
            len(result), symbol, list(result.keys()),
        )
        return result

    # ------------------------------------------------------------------
    # Datos en tiempo real
    # ------------------------------------------------------------------

    def get_current_price(self, symbol: str) -> Optional[dict]:
        """
        Obtener precio actual de un simbolo.

        Returns:
            Dict con: bid, ask, spread, time, last
        """
        if not self.ensure_connected():
            return None

        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("No se pudo obtener tick de %s: %s", symbol, self._mt5.last_error())
            return None

        pip_size = settings.PIP_SIZE.get(symbol, 0.0001)
        spread_pips = (tick.ask - tick.bid) / pip_size

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.ask - tick.bid,
            "spread_pips": round(spread_pips, 1),
            "last": tick.last,
            "time": datetime.fromtimestamp(tick.time),
        }

    def get_account_info(self) -> Optional[dict]:
        """
        Obtener informacion de la cuenta de trading.

        Returns:
            Dict con: balance, equity, margin, free_margin, profit, leverage, currency
        """
        if not self.ensure_connected():
            return None

        account = self._mt5.account_info()
        if account is None:
            logger.error("No se pudo obtener info de cuenta: %s", self._mt5.last_error())
            return None

        return {
            "login": account.login,
            "server": account.server,
            "balance": account.balance,
            "equity": account.equity,
            "margin": account.margin,
            "free_margin": account.margin_free,
            "profit": account.profit,
            "leverage": account.leverage,
            "currency": account.currency,
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save_data(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Guardar DataFrame a CSV en data_cache/.

        Args:
            df: DataFrame a guardar
            filename: Nombre del archivo (sin extension, se agrega .csv)

        Returns:
            Path del archivo guardado.
        """
        filepath = settings.DATA_DIR / f"{filename}.csv"
        df.to_csv(filepath)
        logger.info("Datos guardados en %s (%d filas)", filepath, len(df))
        return filepath

    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Cargar DataFrame desde CSV en data_cache/.

        Args:
            filename: Nombre del archivo (sin extension)

        Returns:
            DataFrame cargado, o None si no existe.
        """
        filepath = settings.DATA_DIR / f"{filename}.csv"
        if not filepath.exists():
            logger.warning("Archivo no encontrado: %s", filepath)
            return None

        df = pd.read_csv(filepath, index_col="time", parse_dates=True)
        logger.info("Datos cargados de %s (%d filas)", filepath, len(df))
        return df

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Obtener informacion detallada de un simbolo (spread, digits, etc.)."""
        if not self.ensure_connected():
            return None

        info = self._mt5.symbol_info(symbol)
        if info is None:
            return None

        return {
            "name": info.name,
            "description": info.description,
            "digits": info.digits,
            "point": info.point,
            "spread": info.spread,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
        }

    def __enter__(self):
        """Context manager - conectar."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager - desconectar."""
        self.disconnect()
        return False

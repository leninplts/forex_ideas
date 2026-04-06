"""
Feature Engine - Calculo de indicadores tecnicos y features para ML.
Transforma datos OHLCV en vectores de features que el modelo puede consumir.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from forex_bot.config import settings

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Motor de generacion de features para el modelo de ML.
    Calcula ~40-50 features basadas en indicadores tecnicos y price action.
    """

    def __init__(self):
        # Nombres de features generadas (se llena al ejecutar add_all_features)
        self._feature_names = []

    @property
    def feature_names(self) -> list:
        """Lista de nombres de features generadas."""
        return self._feature_names.copy()

    # ==================================================================
    # METODO PRINCIPAL
    # ==================================================================

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcular TODAS las features sobre un DataFrame OHLCV.
        Llama a cada metodo de features en orden.

        Args:
            df: DataFrame con columnas open, high, low, close, tick_volume
                y DatetimeIndex.

        Returns:
            DataFrame con todas las features agregadas como columnas nuevas.
        """
        df = df.copy()
        initial_cols = set(df.columns)

        # Tendencia
        df = self.add_sma(df)
        df = self.add_ema(df)
        df = self.add_adx(df)

        # Momentum
        df = self.add_rsi(df)
        df = self.add_stochastic(df)
        df = self.add_macd(df)
        df = self.add_cci(df)

        # Volatilidad
        df = self.add_atr(df)
        df = self.add_bollinger(df)

        # Volumen
        df = self.add_volume_features(df)

        # Price action
        df = self.add_price_action(df)

        # Guardar nombres de features generadas
        self._feature_names = [c for c in df.columns if c not in initial_cols]

        logger.info("Features generadas: %d nuevas columnas", len(self._feature_names))
        return df

    # ==================================================================
    # FEATURES DE TENDENCIA
    # ==================================================================

    def add_sma(self, df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
        """
        Simple Moving Averages + distancias + pendientes + cruces.
        """
        if periods is None:
            periods = settings.SMA_PERIODS

        for p in periods:
            col = f"sma_{p}"
            df[col] = ta.sma(df["close"], length=p)

            # Distancia del precio a la SMA (%)
            df[f"sma_{p}_dist"] = (df["close"] - df[col]) / df[col] * 100

            # Pendiente de la SMA (cambio vs hace 5 periodos)
            df[f"sma_{p}_slope"] = (df[col] - df[col].shift(5)) / df[col].shift(5) * 100

        # Cruces de SMAs (1 = cruce alcista, 0 = no)
        if len(periods) >= 2:
            for i in range(len(periods) - 1):
                fast = f"sma_{periods[i]}"
                slow = f"sma_{periods[i + 1]}"
                if fast in df.columns and slow in df.columns:
                    df[f"sma_cross_{periods[i]}_{periods[i + 1]}"] = (
                        (df[fast] > df[slow]).astype(int)
                    )

        return df

    def add_ema(self, df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
        """
        Exponential Moving Averages + distancias + cruces.
        """
        if periods is None:
            periods = settings.EMA_PERIODS

        for p in periods:
            col = f"ema_{p}"
            df[col] = ta.ema(df["close"], length=p)

            # Distancia del precio a la EMA (%)
            df[f"ema_{p}_dist"] = (df["close"] - df[col]) / df[col] * 100

        # Cruce de EMAs
        if len(periods) >= 2:
            fast = f"ema_{periods[0]}"
            slow = f"ema_{periods[-1]}"
            if fast in df.columns and slow in df.columns:
                df[f"ema_cross_{periods[0]}_{periods[-1]}"] = (
                    (df[fast] > df[slow]).astype(int)
                )

        return df

    def add_adx(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        Average Directional Index + DI+ / DI-.
        ADX > 25 = tendencia fuerte.
        """
        if period is None:
            period = settings.ADX_PERIOD

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=period)
        if adx_df is not None and not adx_df.empty:
            df[f"adx_{period}"] = adx_df.iloc[:, 0]       # ADX
            df[f"di_plus_{period}"] = adx_df.iloc[:, 1]    # DI+
            df[f"di_minus_{period}"] = adx_df.iloc[:, 2]   # DI-

            # Tendencia fuerte (ADX > 25)
            df["adx_strong_trend"] = (df[f"adx_{period}"] > 25).astype(int)

        return df

    # ==================================================================
    # FEATURES DE MOMENTUM
    # ==================================================================

    def add_rsi(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        Relative Strength Index + zonas + pendiente.
        """
        if period is None:
            period = settings.RSI_PERIOD

        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)

        # Zonas
        rsi_col = f"rsi_{period}"
        df["rsi_overbought"] = (df[rsi_col] > 70).astype(int)
        df["rsi_oversold"] = (df[rsi_col] < 30).astype(int)

        # Pendiente del RSI (momentum del momentum)
        df["rsi_slope"] = df[rsi_col] - df[rsi_col].shift(3)

        return df

    def add_stochastic(self, df: pd.DataFrame, k: int = None, d: int = None) -> pd.DataFrame:
        """
        Stochastic Oscillator %K y %D + cruces.
        """
        if k is None:
            k = settings.STOCH_K_PERIOD
        if d is None:
            d = settings.STOCH_D_PERIOD

        stoch_df = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d)
        if stoch_df is not None and not stoch_df.empty:
            df["stoch_k"] = stoch_df.iloc[:, 0]
            df["stoch_d"] = stoch_df.iloc[:, 1]

            # Cruce de stochastic (K cruza D hacia arriba = alcista)
            df["stoch_cross"] = (df["stoch_k"] > df["stoch_d"]).astype(int)

        return df

    def add_macd(self, df: pd.DataFrame, fast: int = None, slow: int = None, signal: int = None) -> pd.DataFrame:
        """
        MACD line, signal line, histograma + cruces.
        """
        if fast is None:
            fast = settings.MACD_FAST
        if slow is None:
            slow = settings.MACD_SLOW
        if signal is None:
            signal = settings.MACD_SIGNAL

        macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        if macd_df is not None and not macd_df.empty:
            df["macd_line"] = macd_df.iloc[:, 0]
            df["macd_histogram"] = macd_df.iloc[:, 1]
            df["macd_signal"] = macd_df.iloc[:, 2]

            # Cruce MACD (line > signal = alcista)
            df["macd_cross"] = (df["macd_line"] > df["macd_signal"]).astype(int)

            # Signo del histograma (positivo = momentum alcista)
            df["macd_hist_positive"] = (df["macd_histogram"] > 0).astype(int)

        return df

    def add_cci(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        Commodity Channel Index + zonas extremas.
        """
        if period is None:
            period = settings.CCI_PERIOD

        df[f"cci_{period}"] = ta.cci(df["high"], df["low"], df["close"], length=period)

        # Zonas extremas
        cci_col = f"cci_{period}"
        df["cci_overbought"] = (df[cci_col] > 100).astype(int)
        df["cci_oversold"] = (df[cci_col] < -100).astype(int)

        return df

    # ==================================================================
    # FEATURES DE VOLATILIDAD
    # ==================================================================

    def add_atr(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        Average True Range + ATR normalizado + ATR ratio.
        """
        if period is None:
            period = settings.ATR_PERIOD

        df[f"atr_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)

        atr_col = f"atr_{period}"

        # ATR como % del precio (normalizado, comparable entre pares)
        df["atr_pct"] = df[atr_col] / df["close"] * 100

        # ATR ratio: ATR actual vs ATR promedio de 50 periodos
        atr_ma = df[atr_col].rolling(window=50).mean()
        df["atr_ratio"] = df[atr_col] / atr_ma

        return df

    def add_bollinger(self, df: pd.DataFrame, period: int = None, std: float = None) -> pd.DataFrame:
        """
        Bollinger Bands + %B + ancho de bandas.
        """
        if period is None:
            period = settings.BB_PERIOD
        if std is None:
            std = settings.BB_STD

        bb_df = ta.bbands(df["close"], length=period, std=std)
        if bb_df is not None and not bb_df.empty:
            # pandas-ta bbands column order: BBL, BBM, BBU, BBB, BBP
            df["bb_lower"] = bb_df.iloc[:, 0]       # Lower band
            df["bb_middle"] = bb_df.iloc[:, 1]      # Middle band (SMA)
            df["bb_upper"] = bb_df.iloc[:, 2]       # Upper band
            df["bb_bandwidth"] = bb_df.iloc[:, 3]   # Bandwidth
            df["bb_pct_b"] = bb_df.iloc[:, 4]       # %B

            # Ancho de bandas normalizado (squeeze detection)
            df["bb_squeeze"] = (
                (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
            )

        return df

    # ==================================================================
    # FEATURES DE VOLUMEN
    # ==================================================================

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features basadas en tick_volume: MA, ratio, OBV.
        """
        vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        if vol_col not in df.columns:
            logger.warning("No hay columna de volumen en el DataFrame")
            return df

        # Volume MA
        ma_period = settings.VOLUME_MA_PERIOD
        df["volume_ma"] = df[vol_col].rolling(window=ma_period).mean()

        # Volume ratio (actual vs promedio)
        df["volume_ratio"] = df[vol_col] / df["volume_ma"]

        # OBV (On Balance Volume)
        if settings.OBV_ENABLED:
            obv = ta.obv(df["close"], df[vol_col])
            if obv is not None:
                df["obv"] = obv
                # OBV slope (tendencia del volumen acumulado)
                df["obv_slope"] = df["obv"] - df["obv"].shift(5)

        return df

    # ==================================================================
    # FEATURES DE PRICE ACTION
    # ==================================================================

    def add_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features derivadas directamente del precio:
        retornos, rango de vela, cuerpo, sombras, estructura.
        """
        # Retornos porcentuales en diferentes periodos
        for p in [1, 3, 5, 10]:
            df[f"return_{p}"] = df["close"].pct_change(p) * 100

        # Rango de vela (high - low) como % del precio
        df["candle_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100

        # Cuerpo de vela como % del rango
        candle_range = df["high"] - df["low"]
        candle_body = (df["close"] - df["open"]).abs()
        df["candle_body_pct"] = np.where(
            candle_range > 0,
            candle_body / candle_range * 100,
            0,
        )

        # Sombra superior como % del rango
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        df["upper_shadow_pct"] = np.where(
            candle_range > 0,
            upper_shadow / candle_range * 100,
            0,
        )

        # Sombra inferior como % del rango
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        df["lower_shadow_pct"] = np.where(
            candle_range > 0,
            lower_shadow / candle_range * 100,
            0,
        )

        # Distancia al high/low de los ultimos N periodos
        for p in [10, 20]:
            rolling_high = df["high"].rolling(window=p).max()
            rolling_low = df["low"].rolling(window=p).min()
            df[f"dist_high_{p}"] = (df["close"] - rolling_high) / rolling_high * 100
            df[f"dist_low_{p}"] = (df["close"] - rolling_low) / rolling_low * 100

        # Higher Highs / Lower Lows (estructura de mercado)
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)

        # Bullish/Bearish candle
        df["bullish_candle"] = (df["close"] > df["open"]).astype(int)

        return df

    # ==================================================================
    # FEATURES MULTI-TIMEFRAME
    # ==================================================================

    def add_higher_timeframe_features(
        self,
        df_primary: pd.DataFrame,
        df_higher: pd.DataFrame,
        suffix: str = "htf",
    ) -> pd.DataFrame:
        """
        Agregar features de un timeframe superior al DataFrame principal.
        Usa merge_asof para alinear timestamps correctamente.

        Args:
            df_primary: DataFrame del timeframe principal (ej: H1)
            df_higher: DataFrame del timeframe superior (ej: H4 o D1)
            suffix: Sufijo para las columnas del tf superior

        Returns:
            df_primary con features del tf superior agregadas.
        """
        df_primary = df_primary.copy()
        df_htf = df_higher.copy()

        # Calcular indicadores en el timeframe superior
        rsi = ta.rsi(df_htf["close"], length=settings.RSI_PERIOD)
        if rsi is not None:
            df_htf[f"rsi_{suffix}"] = rsi

        sma_50 = ta.sma(df_htf["close"], length=50)
        if sma_50 is not None:
            df_htf[f"sma50_{suffix}"] = sma_50
            df_htf[f"trend_{suffix}"] = (df_htf["close"] > df_htf[f"sma50_{suffix}"]).astype(int)

        atr = ta.atr(df_htf["high"], df_htf["low"], df_htf["close"], length=settings.ATR_PERIOD)
        if atr is not None:
            df_htf[f"atr_{suffix}"] = atr

        # Seleccionar solo columnas nuevas para el merge
        htf_cols = [c for c in df_htf.columns if suffix in c]
        if not htf_cols:
            logger.warning("No se generaron features HTF con suffix '%s'", suffix)
            return df_primary

        df_htf_merge = df_htf[htf_cols].copy()
        df_htf_merge.index.name = "time"

        # merge_asof: para cada timestamp de H1, buscar el H4/D1 mas reciente
        # Esto evita lookahead bias
        result = pd.merge_asof(
            df_primary,
            df_htf_merge,
            left_index=True,
            right_index=True,
            direction="backward",  # Solo mirar hacia atras, nunca hacia adelante
        )

        logger.info("Agregadas %d features HTF (%s)", len(htf_cols), suffix)
        return result

    # ==================================================================
    # TARGET (lo que predecimos)
    # ==================================================================

    def create_target(
        self,
        df: pd.DataFrame,
        horizon: int = None,
        min_pips: int = None,
        pip_size: float = 0.0001,
    ) -> pd.Series:
        """
        Crear variable target para el modelo.

        Clasificacion ternaria:
          1 = BUY  (precio sube > min_pips en las siguientes N velas)
         -1 = SELL (precio baja > min_pips en las siguientes N velas)
          0 = HOLD (movimiento insuficiente)

        Args:
            df: DataFrame con columna 'close'
            horizon: Numero de velas hacia adelante para evaluar
            min_pips: Movimiento minimo en pips para generar senal
            pip_size: Tamano de 1 pip (0.0001 para EURUSD, 0.01 para USDJPY)

        Returns:
            Series con valores {-1, 0, 1} del mismo tamanho que df.
            Las ultimas N filas seran NaN (no hay datos futuros).
        """
        if horizon is None:
            horizon = settings.PREDICTION_HORIZON
        if min_pips is None:
            min_pips = settings.MIN_MOVEMENT_PIPS

        min_move = min_pips * pip_size

        # Retorno futuro: precio en N velas - precio actual
        future_return = df["close"].shift(-horizon) - df["close"]

        # Clasificar
        target = pd.Series(0, index=df.index, name="target", dtype=int)
        target[future_return > min_move] = 1     # BUY
        target[future_return < -min_move] = -1   # SELL

        # Las ultimas N filas no tienen datos futuros -> NaN
        target.iloc[-horizon:] = np.nan

        # Estadisticas
        valid = target.dropna()
        buy_pct = (valid == 1).mean() * 100
        sell_pct = (valid == -1).mean() * 100
        hold_pct = (valid == 0).mean() * 100

        logger.info(
            "Target creado (horizon=%d, min_pips=%d): BUY=%.1f%% | HOLD=%.1f%% | SELL=%.1f%%",
            horizon, min_pips, buy_pct, hold_pct, sell_pct,
        )

        return target

    # ==================================================================
    # UTILIDADES
    # ==================================================================

    def remove_lookahead_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Verificar y eliminar cualquier feature que use datos futuros.
        Solo la columna 'target' puede contener informacion futura.

        Verifica que ninguna feature tenga correlacion perfecta con retornos futuros
        (lo cual indicaria data leakage).

        Returns:
            DataFrame limpio.
        """
        df = df.copy()

        if "target" not in df.columns:
            logger.info("No hay columna 'target', nada que verificar")
            return df

        feature_cols = [c for c in self._feature_names if c in df.columns]
        if not feature_cols:
            return df

        # Verificar correlaciones sospechosamente altas con el target
        valid_mask = df["target"].notna()
        suspicious = []

        for col in feature_cols:
            if df[col].notna().sum() < 50:
                continue
            corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, "target"])
            if abs(corr) > 0.95:
                suspicious.append((col, corr))

        if suspicious:
            logger.warning(
                "ALERTA: %d features con correlacion > 0.95 con target (posible lookahead bias):",
                len(suspicious),
            )
            for col, corr in suspicious:
                logger.warning("  %s: corr=%.4f", col, corr)
        else:
            logger.info("Verificacion de lookahead bias: OK (sin features sospechosas)")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Retornar lista de columnas que son features (excluyendo OHLCV y target).
        """
        exclude = {"open", "high", "low", "close", "tick_volume", "volume", "spread", "target"}
        return [c for c in df.columns if c not in exclude]

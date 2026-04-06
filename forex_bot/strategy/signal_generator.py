"""
Signal Generator - Genera senales de trading combinando ML + filtros.
Pipeline: datos -> features -> prediccion ML -> filtros -> senal final con SL/TP.
"""
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from forex_bot.config import settings
from forex_bot.data.feature_engine import FeatureEngine
from forex_bot.models.predictor import ModelPredictor

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generador de senales de trading.
    Combina predicciones del modelo ML con filtros de tendencia,
    volatilidad, spread y horario para producir senales operables.
    """

    def __init__(self, predictor: ModelPredictor, feature_engine: FeatureEngine):
        """
        Args:
            predictor: ModelPredictor con modelo cargado
            feature_engine: FeatureEngine para calcular indicadores
        """
        self.predictor = predictor
        self.feature_engine = feature_engine

    # ------------------------------------------------------------------
    # Generacion de senales
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame = None,
        df_d1: pd.DataFrame = None,
        current_price: dict = None,
    ) -> dict:
        """
        Pipeline completo de generacion de senal para un par.

        1. Calcular features en H1
        2. Agregar features multi-timeframe (H4, D1)
        3. Obtener prediccion del modelo ML
        4. Aplicar filtros (tendencia, volatilidad, spread, horario)
        5. Calcular SL/TP basados en ATR
        6. Retornar senal final

        Args:
            symbol: Par de divisas (ej: "EURUSD")
            df_h1: DataFrame H1 con OHLCV (minimo 200 barras)
            df_h4: DataFrame H4 (opcional, para multi-timeframe)
            df_d1: DataFrame D1 (opcional, para filtro de tendencia)
            current_price: Dict con {bid, ask, spread_pips} (opcional)

        Returns:
            Dict con senal completa:
                symbol, signal, confidence, entry_price,
                sl_price, tp_price, sl_pips, tp_pips,
                lot_size_suggestion, reason, filters_passed, timestamp
        """
        # Resultado base (HOLD por defecto)
        result = {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0.0,
            "entry_price": 0.0,
            "sl_price": 0.0,
            "tp_price": 0.0,
            "sl_pips": 0.0,
            "tp_pips": 0.0,
            "reason": "",
            "filters_passed": True,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. Calcular features
        df = self.feature_engine.add_all_features(df_h1)

        # 2. Multi-timeframe
        if df_h4 is not None:
            df = self.feature_engine.add_higher_timeframe_features(df, df_h4, suffix="h4")
        if df_d1 is not None:
            df = self.feature_engine.add_higher_timeframe_features(df, df_d1, suffix="d1")

        # 3. Prediccion ML
        if not self.predictor.is_loaded:
            result["reason"] = "Modelo no cargado"
            return result

        prediction = self.predictor.predict(df)
        result["signal"] = prediction["signal"]
        result["confidence"] = prediction["confidence"]

        # Si es HOLD o baja confianza, no seguir procesando
        if not self.predictor.should_trade(prediction):
            result["signal"] = "HOLD"
            reason_parts = []
            if prediction["signal"] == "HOLD":
                reason_parts.append("ML predice HOLD")
            else:
                reason_parts.append(
                    f"Confianza baja: {prediction['confidence']:.1%} < {settings.CONFIDENCE_THRESHOLD:.1%}"
                )
            result["reason"] = " | ".join(reason_parts)
            return result

        # 4. Aplicar filtros
        filter_result = self.apply_filters(prediction, df, df_d1, current_price)
        if not filter_result["passed"]:
            result["signal"] = "HOLD"
            result["filters_passed"] = False
            result["reason"] = f"Filtro rechazado: {filter_result['reason']}"
            return result

        # 5. Calcular SL/TP
        entry_price = df["close"].iloc[-1]
        if current_price:
            entry_price = current_price.get("ask", entry_price) if prediction["signal"] == "BUY" \
                else current_price.get("bid", entry_price)

        sl_price, tp_price, sl_pips, tp_pips = self.calculate_sl_tp(
            prediction["signal"], entry_price, df, symbol,
        )

        result["entry_price"] = entry_price
        result["sl_price"] = sl_price
        result["tp_price"] = tp_price
        result["sl_pips"] = sl_pips
        result["tp_pips"] = tp_pips
        result["reason"] = (
            f"ML: {prediction['signal']} ({prediction['confidence']:.1%}) | "
            f"SL: {sl_pips:.1f} pips | TP: {tp_pips:.1f} pips | "
            f"R:R 1:{tp_pips / sl_pips:.1f}" if sl_pips > 0 else "SL calculado en 0"
        )

        logger.info(
            "SENAL %s %s @ %.5f | SL: %.5f (%.1f pips) | TP: %.5f (%.1f pips) | Conf: %.1f%%",
            result["signal"], symbol, entry_price,
            sl_price, sl_pips, tp_price, tp_pips,
            result["confidence"] * 100,
        )

        return result

    # ------------------------------------------------------------------
    # Filtros
    # ------------------------------------------------------------------

    def apply_filters(
        self,
        prediction: dict,
        df_h1: pd.DataFrame,
        df_d1: pd.DataFrame = None,
        current_price: dict = None,
    ) -> dict:
        """
        Aplicar filtros a la senal para evitar malas entradas.

        Returns:
            Dict con {passed: bool, reason: str}
        """
        signal = prediction["signal"]
        reasons = []

        # --- Filtro de tendencia D1: no operar contra la tendencia diaria ---
        if df_d1 is not None and len(df_d1) >= 50:
            sma50_d1 = ta.sma(df_d1["close"], length=50)
            if sma50_d1 is not None and not sma50_d1.empty:
                last_close_d1 = df_d1["close"].iloc[-1]
                last_sma50 = sma50_d1.iloc[-1]
                if not np.isnan(last_sma50):
                    daily_trend = "UP" if last_close_d1 > last_sma50 else "DOWN"
                    if signal == "BUY" and daily_trend == "DOWN":
                        reasons.append(f"Contra tendencia D1 (BUY pero D1 bajista, close < SMA50)")
                    elif signal == "SELL" and daily_trend == "UP":
                        reasons.append(f"Contra tendencia D1 (SELL pero D1 alcista, close > SMA50)")

        # --- Filtro de volatilidad: no operar si ATR es anormalmente bajo o alto ---
        if "atr_ratio" in df_h1.columns:
            atr_ratio = df_h1["atr_ratio"].iloc[-1]
            if not np.isnan(atr_ratio):
                if atr_ratio < 0.5:
                    reasons.append(f"Volatilidad muy baja (ATR ratio: {atr_ratio:.2f})")
                elif atr_ratio > 3.0:
                    reasons.append(f"Volatilidad excesiva (ATR ratio: {atr_ratio:.2f})")

        # --- Filtro de spread: no operar si spread es muy alto ---
        if current_price and "spread_pips" in current_price:
            spread = current_price["spread_pips"]
            max_spread = 5.0  # Maximo 5 pips de spread
            if spread > max_spread:
                reasons.append(f"Spread alto: {spread:.1f} pips > {max_spread}")

        # --- Filtro de horario: evitar baja liquidez ---
        now = datetime.now()
        day_name = now.strftime("%A").lower()
        hour = now.hour

        avoid_hours = settings.AVOID_TRADING_HOURS.get(day_name, [])
        if hour in avoid_hours:
            reasons.append(f"Horario de baja liquidez ({day_name} {hour}:00 UTC)")

        # --- Resultado ---
        if reasons:
            reason_text = " | ".join(reasons)
            logger.info("Filtros rechazaron senal: %s", reason_text)
            return {"passed": False, "reason": reason_text}

        return {"passed": True, "reason": "OK"}

    # ------------------------------------------------------------------
    # SL / TP
    # ------------------------------------------------------------------

    def calculate_sl_tp(
        self,
        signal: str,
        entry_price: float,
        df: pd.DataFrame,
        symbol: str,
    ) -> tuple:
        """
        Calcular Stop Loss y Take Profit basados en ATR.

        SL = entry +/- (ATR * SL_ATR_MULTIPLIER)
        TP = entry +/- (ATR * TP_ATR_MULTIPLIER)

        Args:
            signal: "BUY" o "SELL"
            entry_price: Precio de entrada
            df: DataFrame con columna ATR calculada
            symbol: Par de divisas

        Returns:
            Tuple (sl_price, tp_price, sl_pips, tp_pips)
        """
        pip_size = settings.PIP_SIZE.get(symbol, 0.0001)

        # Obtener ATR actual
        atr_col = f"atr_{settings.ATR_PERIOD}"
        if atr_col in df.columns:
            atr = df[atr_col].iloc[-1]
        else:
            # Calcular ATR si no existe
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=settings.ATR_PERIOD)
            atr = atr_series.iloc[-1] if atr_series is not None else 0

        if np.isnan(atr) or atr <= 0:
            # Fallback: usar rango promedio de las ultimas 14 velas
            atr = (df["high"] - df["low"]).tail(14).mean()
            logger.warning("ATR invalido, usando rango promedio: %.5f", atr)

        sl_distance = atr * settings.SL_ATR_MULTIPLIER
        tp_distance = atr * settings.TP_ATR_MULTIPLIER

        # Minimo SL de 10 pips para evitar stops demasiado tight
        min_sl_distance = 10 * pip_size
        if sl_distance < min_sl_distance:
            sl_distance = min_sl_distance
            tp_distance = sl_distance * settings.MIN_RISK_REWARD
            logger.info("SL ajustado al minimo de 10 pips")

        if signal == "BUY":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        elif signal == "SELL":
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        else:
            return entry_price, entry_price, 0.0, 0.0

        sl_pips = sl_distance / pip_size
        tp_pips = tp_distance / pip_size

        # Redondear precios al digito correcto
        digits = 5 if pip_size == 0.0001 else 3
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)

        return sl_price, tp_price, sl_pips, tp_pips

"""
Model Predictor - Predicciones en tiempo real con modelo entrenado.
Carga un modelo guardado y genera predicciones BUY/SELL/HOLD.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from forex_bot.config import settings

logger = logging.getLogger(__name__)

# Mapeo inverso: clase numerica -> senal de trading
# El trainer mapea {-1:0, 0:1, 1:2}, asi que el inverso es:
CLASS_TO_SIGNAL = {0: "SELL", 1: "HOLD", 2: "BUY"}
SIGNAL_TO_CLASS = {"SELL": 0, "HOLD": 1, "BUY": 2}


class ModelPredictor:
    """
    Predictor de senales de trading usando modelo ML entrenado.
    Carga modelo, valida features, genera predicciones con confianza.
    """

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Ruta al archivo del modelo (.json o .pkl).
                        Si None, busca el mas reciente en models/saved/.
        """
        self.model = None
        self.model_type = None
        self._feature_names = []
        self._metadata = {}
        self._train_date = None

        if model_path:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """
        Cargar modelo desde archivo.

        Args:
            model_path: Ruta al archivo del modelo.
        """
        filepath = Path(model_path)
        if not filepath.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {filepath}")

        if filepath.suffix == ".json":
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(filepath))
            self.model_type = "xgboost"

        elif filepath.suffix == ".pkl":
            import joblib
            self.model = joblib.load(filepath)
            self.model_type = "lightgbm"

        else:
            raise ValueError(f"Extension no soportada: {filepath.suffix}")

        # Cargar metadata
        meta_path = filepath.with_suffix("").with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self._metadata = json.load(f)
            self._feature_names = self._metadata.get("feature_names", [])
            train_date_str = self._metadata.get("train_date", "")
            if train_date_str:
                try:
                    self._train_date = datetime.fromisoformat(train_date_str)
                except ValueError:
                    self._train_date = None

        logger.info(
            "Modelo cargado: %s (%s, %d features)",
            filepath.name, self.model_type, len(self._feature_names),
        )

    def load_latest(self) -> bool:
        """
        Cargar el modelo mas reciente de models/saved/.

        Returns:
            True si se cargo un modelo, False si no hay modelos.
        """
        saved_dir = settings.MODELS_DIR
        model_files = list(saved_dir.glob("*.json")) + list(saved_dir.glob("*.pkl"))

        if not model_files:
            logger.warning("No hay modelos guardados en %s", saved_dir)
            return False

        # Ordenar por fecha de modificacion, tomar el mas reciente
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        self.load(str(latest))
        return True

    # ------------------------------------------------------------------
    # Prediccion
    # ------------------------------------------------------------------

    def predict(self, features_df: pd.DataFrame) -> dict:
        """
        Generar prediccion a partir de features calculadas.

        Args:
            features_df: DataFrame con features. Puede ser 1 fila (prediccion individual)
                         o multiples filas.

        Returns:
            Dict con:
                signal: "BUY", "SELL", o "HOLD"
                confidence: Probabilidad de la clase predicha (0.0 - 1.0)
                probabilities: Dict con probabilidades por clase
                class_id: ID numerico de la clase predicha
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado. Llamar a load() primero.")

        # Verificar y alinear features
        X = self._align_features(features_df)

        # Predecir
        pred_class = self.model.predict(X)
        pred_proba = self.model.predict_proba(X)

        # Tomar la ultima fila (prediccion mas reciente)
        if len(pred_class) > 1:
            pred_class = pred_class[-1:]
            pred_proba = pred_proba[-1:]

        class_id = int(pred_class[0])
        signal = CLASS_TO_SIGNAL.get(class_id, "HOLD")
        probas = pred_proba[0]
        confidence = float(probas[class_id])

        result = {
            "signal": signal,
            "confidence": confidence,
            "probabilities": {
                "SELL": float(probas[0]),
                "HOLD": float(probas[1]),
                "BUY": float(probas[2]),
            },
            "class_id": class_id,
        }

        logger.info(
            "Prediccion: %s (confianza=%.2f%%) | SELL=%.1f%% HOLD=%.1f%% BUY=%.1f%%",
            signal, confidence * 100,
            probas[0] * 100, probas[1] * 100, probas[2] * 100,
        )

        return result

    def should_trade(self, prediction: dict) -> bool:
        """
        Determinar si la prediccion tiene suficiente confianza para operar.

        Args:
            prediction: Dict retornado por predict()

        Returns:
            True si se debe operar, False si no.
        """
        signal = prediction["signal"]
        confidence = prediction["confidence"]

        # No operar si la senal es HOLD
        if signal == "HOLD":
            return False

        # Verificar umbral de confianza
        if confidence < settings.CONFIDENCE_THRESHOLD:
            logger.info(
                "Senal %s rechazada: confianza %.2f%% < umbral %.2f%%",
                signal, confidence * 100, settings.CONFIDENCE_THRESHOLD * 100,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _align_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Alinear columnas del DataFrame con las features esperadas por el modelo.
        Agrega columnas faltantes con 0 y reordena.
        """
        if not self._feature_names:
            # Sin metadata, usar las columnas tal cual (sin OHLCV)
            non_feature = {"open", "high", "low", "close", "tick_volume", "volume", "spread", "target"}
            cols = [c for c in features_df.columns if c not in non_feature]
            return features_df[cols]

        # Verificar features faltantes
        missing = set(self._feature_names) - set(features_df.columns)
        if missing:
            logger.warning("Features faltantes (se rellenaran con 0): %s", missing)
            for col in missing:
                features_df = features_df.copy()
                features_df[col] = 0

        # Features extras (no usadas por el modelo)
        extra = set(features_df.columns) - set(self._feature_names)
        extra -= {"open", "high", "low", "close", "tick_volume", "volume", "spread", "target"}
        if extra:
            logger.debug("Features extra ignoradas: %d", len(extra))

        # Retornar solo las features esperadas, en el orden correcto
        return features_df[self._feature_names]

    def get_model_age_days(self) -> Optional[int]:
        """
        Obtener la edad del modelo en dias desde el ultimo entrenamiento.

        Returns:
            Dias desde el entrenamiento, o None si no hay metadata.
        """
        if self._train_date is None:
            return None

        age = (datetime.now() - self._train_date).days
        return age

    def needs_retraining(self) -> bool:
        """
        Verificar si el modelo necesita reentrenamiento.

        Returns:
            True si el modelo es mas viejo que MODEL_RETRAIN_DAYS.
        """
        age = self.get_model_age_days()
        if age is None:
            logger.warning("No se puede determinar edad del modelo (sin metadata)")
            return True

        needs = age > settings.MODEL_RETRAIN_DAYS
        if needs:
            logger.warning(
                "Modelo tiene %d dias (limite: %d). Se recomienda reentrenar.",
                age, settings.MODEL_RETRAIN_DAYS,
            )
        return needs

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def metadata(self) -> dict:
        return self._metadata.copy()

    @property
    def feature_names(self) -> list:
        return self._feature_names.copy()

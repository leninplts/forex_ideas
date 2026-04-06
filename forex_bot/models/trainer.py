"""
Model Trainer - Entrenamiento, evaluacion y persistencia de modelos ML.
Soporta XGBoost y LightGBM con walk-forward validation.

API verificada contra documentacion oficial:
  - XGBoost 3.2.0: early_stopping_rounds en constructor, eval_set en fit()
  - LightGBM 4.6.0: eval_set en fit(), callbacks para early stopping
  - scikit-learn 1.8.0: TimeSeriesSplit, classification_report
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit

from forex_bot.config import settings

logger = logging.getLogger(__name__)

# Mapeo de labels para el target: {-1: SELL, 0: HOLD, 1: BUY}
TARGET_LABELS = {-1: "SELL", 0: "HOLD", 1: "BUY"}

# Columnas que NO son features (se excluyen antes de entrenar)
NON_FEATURE_COLS = {"open", "high", "low", "close", "tick_volume", "volume", "spread", "target"}


class ModelTrainer:
    """
    Entrenador de modelos ML para clasificacion de senales de trading.
    Soporta XGBoost (XGBClassifier) y LightGBM (LGBMClassifier).
    """

    def __init__(self, model_type: str = None):
        """
        Args:
            model_type: "xgboost" o "lightgbm" (default: settings.MODEL_TYPE)
        """
        self.model_type = model_type or settings.MODEL_TYPE
        self.model = None
        self._feature_names = []
        self._metadata = {}

        logger.info("ModelTrainer inicializado con tipo: %s", self.model_type)

    # ------------------------------------------------------------------
    # Preparacion de datos
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Separar features (X) del target (y).
        Elimina columnas OHLCV, target, y filas con NaN.

        Args:
            df: DataFrame con features y columna 'target'

        Returns:
            Tuple (X: pd.DataFrame, y: pd.Series) limpios y listos para entrenar.
        """
        if "target" not in df.columns:
            raise ValueError("El DataFrame no tiene columna 'target'")

        # Separar features del target
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        X = df[feature_cols].copy()
        y = df["target"].copy()

        # Eliminar filas donde target es NaN (ultimas N filas sin datos futuros)
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Eliminar filas con NaN en features (primeras filas por lookback de indicadores)
        nan_rows = X.isna().any(axis=1)
        if nan_rows.any():
            n_nan = nan_rows.sum()
            X = X[~nan_rows]
            y = y.loc[X.index]
            logger.info("Eliminadas %d filas con NaN en features", n_nan)

        # Remap target: {-1, 0, 1} -> {0, 1, 2} para clasificacion multiclase
        # XGBoost y LightGBM esperan clases empezando desde 0
        y = y.map({-1: 0, 0: 1, 1: 2}).astype(int)

        self._feature_names = list(X.columns)

        logger.info(
            "Features preparadas: %d muestras, %d features | Clases: SELL=%d, HOLD=%d, BUY=%d",
            len(X), len(self._feature_names),
            (y == 0).sum(), (y == 1).sum(), (y == 2).sum(),
        )
        return X, y

    @staticmethod
    def _compute_sample_weights(y: pd.Series) -> np.ndarray:
        """
        Calcular pesos por muestra para compensar desbalance de clases.
        Las clases minoritarias reciben mayor peso.
        Usa el esquema 'balanced': weight_i = n_samples / (n_classes * count_i)
        """
        classes = y.unique()
        n_samples = len(y)
        n_classes = len(classes)
        weights = np.ones(n_samples)

        for c in classes:
            mask = (y == c)
            count = mask.sum()
            if count > 0:
                w = n_samples / (n_classes * count)
                weights[mask.values] = w

        logger.info(
            "Sample weights: %s",
            {int(c): round(n_samples / (n_classes * (y == c).sum()), 2) for c in sorted(classes)},
        )
        return weights

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def _create_model(self, params: dict = None):
        """Crear instancia del modelo segun tipo."""
        if self.model_type == "xgboost":
            import xgboost as xgb

            p = (params or settings.XGBOOST_PARAMS).copy()
            # XGBoost 3.2: early_stopping_rounds va en el constructor
            early_stopping = p.pop("early_stopping_rounds", 50)
            eval_metric = p.pop("eval_metric", "mlogloss")
            p.pop("num_class", None)  # XGBClassifier lo infiere automaticamente

            return xgb.XGBClassifier(
                early_stopping_rounds=early_stopping,
                eval_metric=eval_metric,
                verbosity=0,
                **p,
            )

        elif self.model_type == "lightgbm":
            import lightgbm as lgb

            p = (params or settings.LIGHTGBM_PARAMS).copy()
            p.pop("metric", None)  # Se pasa como eval_metric en fit()
            p.pop("early_stopping_rounds", None)  # Se usa callback en fit()

            return lgb.LGBMClassifier(
                importance_type="gain",
                **p,
            )

        else:
            raise ValueError(f"model_type desconocido: {self.model_type}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict = None,
    ):
        """
        Entrenar modelo con early stopping usando validacion.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validacion
            y_val: Target de validacion
            params: Hiperparametros custom (optional, usa settings por default)

        Returns:
            Modelo entrenado.
        """
        self.model = self._create_model(params)

        # Calcular sample_weight para compensar desbalance de clases
        sample_weights = self._compute_sample_weights(y_train)

        logger.info(
            "Entrenando %s: train=%d, val=%d, features=%d",
            self.model_type, len(X_train), len(X_val), X_train.shape[1],
        )

        if self.model_type == "xgboost":
            # XGBoost 3.2: eval_set en fit(), early_stopping ya en constructor
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            best_iter = self.model.best_iteration
            logger.info("XGBoost entrenado | Best iteration: %d", best_iter)

        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            # LightGBM 4.6: eval_set en fit(), early stopping via callbacks
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0),  # 0 = silencioso
                ],
            )
            best_iter = self.model.best_iteration_
            logger.info("LightGBM entrenado | Best iteration: %d", best_iter)

        # Guardar metadata
        self._metadata = {
            "model_type": self.model_type,
            "train_date": datetime.now().isoformat(),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": X_train.shape[1],
            "feature_names": self._feature_names,
            "best_iteration": best_iter,
        }

        return self.model

    # ------------------------------------------------------------------
    # Evaluacion
    # ------------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluar modelo entrenado con metricas de clasificacion.

        Args:
            X_test: Features de test
            y_test: Target de test (clases 0, 1, 2)

        Returns:
            Dict con metricas: accuracy, f1, classification_report, confusion_matrix,
            probabilities.
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llamar a train() primero.")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred,
            target_names=["SELL", "HOLD", "BUY"],
            output_dict=True,
        )
        cm = confusion_matrix(y_test, y_pred)

        logger.info("Evaluacion: Accuracy=%.4f | F1=%.4f", acc, f1)
        logger.info("Confusion Matrix:\n%s", cm)

        return {
            "accuracy": acc,
            "f1_weighted": f1,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "probabilities": y_proba,
            "predictions": y_pred,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Obtener feature importance del modelo entrenado.

        Returns:
            DataFrame con columnas [feature, importance] ordenado de mayor a menor.
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado.")

        importances = self.model.feature_importances_
        fi = pd.DataFrame({
            "feature": self._feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        logger.info("Top 10 features:\n%s", fi.head(10).to_string())
        return fi

    # ------------------------------------------------------------------
    # Optimizacion de hiperparametros
    # ------------------------------------------------------------------

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 3,
        n_iter: int = 20,
    ) -> dict:
        """
        Busqueda de hiperparametros con RandomizedSearchCV y TimeSeriesSplit.

        Args:
            X: Features completas
            y: Target completo
            n_splits: Numero de splits para cross-validation temporal
            n_iter: Numero de combinaciones a probar

        Returns:
            Dict con mejores hiperparametros encontrados.
        """
        from sklearn.model_selection import RandomizedSearchCV

        # TimeSeriesSplit para respetar orden temporal
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Grid de parametros a explorar
        if self.model_type == "xgboost":
            import xgboost as xgb
            base_model = xgb.XGBClassifier(
                eval_metric="mlogloss",
                verbosity=0,
                random_state=42,
                n_jobs=-1,
            )
            param_dist = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 4, 5, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.6, 0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
                "min_child_weight": [1, 3, 5, 7],
                "gamma": [0, 0.1, 0.2, 0.5],
                "reg_alpha": [0, 0.01, 0.1, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0],
            }

        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            base_model = lgb.LGBMClassifier(
                verbose=-1,
                random_state=42,
                n_jobs=-1,
            )
            param_dist = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 4, 5, 6, 8, -1],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63, 127],
                "subsample": [0.6, 0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
                "min_child_weight": [1, 3, 5, 7],
                "reg_alpha": [0, 0.01, 0.1, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0],
            }
        else:
            raise ValueError(f"model_type desconocido: {self.model_type}")

        logger.info(
            "Iniciando optimizacion de hiperparametros (%d iteraciones, %d splits)",
            n_iter, n_splits,
        )

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring="f1_weighted",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X, y)

        best_params = search.best_params_
        best_score = search.best_score_

        logger.info("Mejores parametros encontrados (score=%.4f):", best_score)
        for k, v in best_params.items():
            logger.info("  %s: %s", k, v)

        return {
            "best_params": best_params,
            "best_score": best_score,
        }

    # ------------------------------------------------------------------
    # Walk-Forward Validation
    # ------------------------------------------------------------------

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        train_window_days: int = None,
        val_window_days: int = None,
        step_days: int = 30,
    ) -> dict:
        """
        Walk-forward validation: entrenar/evaluar en ventanas temporales sucesivas.
        ESTO ES CRITICO para evitar overfitting en datos financieros.

        Args:
            df: DataFrame completo con features y target
            train_window_days: Dias de entrenamiento (default: settings)
            val_window_days: Dias de validacion (default: settings)
            step_days: Dias de avance entre splits

        Returns:
            Dict con metricas agregadas y por ventana.
        """
        from forex_bot.data.preprocessor import DataPreprocessor

        if train_window_days is None:
            train_window_days = settings.TRAIN_WINDOW_DAYS
        if val_window_days is None:
            val_window_days = settings.VALIDATION_WINDOW_DAYS

        pp = DataPreprocessor()
        splits = pp.create_walk_forward_splits(
            df, train_window_days, val_window_days, step_days,
        )

        if not splits:
            logger.error("No se generaron splits. Datos insuficientes.")
            return {"error": "No splits generated"}

        results_per_window = []
        all_predictions = []
        all_actuals = []

        for i, (train_df, val_df) in enumerate(splits):
            logger.info("Walk-forward split %d/%d", i + 1, len(splits))

            # Preparar features
            try:
                X_train, y_train = self.prepare_features(train_df)
                X_val, y_val = self.prepare_features(val_df)
            except Exception as e:
                logger.warning("Skip split %d: %s", i + 1, e)
                continue

            if len(X_train) < 50 or len(X_val) < 10:
                logger.warning("Skip split %d: datos insuficientes (train=%d, val=%d)", i + 1, len(X_train), len(X_val))
                continue

            # Entrenar y evaluar
            self.train(X_train, y_train, X_val, y_val)
            metrics = self.evaluate(X_val, y_val)

            results_per_window.append({
                "split": i + 1,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
            })

            all_predictions.extend(metrics["predictions"].tolist())
            all_actuals.extend(y_val.tolist())

        if not results_per_window:
            return {"error": "All splits failed"}

        # Metricas agregadas
        avg_accuracy = np.mean([r["accuracy"] for r in results_per_window])
        avg_f1 = np.mean([r["f1_weighted"] for r in results_per_window])
        std_accuracy = np.std([r["accuracy"] for r in results_per_window])

        # Metricas globales sobre todas las predicciones acumuladas
        global_accuracy = accuracy_score(all_actuals, all_predictions)
        global_f1 = f1_score(all_actuals, all_predictions, average="weighted")

        logger.info(
            "Walk-forward completo: %d splits | Avg Accuracy=%.4f (±%.4f) | Avg F1=%.4f",
            len(results_per_window), avg_accuracy, std_accuracy, avg_f1,
        )
        logger.info(
            "Global (acumulado): Accuracy=%.4f | F1=%.4f",
            global_accuracy, global_f1,
        )

        return {
            "n_splits": len(results_per_window),
            "avg_accuracy": avg_accuracy,
            "std_accuracy": std_accuracy,
            "avg_f1": avg_f1,
            "global_accuracy": global_accuracy,
            "global_f1": global_f1,
            "per_window": results_per_window,
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save_model(self, filepath: str = None) -> Path:
        """
        Guardar modelo entrenado y metadata.

        Para XGBoost: save_model() nativo a JSON.
        Para LightGBM: joblib (sklearn-compatible).

        Args:
            filepath: Ruta del archivo (sin extension). Default: models/saved/<model_type>_<fecha>

        Returns:
            Path del archivo guardado.
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar.")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = settings.MODELS_DIR / f"{self.model_type}_{timestamp}"
        else:
            filepath = Path(filepath)

        # Guardar modelo
        if self.model_type == "xgboost":
            model_path = filepath.with_suffix(".json")
            self.model.save_model(str(model_path))
        else:
            model_path = filepath.with_suffix(".pkl")
            joblib.dump(self.model, model_path)

        # Guardar metadata
        meta_path = filepath.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

        logger.info("Modelo guardado: %s", model_path)
        logger.info("Metadata guardada: %s", meta_path)

        return model_path

    def load_model(self, filepath: str) -> None:
        """
        Cargar modelo desde archivo.

        Args:
            filepath: Ruta al archivo del modelo (.json para XGBoost, .pkl para LightGBM)
        """
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(filepath))
            self.model_type = "xgboost"

        elif filepath.suffix == ".pkl":
            self.model = joblib.load(filepath)
            self.model_type = "lightgbm"

        else:
            raise ValueError(f"Extension no soportada: {filepath.suffix}")

        # Cargar metadata si existe
        meta_path = filepath.with_suffix("").with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self._metadata = json.load(f)
            self._feature_names = self._metadata.get("feature_names", [])
            logger.info(
                "Modelo cargado: %s (entrenado: %s, features: %d)",
                filepath, self._metadata.get("train_date", "?"), len(self._feature_names),
            )
        else:
            logger.info("Modelo cargado: %s (sin metadata)", filepath)

    @property
    def metadata(self) -> dict:
        return self._metadata.copy()

    @property
    def feature_names(self) -> list:
        return self._feature_names.copy()

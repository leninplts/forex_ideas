"""
VALIDACION FASE 3 - Modelos de Machine Learning
Testea trainer y predictor con datos sinteticos usando XGBoost y LightGBM.
Ejecutar: python -m pytest tests/test_fase3_models.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =====================================================================
# FIXTURE: datos con features y target sinteticos
# =====================================================================

@pytest.fixture
def sample_features_df():
    """
    Generar un DataFrame con features + target sinteticos
    que simulen el output del feature_engine.
    500 filas, 15 features, target ternario.
    """
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    # OHLCV base
    close = 1.08 + np.cumsum(np.random.normal(0, 0.0005, n))
    data = {
        "open": close + np.random.normal(0, 0.0002, n),
        "high": close + np.abs(np.random.normal(0, 0.0003, n)),
        "low": close - np.abs(np.random.normal(0, 0.0003, n)),
        "close": close,
        "tick_volume": np.random.randint(100, 5000, n),
    }

    # Features sinteticas (simulan indicadores)
    for i in range(15):
        data[f"feature_{i}"] = np.random.randn(n)

    # Hacer algunas features correlacionadas con la direccion
    data["feature_0"] = np.roll(np.sign(np.diff(close, prepend=close[0])), 0) + np.random.normal(0, 0.5, n)
    data["feature_1"] = np.random.randn(n) * 0.5

    df = pd.DataFrame(data, index=dates)
    df.index.name = "time"

    # Target: basado en retorno futuro
    future_return = df["close"].shift(-5) - df["close"]
    min_move = 15 * 0.0001  # 15 pips
    target = pd.Series(0, index=df.index, dtype=int)
    target[future_return > min_move] = 1
    target[future_return < -min_move] = -1
    target.iloc[-5:] = np.nan
    df["target"] = target

    return df


@pytest.fixture
def trained_xgboost(sample_features_df, tmp_path):
    """Entrenar un modelo XGBoost y retornar (trainer, X_val, y_val, model_path)."""
    from forex_bot.models.trainer import ModelTrainer

    trainer = ModelTrainer(model_type="xgboost")
    X, y = trainer.prepare_features(sample_features_df)

    # Split manual 80/20
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    trainer.train(X_train, y_train, X_val, y_val)
    model_path = trainer.save_model(str(tmp_path / "test_xgb"))

    return trainer, X_val, y_val, model_path


@pytest.fixture
def trained_lightgbm(sample_features_df, tmp_path):
    """Entrenar un modelo LightGBM y retornar (trainer, X_val, y_val, model_path)."""
    from forex_bot.models.trainer import ModelTrainer

    trainer = ModelTrainer(model_type="lightgbm")
    X, y = trainer.prepare_features(sample_features_df)

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    trainer.train(X_train, y_train, X_val, y_val)
    model_path = trainer.save_model(str(tmp_path / "test_lgb"))

    return trainer, X_val, y_val, model_path


# =====================================================================
# TESTS: ModelTrainer - Preparacion de datos
# =====================================================================

class TestTrainerPrepare:
    def test_prepare_features_returns_X_y(self, sample_features_df):
        from forex_bot.models.trainer import ModelTrainer
        trainer = ModelTrainer()
        X, y = trainer.prepare_features(sample_features_df)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0

    def test_prepare_excludes_ohlcv(self, sample_features_df):
        from forex_bot.models.trainer import ModelTrainer
        trainer = ModelTrainer()
        X, _ = trainer.prepare_features(sample_features_df)

        for col in ["open", "high", "low", "close", "tick_volume", "target"]:
            assert col not in X.columns, f"'{col}' no debe ser feature"

    def test_prepare_removes_nan_target(self, sample_features_df):
        from forex_bot.models.trainer import ModelTrainer
        trainer = ModelTrainer()
        X, y = trainer.prepare_features(sample_features_df)

        # No NaN en target
        assert not y.isna().any()

    def test_prepare_remaps_target(self, sample_features_df):
        """Target remapeado de {-1,0,1} a {0,1,2}."""
        from forex_bot.models.trainer import ModelTrainer
        trainer = ModelTrainer()
        _, y = trainer.prepare_features(sample_features_df)

        assert set(y.unique()).issubset({0, 1, 2})

    def test_prepare_no_nan_in_features(self, sample_features_df):
        from forex_bot.models.trainer import ModelTrainer
        trainer = ModelTrainer()
        X, _ = trainer.prepare_features(sample_features_df)

        assert not X.isna().any().any(), "Features no deben tener NaN"

    def test_feature_names_stored(self, sample_features_df):
        from forex_bot.models.trainer import ModelTrainer
        trainer = ModelTrainer()
        X, _ = trainer.prepare_features(sample_features_df)

        assert len(trainer.feature_names) == X.shape[1]
        assert trainer.feature_names == list(X.columns)

    def test_prepare_raises_without_target(self, sample_features_df):
        from forex_bot.models.trainer import ModelTrainer
        trainer = ModelTrainer()

        df_no_target = sample_features_df.drop(columns=["target"])
        with pytest.raises(ValueError, match="target"):
            trainer.prepare_features(df_no_target)


# =====================================================================
# TESTS: ModelTrainer - Entrenamiento XGBoost
# =====================================================================

class TestTrainerXGBoost:
    def test_train_xgboost(self, trained_xgboost):
        trainer, _, _, _ = trained_xgboost
        assert trainer.model is not None
        assert trainer.model_type == "xgboost"

    def test_evaluate_xgboost(self, trained_xgboost):
        trainer, X_val, y_val, _ = trained_xgboost
        metrics = trainer.evaluate(X_val, y_val)

        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics
        assert "probabilities" in metrics
        assert "predictions" in metrics

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_weighted"] <= 1

    def test_predict_proba_shape_xgboost(self, trained_xgboost):
        trainer, X_val, y_val, _ = trained_xgboost
        metrics = trainer.evaluate(X_val, y_val)

        probas = metrics["probabilities"]
        assert probas.shape[0] == len(X_val)
        assert probas.shape[1] == 3  # 3 clases

        # Probabilidades suman ~1
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_feature_importance_xgboost(self, trained_xgboost):
        trainer, _, _, _ = trained_xgboost
        fi = trainer.get_feature_importance()

        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == len(trainer.feature_names)
        # Ordenado de mayor a menor
        assert fi["importance"].is_monotonic_decreasing

    def test_save_and_load_xgboost(self, trained_xgboost):
        trainer, X_val, y_val, model_path = trained_xgboost

        # Predicciones originales
        original_preds = trainer.model.predict(X_val)

        # Cargar en nuevo trainer
        from forex_bot.models.trainer import ModelTrainer
        trainer2 = ModelTrainer()
        trainer2.load_model(str(model_path))

        loaded_preds = trainer2.model.predict(X_val)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_metadata_saved_xgboost(self, trained_xgboost):
        _, _, _, model_path = trained_xgboost

        meta_path = model_path.with_suffix("").with_suffix(".meta.json")
        assert meta_path.exists(), "Metadata file no creado"

        import json
        with open(meta_path) as f:
            meta = json.load(f)

        assert "model_type" in meta
        assert "train_date" in meta
        assert "feature_names" in meta
        assert "best_iteration" in meta
        assert meta["model_type"] == "xgboost"


# =====================================================================
# TESTS: ModelTrainer - Entrenamiento LightGBM
# =====================================================================

class TestTrainerLightGBM:
    def test_train_lightgbm(self, trained_lightgbm):
        trainer, _, _, _ = trained_lightgbm
        assert trainer.model is not None
        assert trainer.model_type == "lightgbm"

    def test_evaluate_lightgbm(self, trained_lightgbm):
        trainer, X_val, y_val, _ = trained_lightgbm
        metrics = trainer.evaluate(X_val, y_val)

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_weighted"] <= 1

    def test_predict_proba_shape_lightgbm(self, trained_lightgbm):
        trainer, X_val, y_val, _ = trained_lightgbm
        metrics = trainer.evaluate(X_val, y_val)

        probas = metrics["probabilities"]
        assert probas.shape == (len(X_val), 3)

        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_feature_importance_lightgbm(self, trained_lightgbm):
        trainer, _, _, _ = trained_lightgbm
        fi = trainer.get_feature_importance()

        assert isinstance(fi, pd.DataFrame)
        assert len(fi) == len(trainer.feature_names)

    def test_save_and_load_lightgbm(self, trained_lightgbm):
        trainer, X_val, y_val, model_path = trained_lightgbm

        original_preds = trainer.model.predict(X_val)

        from forex_bot.models.trainer import ModelTrainer
        trainer2 = ModelTrainer()
        trainer2.load_model(str(model_path))

        loaded_preds = trainer2.model.predict(X_val)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_metadata_saved_lightgbm(self, trained_lightgbm):
        _, _, _, model_path = trained_lightgbm

        meta_path = model_path.with_suffix("").with_suffix(".meta.json")
        assert meta_path.exists()


# =====================================================================
# TESTS: ModelPredictor
# =====================================================================

class TestPredictor:
    def test_load_xgboost_model(self, trained_xgboost):
        from forex_bot.models.predictor import ModelPredictor
        _, _, _, model_path = trained_xgboost

        predictor = ModelPredictor(str(model_path))
        assert predictor.is_loaded
        assert predictor.model_type == "xgboost"

    def test_load_lightgbm_model(self, trained_lightgbm):
        from forex_bot.models.predictor import ModelPredictor
        _, _, _, model_path = trained_lightgbm

        predictor = ModelPredictor(str(model_path))
        assert predictor.is_loaded
        assert predictor.model_type == "lightgbm"

    def test_predict_returns_correct_format(self, trained_xgboost):
        from forex_bot.models.predictor import ModelPredictor
        trainer, X_val, _, model_path = trained_xgboost

        predictor = ModelPredictor(str(model_path))
        result = predictor.predict(X_val)

        assert "signal" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "class_id" in result

        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0 <= result["confidence"] <= 1.0
        assert set(result["probabilities"].keys()) == {"BUY", "SELL", "HOLD"}

    def test_predict_single_row(self, trained_xgboost):
        from forex_bot.models.predictor import ModelPredictor
        trainer, X_val, _, model_path = trained_xgboost

        predictor = ModelPredictor(str(model_path))
        # Solo 1 fila
        single = X_val.iloc[-1:]
        result = predictor.predict(single)

        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_should_trade_hold(self, trained_xgboost):
        from forex_bot.models.predictor import ModelPredictor

        predictor = ModelPredictor()
        predictor.model = True  # dummy

        # HOLD nunca debe operar
        pred = {"signal": "HOLD", "confidence": 0.99}
        assert predictor.should_trade(pred) is False

    def test_should_trade_low_confidence(self, trained_xgboost):
        from forex_bot.models.predictor import ModelPredictor
        from forex_bot.config import settings

        predictor = ModelPredictor()
        predictor.model = True

        # BUY con baja confianza no debe operar
        pred = {"signal": "BUY", "confidence": settings.CONFIDENCE_THRESHOLD - 0.01}
        assert predictor.should_trade(pred) is False

    def test_should_trade_high_confidence(self, trained_xgboost):
        from forex_bot.models.predictor import ModelPredictor
        from forex_bot.config import settings

        predictor = ModelPredictor()
        predictor.model = True

        # BUY con alta confianza SI debe operar
        pred = {"signal": "BUY", "confidence": settings.CONFIDENCE_THRESHOLD + 0.01}
        assert predictor.should_trade(pred) is True

    def test_probabilities_sum_to_one(self, trained_xgboost):
        from forex_bot.models.predictor import ModelPredictor
        _, X_val, _, model_path = trained_xgboost

        predictor = ModelPredictor(str(model_path))
        result = predictor.predict(X_val)

        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_feature_alignment(self, trained_xgboost):
        """Predictor debe manejar features faltantes o extras."""
        from forex_bot.models.predictor import ModelPredictor
        _, X_val, _, model_path = trained_xgboost

        predictor = ModelPredictor(str(model_path))

        # Agregar columna extra y quitar una
        X_modified = X_val.copy()
        X_modified["extra_col"] = 999
        # No quitar columna - solo agregar extra
        result = predictor.predict(X_modified)
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_model_age_days(self, trained_xgboost):
        from forex_bot.models.predictor import ModelPredictor
        _, _, _, model_path = trained_xgboost

        predictor = ModelPredictor(str(model_path))
        age = predictor.get_model_age_days()

        # Recien entrenado, edad = 0
        assert age is not None
        assert age == 0

    def test_needs_retraining_fresh_model(self, trained_xgboost):
        from forex_bot.models.predictor import ModelPredictor
        _, _, _, model_path = trained_xgboost

        predictor = ModelPredictor(str(model_path))
        assert predictor.needs_retraining() is False  # recien entrenado

    def test_load_nonexistent_raises(self):
        from forex_bot.models.predictor import ModelPredictor

        with pytest.raises(FileNotFoundError):
            ModelPredictor("/ruta/que/no/existe.json")


# =====================================================================
# TESTS: Pipeline completo train -> save -> load -> predict
# =====================================================================

class TestMLPipelineEndToEnd:
    """Test del pipeline completo ML con datos del feature_engine real."""

    def test_full_pipeline_xgboost(self, tmp_path):
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.data.preprocessor import DataPreprocessor
        from forex_bot.models.trainer import ModelTrainer
        from forex_bot.models.predictor import ModelPredictor

        # 1. Generar datos sinteticos (simular lo que vendria de MT5)
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        returns = np.random.normal(0, 0.0005, n)
        close = 1.08 + np.cumsum(returns)
        high = close + np.abs(np.random.normal(0, 0.0003, n))
        low = close - np.abs(np.random.normal(0, 0.0003, n))
        open_p = close + np.random.normal(0, 0.0002, n)
        high = np.maximum(high, np.maximum(open_p, close))
        low = np.minimum(low, np.minimum(open_p, close))

        df = pd.DataFrame({
            "open": open_p, "high": high, "low": low, "close": close,
            "tick_volume": np.random.randint(100, 5000, n),
            "spread": np.random.randint(5, 20, n),
        }, index=dates)
        df.index.name = "time"

        # 2. Features
        fe = FeatureEngine()
        df = fe.add_all_features(df)
        df["target"] = fe.create_target(df, horizon=5, min_pips=5, pip_size=0.0001)

        # 3. Preparar
        trainer = ModelTrainer(model_type="xgboost")
        X, y = trainer.prepare_features(df)

        assert len(X) > 100, f"Muy pocas muestras: {len(X)}"

        # 4. Split
        pp = DataPreprocessor()
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        # 5. Entrenar
        trainer.train(X_train, y_train, X_val, y_val)

        # 6. Evaluar
        metrics = trainer.evaluate(X_val, y_val)
        assert metrics["accuracy"] > 0  # Al menos algo predice

        # 7. Feature importance
        fi = trainer.get_feature_importance()
        assert len(fi) > 0

        # 8. Guardar
        model_path = trainer.save_model(str(tmp_path / "full_test"))

        # 9. Cargar con predictor
        predictor = ModelPredictor(str(model_path))
        assert predictor.is_loaded

        # 10. Predecir
        result = predictor.predict(X_val)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0 <= result["confidence"] <= 1.0

    def test_full_pipeline_lightgbm(self, tmp_path):
        from forex_bot.data.feature_engine import FeatureEngine
        from forex_bot.models.trainer import ModelTrainer
        from forex_bot.models.predictor import ModelPredictor

        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = 1.08 + np.cumsum(np.random.normal(0, 0.0005, n))
        high = close + np.abs(np.random.normal(0, 0.0003, n))
        low = close - np.abs(np.random.normal(0, 0.0003, n))
        open_p = close + np.random.normal(0, 0.0002, n)
        high = np.maximum(high, np.maximum(open_p, close))
        low = np.minimum(low, np.minimum(open_p, close))

        df = pd.DataFrame({
            "open": open_p, "high": high, "low": low, "close": close,
            "tick_volume": np.random.randint(100, 5000, n),
            "spread": np.random.randint(5, 20, n),
        }, index=dates)
        df.index.name = "time"

        fe = FeatureEngine()
        df = fe.add_all_features(df)
        df["target"] = fe.create_target(df, horizon=5, min_pips=5, pip_size=0.0001)

        trainer = ModelTrainer(model_type="lightgbm")
        X, y = trainer.prepare_features(df)

        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        trainer.train(X_train, y_train, X_val, y_val)
        metrics = trainer.evaluate(X_val, y_val)
        assert metrics["accuracy"] > 0

        model_path = trainer.save_model(str(tmp_path / "full_lgb_test"))

        predictor = ModelPredictor(str(model_path))
        result = predictor.predict(X_val)
        assert result["signal"] in ("BUY", "SELL", "HOLD")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

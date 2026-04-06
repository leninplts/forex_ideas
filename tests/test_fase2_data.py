"""
VALIDACION FASE 2 - Data Pipeline
Testea collector (sin MT5), preprocessor y feature engine con datos sinteticos.
Ejecutar: python -m pytest tests/test_fase2_data.py -v
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =====================================================================
# FIXTURE: datos OHLCV sinteticos que simulan velas de Forex
# =====================================================================

@pytest.fixture
def sample_ohlcv():
    """Generar ~500 velas H1 sinteticas que simulan EURUSD."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    # Simular random walk para el precio
    returns = np.random.normal(0, 0.0005, n)  # ~5 pips std por vela
    close = 1.0800 + np.cumsum(returns)

    # Generar OHLC realista
    high = close + np.abs(np.random.normal(0, 0.0003, n))
    low = close - np.abs(np.random.normal(0, 0.0003, n))
    open_price = close + np.random.normal(0, 0.0002, n)

    # Asegurar consistencia: high >= max(open,close), low <= min(open,close)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.randint(100, 5000, n)

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": volume,
        "spread": np.random.randint(5, 20, n),
    }, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def sample_ohlcv_dirty(sample_ohlcv):
    """Datos con problemas: duplicados, NaN, OHLC inconsistente."""
    df = sample_ohlcv.copy()

    # Agregar duplicados
    dup_row = df.iloc[10:11].copy()
    df = pd.concat([df, dup_row])

    # Agregar NaN
    df.loc[df.index[20], "close"] = np.nan
    df.loc[df.index[21], "high"] = np.nan

    # Hacer OHLC inconsistente (high < close en una fila)
    idx = df.index[30]
    df.loc[idx, "high"] = df.loc[idx, "close"] - 0.001

    return df


@pytest.fixture
def sample_ohlcv_h4():
    """Velas H4 para test multi-timeframe."""
    np.random.seed(99)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")

    returns = np.random.normal(0, 0.001, n)
    close = 1.0800 + np.cumsum(returns)
    high = close + np.abs(np.random.normal(0, 0.0005, n))
    low = close - np.abs(np.random.normal(0, 0.0005, n))
    open_price = close + np.random.normal(0, 0.0003, n)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": np.random.randint(500, 10000, n),
        "spread": np.random.randint(5, 20, n),
    }, index=dates)
    df.index.name = "time"
    return df


# =====================================================================
# TESTS: DataCollector (solo lo que no requiere MT5)
# =====================================================================

class TestDataCollectorImport:
    """Verificar que el collector se importa correctamente."""

    def test_import(self):
        from forex_bot.data.collector import DataCollector
        collector = DataCollector()
        assert collector is not None
        assert collector.is_connected is False

    def test_save_and_load_data(self, sample_ohlcv, tmp_path):
        from forex_bot.data.collector import DataCollector
        from forex_bot.config import settings

        # Temporalmente usar tmp_path como DATA_DIR
        original_dir = settings.DATA_DIR
        settings.DATA_DIR = tmp_path

        collector = DataCollector()
        filepath = collector.save_data(sample_ohlcv, "test_eurusd_h1")
        assert filepath.exists()

        loaded = collector.load_data("test_eurusd_h1")
        assert loaded is not None
        assert len(loaded) == len(sample_ohlcv)
        assert list(loaded.columns) == list(sample_ohlcv.columns)

        # Restaurar
        settings.DATA_DIR = original_dir

    def test_load_nonexistent(self, tmp_path):
        from forex_bot.data.collector import DataCollector
        from forex_bot.config import settings

        original_dir = settings.DATA_DIR
        settings.DATA_DIR = tmp_path

        collector = DataCollector()
        result = collector.load_data("no_existe")
        assert result is None

        settings.DATA_DIR = original_dir


# =====================================================================
# TESTS: DataPreprocessor
# =====================================================================

class TestPreprocessorClean:
    """Tests de limpieza de datos."""

    def test_clean_removes_duplicates(self, sample_ohlcv_dirty):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        clean = pp.clean_data(sample_ohlcv_dirty)
        assert not clean.index.duplicated().any(), "Aun hay duplicados"

    def test_clean_removes_nan(self, sample_ohlcv_dirty):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        clean = pp.clean_data(sample_ohlcv_dirty)
        ohlc = ["open", "high", "low", "close"]
        assert not clean[ohlc].isna().any().any(), "Aun hay NaN en OHLC"

    def test_clean_fixes_ohlc_consistency(self, sample_ohlcv_dirty):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        clean = pp.clean_data(sample_ohlcv_dirty)
        # high >= max(open, close) para todas las filas
        assert (clean["high"] >= clean[["open", "close"]].max(axis=1)).all()
        # low <= min(open, close) para todas las filas
        assert (clean["low"] <= clean[["open", "close"]].min(axis=1)).all()

    def test_clean_preserves_order(self, sample_ohlcv):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        clean = pp.clean_data(sample_ohlcv)
        assert clean.index.is_monotonic_increasing

    def test_clean_data_is_copy(self, sample_ohlcv):
        """clean_data no debe modificar el DataFrame original."""
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        original_len = len(sample_ohlcv)
        clean = pp.clean_data(sample_ohlcv)
        assert len(sample_ohlcv) == original_len


class TestPreprocessorFillGaps:
    def test_fill_ffill(self, sample_ohlcv):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        # Introducir NaN
        df = sample_ohlcv.copy()
        df.loc[df.index[50], "close"] = np.nan
        filled = pp.fill_gaps(df, method="ffill")
        assert not filled["close"].isna().any()

    def test_fill_interpolate(self, sample_ohlcv):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        df = sample_ohlcv.copy()
        df.loc[df.index[50], "close"] = np.nan
        filled = pp.fill_gaps(df, method="interpolate")
        assert not filled["close"].isna().any()


class TestPreprocessorNormalize:
    def test_zscore_normalization(self, sample_ohlcv):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        cols = ["close"]
        normed = pp.normalize_data(sample_ohlcv, cols, method="zscore")

        # Mean ~= 0, std ~= 1
        assert abs(normed["close"].mean()) < 0.01
        assert abs(normed["close"].std() - 1.0) < 0.01

    def test_minmax_normalization(self, sample_ohlcv):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        cols = ["close"]
        normed = pp.normalize_data(sample_ohlcv, cols, method="minmax")

        # Rango [0, 1]
        assert normed["close"].min() >= -0.001
        assert normed["close"].max() <= 1.001

    def test_denormalize_recovers_original(self, sample_ohlcv):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        cols = ["close"]
        original_close = sample_ohlcv["close"].copy()

        normed = pp.normalize_data(sample_ohlcv, cols, method="zscore")
        recovered = pp.denormalize(normed, cols)

        np.testing.assert_allclose(recovered["close"].values, original_close.values, rtol=1e-10)

    def test_scalers_stored(self, sample_ohlcv):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        pp.normalize_data(sample_ohlcv, ["close"], method="zscore")
        assert "close" in pp.scalers
        assert "mean" in pp.scalers["close"]
        assert "std" in pp.scalers["close"]


class TestPreprocessorSplit:
    def test_split_temporal(self, sample_ohlcv):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        train, val, test = pp.split_data(sample_ohlcv, train_ratio=0.7, val_ratio=0.15)

        # Proporciones aproximadas
        total = len(sample_ohlcv)
        assert abs(len(train) / total - 0.7) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02

        # No se pierde ni se duplica data
        assert len(train) + len(val) + len(test) == total

        # Orden temporal: train < val < test
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_walk_forward_splits(self, sample_ohlcv):
        from forex_bot.data.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        # Con 500 velas H1 (~21 dias), usar ventanas pequenas
        splits = pp.create_walk_forward_splits(
            sample_ohlcv,
            train_window_days=10,
            val_window_days=3,
            step_days=2,
        )

        assert len(splits) > 0, "Debe generar al menos 1 split"

        for train_df, val_df in splits:
            # Train va antes de val temporalmente
            assert train_df.index[-1] <= val_df.index[0]
            # Ambos tienen datos
            assert len(train_df) > 0
            assert len(val_df) > 0


# =====================================================================
# TESTS: FeatureEngine
# =====================================================================

class TestFeatureEngineTrend:
    """Test features de tendencia."""

    def test_add_sma(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_sma(sample_ohlcv.copy())

        for p in [20, 50, 200]:
            assert f"sma_{p}" in df.columns
            assert f"sma_{p}_dist" in df.columns
            assert f"sma_{p}_slope" in df.columns

        # Verificar cruce SMA
        assert "sma_cross_20_50" in df.columns
        assert df["sma_cross_20_50"].isin([0, 1]).all()

    def test_add_ema(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_ema(sample_ohlcv.copy())

        for p in [9, 21]:
            assert f"ema_{p}" in df.columns
            assert f"ema_{p}_dist" in df.columns

        assert "ema_cross_9_21" in df.columns

    def test_add_adx(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_adx(sample_ohlcv.copy())

        assert "adx_14" in df.columns
        assert "di_plus_14" in df.columns
        assert "di_minus_14" in df.columns
        assert "adx_strong_trend" in df.columns

        # ADX esta entre 0 y 100 (donde no es NaN)
        valid_adx = df["adx_14"].dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()


class TestFeatureEngineMomentum:
    """Test features de momentum."""

    def test_add_rsi(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_rsi(sample_ohlcv.copy())

        assert "rsi_14" in df.columns
        assert "rsi_overbought" in df.columns
        assert "rsi_oversold" in df.columns
        assert "rsi_slope" in df.columns

        # RSI entre 0 y 100
        valid_rsi = df["rsi_14"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_add_stochastic(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_stochastic(sample_ohlcv.copy())

        assert "stoch_k" in df.columns
        assert "stoch_d" in df.columns
        assert "stoch_cross" in df.columns

    def test_add_macd(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_macd(sample_ohlcv.copy())

        assert "macd_line" in df.columns
        assert "macd_histogram" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_cross" in df.columns
        assert "macd_hist_positive" in df.columns

    def test_add_cci(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_cci(sample_ohlcv.copy())

        assert "cci_20" in df.columns
        assert "cci_overbought" in df.columns
        assert "cci_oversold" in df.columns


class TestFeatureEngineVolatility:
    """Test features de volatilidad."""

    def test_add_atr(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_atr(sample_ohlcv.copy())

        assert "atr_14" in df.columns
        assert "atr_pct" in df.columns
        assert "atr_ratio" in df.columns

        # ATR siempre positivo
        valid_atr = df["atr_14"].dropna()
        assert (valid_atr >= 0).all()

    def test_add_bollinger(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_bollinger(sample_ohlcv.copy())

        assert "bb_upper" in df.columns
        assert "bb_middle" in df.columns
        assert "bb_lower" in df.columns
        assert "bb_pct_b" in df.columns
        assert "bb_squeeze" in df.columns

        # Upper > middle > lower
        valid = df[["bb_upper", "bb_middle", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()


class TestFeatureEngineVolume:
    def test_add_volume_features(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_volume_features(sample_ohlcv.copy())

        assert "volume_ma" in df.columns
        assert "volume_ratio" in df.columns
        assert "obv" in df.columns


class TestFeatureEnginePriceAction:
    def test_add_price_action(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_price_action(sample_ohlcv.copy())

        # Retornos
        for p in [1, 3, 5, 10]:
            assert f"return_{p}" in df.columns

        # Candle anatomy
        assert "candle_range_pct" in df.columns
        assert "candle_body_pct" in df.columns
        assert "upper_shadow_pct" in df.columns
        assert "lower_shadow_pct" in df.columns

        # Structure
        assert "higher_high" in df.columns
        assert "lower_low" in df.columns
        assert "bullish_candle" in df.columns

        # Percentages deben ser >= 0
        valid = df["candle_body_pct"].dropna()
        assert (valid >= 0).all()


class TestFeatureEngineMultiTimeframe:
    def test_add_htf_features(self, sample_ohlcv, sample_ohlcv_h4):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_higher_timeframe_features(sample_ohlcv, sample_ohlcv_h4, suffix="h4")

        assert "rsi_h4" in df.columns
        assert "trend_h4" in df.columns
        assert "atr_h4" in df.columns

        # No debe haber mas filas que el original
        assert len(df) == len(sample_ohlcv)


class TestFeatureEngineTarget:
    def test_create_target(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        target = fe.create_target(sample_ohlcv, horizon=5, min_pips=15, pip_size=0.0001)

        # Mismo tamanho que el DataFrame
        assert len(target) == len(sample_ohlcv)

        # Solo valores validos: -1, 0, 1, NaN
        valid = target.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

        # Las ultimas N filas deben ser NaN (sin datos futuros)
        assert target.iloc[-5:].isna().all()

        # Debe haber al menos algunas senales (no todo HOLD)
        assert (valid != 0).any(), "Todas las senales son HOLD, ajustar min_pips"

    def test_target_no_future_data(self, sample_ohlcv):
        """El target usa shift(-N), verificar que no filtra datos futuros."""
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        target = fe.create_target(sample_ohlcv, horizon=5, min_pips=1, pip_size=0.0001)

        # Las ultimas 5 filas DEBEN ser NaN
        assert target.iloc[-5:].isna().all()
        # Las primeras filas NO deben ser NaN
        assert target.iloc[0:10].notna().all()


class TestFeatureEngineAllFeatures:
    """Test del pipeline completo de features."""

    def test_add_all_features(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv)

        # Debe tener muchas mas columnas que el original
        assert len(df.columns) > len(sample_ohlcv.columns) + 20

        # feature_names debe estar poblado
        assert len(fe.feature_names) > 20

        # Columnas originales deben seguir
        for col in ["open", "high", "low", "close", "tick_volume"]:
            assert col in df.columns

    def test_feature_names_match_columns(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv)

        # Cada feature name debe ser una columna del DataFrame
        for name in fe.feature_names:
            assert name in df.columns, f"Feature '{name}' no esta en el DataFrame"

    def test_get_feature_columns(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv)

        feature_cols = fe.get_feature_columns(df)
        assert len(feature_cols) > 20
        # No debe incluir OHLCV
        for col in ["open", "high", "low", "close", "tick_volume"]:
            assert col not in feature_cols

    def test_no_inf_values(self, sample_ohlcv):
        """Las features no deben tener valores infinitos."""
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv)

        feature_cols = fe.get_feature_columns(df)
        for col in feature_cols:
            inf_count = np.isinf(df[col].dropna()).sum()
            assert inf_count == 0, f"Feature '{col}' tiene {inf_count} valores infinitos"

    def test_remove_lookahead_bias(self, sample_ohlcv):
        from forex_bot.data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = fe.add_all_features(sample_ohlcv)
        df["target"] = fe.create_target(df, horizon=5, min_pips=5, pip_size=0.0001)

        # No debe arrojar error
        clean = fe.remove_lookahead_bias(df)
        assert clean is not None


# =====================================================================
# TEST: Pipeline completo end-to-end
# =====================================================================

class TestPipelineEndToEnd:
    """Test del pipeline completo: clean -> features -> target -> split."""

    def test_full_pipeline(self, sample_ohlcv_dirty, sample_ohlcv_h4):
        from forex_bot.data.preprocessor import DataPreprocessor
        from forex_bot.data.feature_engine import FeatureEngine

        pp = DataPreprocessor()
        fe = FeatureEngine()

        # 1. Limpiar
        df = pp.clean_data(sample_ohlcv_dirty)
        assert not df.index.duplicated().any()

        # 2. Features
        df = fe.add_all_features(df)
        assert len(fe.feature_names) > 20

        # 3. Multi-timeframe
        df = fe.add_higher_timeframe_features(df, sample_ohlcv_h4, suffix="h4")
        assert "rsi_h4" in df.columns

        # 4. Target
        df["target"] = fe.create_target(df, horizon=5, min_pips=5, pip_size=0.0001)

        # 5. Verificar lookahead
        df = fe.remove_lookahead_bias(df)

        # 6. Drop NaN (de indicadores con lookback)
        df_clean = df.dropna()
        assert len(df_clean) > 100, f"Solo quedan {len(df_clean)} filas despues de dropna"

        # 7. Split
        train, val, test = pp.split_data(df_clean)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

        # 8. Verificar que features son numericas
        feature_cols = fe.get_feature_columns(df_clean)
        for col in feature_cols:
            assert df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int], \
                f"Feature '{col}' no es numerica: {df_clean[col].dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Data Preprocessor - Limpieza, normalizacion y splitting de datos.
Prepara los datos crudos para el modelo de ML.
"""
import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Limpieza, normalizacion y splitting temporal de datos de mercado.
    """

    def __init__(self):
        # Almacenar scalers para poder des-normalizar despues
        self._scalers = {}

    # ------------------------------------------------------------------
    # Limpieza
    # ------------------------------------------------------------------

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpiar datos crudos OHLCV.
          - Eliminar filas duplicadas por timestamp
          - Eliminar filas con NaN en OHLC
          - Verificar consistencia OHLC (high >= max(open,close), etc.)
          - Reportar gaps inusuales

        Args:
            df: DataFrame con index DatetimeIndex y columnas OHLCV

        Returns:
            DataFrame limpio (copia, no modifica el original).
        """
        df = df.copy()
        initial_len = len(df)

        # 1. Eliminar duplicados por timestamp
        duplicates = df.index.duplicated(keep="first")
        if duplicates.any():
            n_dup = duplicates.sum()
            df = df[~duplicates]
            logger.warning("Eliminados %d timestamps duplicados", n_dup)

        # 2. Ordenar por tiempo
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)
            logger.info("Datos reordenados cronologicamente")

        # 3. Eliminar filas con NaN en OHLC
        ohlc = ["open", "high", "low", "close"]
        ohlc_present = [c for c in ohlc if c in df.columns]
        nan_mask = df[ohlc_present].isna().any(axis=1)
        if nan_mask.any():
            n_nan = nan_mask.sum()
            df = df[~nan_mask]
            logger.warning("Eliminadas %d filas con NaN en OHLC", n_nan)

        # 4. Verificar consistencia OHLC
        if all(c in df.columns for c in ohlc):
            inconsistent = (
                (df["high"] < df[["open", "close"]].max(axis=1)) |
                (df["low"] > df[["open", "close"]].min(axis=1)) |
                (df["high"] < df["low"])
            )
            if inconsistent.any():
                n_bad = inconsistent.sum()
                logger.warning(
                    "Detectadas %d velas con OHLC inconsistente (high < close, etc.). Corrigiendo...",
                    n_bad,
                )
                # Corregir: ajustar high/low para que sean consistentes
                df.loc[inconsistent, "high"] = df.loc[inconsistent, ["open", "high", "low", "close"]].max(axis=1)
                df.loc[inconsistent, "low"] = df.loc[inconsistent, ["open", "high", "low", "close"]].min(axis=1)

        # 5. Detectar gaps inusuales
        self._report_gaps(df)

        removed = initial_len - len(df)
        if removed > 0:
            logger.info("Limpieza completada: %d filas eliminadas (%d -> %d)", removed, initial_len, len(df))
        else:
            logger.info("Limpieza completada: datos limpios (%d filas)", len(df))

        return df

    def _report_gaps(self, df: pd.DataFrame):
        """Detectar y reportar gaps temporales inusuales."""
        if len(df) < 2:
            return

        time_diffs = df.index.to_series().diff()
        median_diff = time_diffs.median()

        # Un gap es "inusual" si es > 5x la mediana y no es fin de semana
        threshold = median_diff * 5
        large_gaps = time_diffs[time_diffs > threshold]

        for gap_time, gap_size in large_gaps.items():
            # Verificar si es fin de semana (normal en Forex)
            if gap_time.weekday() in (0, 6):  # Lunes o domingo
                continue
            logger.warning(
                "Gap inusual detectado: %s (duracion: %s)",
                gap_time.strftime("%Y-%m-%d %H:%M"),
                gap_size,
            )

    # ------------------------------------------------------------------
    # Fill gaps
    # ------------------------------------------------------------------

    def fill_gaps(self, df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """
        Rellenar gaps pequenos en los datos.

        Args:
            df: DataFrame con datos OHLCV
            method: Metodo de relleno - "ffill" (forward fill) o "interpolate"

        Returns:
            DataFrame con gaps rellenados.
        """
        df = df.copy()

        if method == "ffill":
            # Forward fill: copiar la vela anterior (max 3 velas)
            filled = df.ffill(limit=3)
        elif method == "interpolate":
            filled = df.interpolate(method="time", limit=3)
        else:
            logger.error("Metodo de fill desconocido: %s", method)
            return df

        n_filled = filled.notna().sum().sum() - df.notna().sum().sum()
        if n_filled > 0:
            logger.info("Rellenados %d valores con metodo '%s'", n_filled, method)

        return filled

    # ------------------------------------------------------------------
    # Normalizacion
    # ------------------------------------------------------------------

    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: list,
        method: str = "zscore",
    ) -> pd.DataFrame:
        """
        Normalizar columnas especificas del DataFrame.

        Args:
            df: DataFrame con features
            columns: Lista de columnas a normalizar
            method: "zscore" (mean=0, std=1) o "minmax" (0-1)

        Returns:
            DataFrame con columnas normalizadas (copia).
        """
        df = df.copy()
        cols_to_norm = [c for c in columns if c in df.columns]

        for col in cols_to_norm:
            series = df[col].dropna()

            if method == "zscore":
                mean = series.mean()
                std = series.std()
                if std == 0:
                    logger.warning("Std=0 para columna '%s', skip normalizacion", col)
                    continue
                df[col] = (df[col] - mean) / std
                self._scalers[col] = {"method": "zscore", "mean": mean, "std": std}

            elif method == "minmax":
                min_val = series.min()
                max_val = series.max()
                range_val = max_val - min_val
                if range_val == 0:
                    logger.warning("Range=0 para columna '%s', skip normalizacion", col)
                    continue
                df[col] = (df[col] - min_val) / range_val
                self._scalers[col] = {"method": "minmax", "min": min_val, "max": max_val}

            else:
                logger.error("Metodo de normalizacion desconocido: %s", method)
                return df

        logger.info("Normalizadas %d columnas con metodo '%s'", len(cols_to_norm), method)
        return df

    def denormalize(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Des-normalizar columnas usando los scalers guardados.

        Args:
            df: DataFrame con datos normalizados
            columns: Lista de columnas a des-normalizar

        Returns:
            DataFrame des-normalizado.
        """
        df = df.copy()

        for col in columns:
            if col not in self._scalers:
                logger.warning("No hay scaler guardado para '%s'", col)
                continue

            scaler = self._scalers[col]
            if scaler["method"] == "zscore":
                df[col] = df[col] * scaler["std"] + scaler["mean"]
            elif scaler["method"] == "minmax":
                df[col] = df[col] * (scaler["max"] - scaler["min"]) + scaler["min"]

        return df

    @property
    def scalers(self) -> dict:
        """Retorna los scalers guardados para persistencia."""
        return self._scalers.copy()

    # ------------------------------------------------------------------
    # Splitting temporal
    # ------------------------------------------------------------------

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple:
        """
        Split temporal de datos (NO aleatorio).

        Args:
            df: DataFrame ordenado cronologicamente
            train_ratio: Proporcion para entrenamiento (default 70%)
            val_ratio: Proporcion para validacion (default 15%)
            El resto es test (default 15%)

        Returns:
            Tuple de (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(
            "Split temporal: train=%d (%.0f%%) | val=%d (%.0f%%) | test=%d (%.0f%%)",
            len(train_df), len(train_df) / n * 100,
            len(val_df), len(val_df) / n * 100,
            len(test_df), len(test_df) / n * 100,
        )
        logger.info(
            "  Train: %s -> %s",
            train_df.index[0].strftime("%Y-%m-%d"),
            train_df.index[-1].strftime("%Y-%m-%d"),
        )
        logger.info(
            "  Val:   %s -> %s",
            val_df.index[0].strftime("%Y-%m-%d"),
            val_df.index[-1].strftime("%Y-%m-%d"),
        )
        logger.info(
            "  Test:  %s -> %s",
            test_df.index[0].strftime("%Y-%m-%d"),
            test_df.index[-1].strftime("%Y-%m-%d"),
        )

        return train_df, val_df, test_df

    def create_walk_forward_splits(
        self,
        df: pd.DataFrame,
        train_window_days: int = 365,
        val_window_days: int = 60,
        step_days: int = 30,
    ) -> list:
        """
        Generar splits para Walk-Forward Analysis.
        Esto es CRITICO para evitar overfitting en trading.

        Ejemplo con train=365d, val=60d, step=30d:
          Split 1: Train [Jan2023 - Dec2023] -> Val [Jan2024 - Feb2024]
          Split 2: Train [Feb2023 - Jan2024] -> Val [Feb2024 - Mar2024]
          Split 3: Train [Mar2023 - Feb2024] -> Val [Mar2024 - Apr2024]
          ...

        Args:
            df: DataFrame completo ordenado cronologicamente
            train_window_days: Dias de entrenamiento
            val_window_days: Dias de validacion
            step_days: Dias de avance entre splits

        Returns:
            Lista de tuples (train_df, val_df) para cada split.
        """
        splits = []
        start_date = df.index[0]
        end_date = df.index[-1]

        train_delta = timedelta(days=train_window_days)
        val_delta = timedelta(days=val_window_days)
        step_delta = timedelta(days=step_days)

        current_start = start_date

        while True:
            train_end = current_start + train_delta
            val_start = train_end
            val_end = val_start + val_delta

            # Verificar que hay datos suficientes
            if val_end > end_date:
                break

            train_df = df[(df.index >= current_start) & (df.index < train_end)].copy()
            val_df = df[(df.index >= val_start) & (df.index < val_end)].copy()

            # Minimo de datos para que sea valido
            if len(train_df) >= 100 and len(val_df) >= 10:
                splits.append((train_df, val_df))

            current_start += step_delta

        logger.info(
            "Walk-forward: %d splits generados (train=%dd, val=%dd, step=%dd)",
            len(splits), train_window_days, val_window_days, step_days,
        )
        if splits:
            logger.info(
                "  Primer split: train hasta %s, val hasta %s",
                splits[0][0].index[-1].strftime("%Y-%m-%d"),
                splits[0][1].index[-1].strftime("%Y-%m-%d"),
            )
            logger.info(
                "  Ultimo split: train hasta %s, val hasta %s",
                splits[-1][0].index[-1].strftime("%Y-%m-%d"),
                splits[-1][1].index[-1].strftime("%Y-%m-%d"),
            )

        return splits

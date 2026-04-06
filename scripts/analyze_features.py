"""
Feature Analysis con SHAP - Analisis de importancia de features para el modelo ML.

Objetivo: Entender QUE features aportan valor y cuales son ruido.
Esto es critico antes de hacer feature selection (Fase 2).

Analisis:
  1. Feature importance del modelo (gain-based)
  2. SHAP values (impacto real de cada feature en las predicciones)
  3. Correlacion entre features (detectar multicolinealidad)
  4. Correlacion feature-target (detectar features sin señal)
  5. Recomendaciones de features a eliminar

Uso:
  python scripts/analyze_features.py
  python scripts/analyze_features.py --symbol EURUSDm --top 20
  python scripts/analyze_features.py --symbol GBPUSDm --save-plots
"""
import argparse
import logging
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from forex_bot.config import settings
from forex_bot.data.preprocessor import DataPreprocessor
from forex_bot.data.feature_engine import FeatureEngine
from forex_bot.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def load_prepared_data(symbol: str, model_type: str = None) -> tuple:
    """
    Cargar datos y preparar features + target para analisis.
    
    Returns:
        Tuple (X_train, y_train, X_test, y_test, trainer, feature_names)
    """
    model_type = model_type or settings.MODEL_TYPE
    data_dir = settings.DATA_DIR
    pp = DataPreprocessor()
    fe = FeatureEngine()
    
    # Cargar H1
    h1_path = data_dir / f"{symbol}_H1.csv"
    if not h1_path.exists():
        print(f"ERROR: No se encontro {h1_path}")
        sys.exit(1)
    
    df = pd.read_csv(h1_path, index_col="time", parse_dates=True)
    df = pp.clean_data(df)
    df = fe.add_all_features(df)
    
    # Cargar H4, D1
    h4_path = data_dir / f"{symbol}_H4.csv"
    d1_path = data_dir / f"{symbol}_D1.csv"
    
    if h4_path.exists():
        df_h4 = pd.read_csv(h4_path, index_col="time", parse_dates=True)
        df_h4 = pp.clean_data(df_h4)
        df = fe.add_higher_timeframe_features(df, df_h4, suffix="h4")
    
    if d1_path.exists():
        df_d1 = pd.read_csv(d1_path, index_col="time", parse_dates=True)
        df_d1 = pp.clean_data(df_d1)
        df = fe.add_higher_timeframe_features(df, df_d1, suffix="d1")
    
    # Target
    pip_size = settings.PIP_SIZE.get(symbol, 0.0001)
    df["target"] = fe.create_target(df, pip_size=pip_size)
    
    # Preparar features
    trainer = ModelTrainer(model_type=model_type)
    X, y = trainer.prepare_features(df)
    
    # Split
    split_idx = int(len(X) * 0.7)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    return X_train, y_train, X_test, y_test, trainer, list(X.columns)


def analyze_model_importance(trainer: ModelTrainer, top_n: int = 20) -> pd.DataFrame:
    """
    Analisis 1: Feature importance del modelo (gain-based).
    """
    print(f"\n{'='*60}")
    print(f"  1. FEATURE IMPORTANCE (Model-based)")
    print(f"{'='*60}")
    
    fi = trainer.get_feature_importance()
    fi["importance_pct"] = fi["importance"] / fi["importance"].sum() * 100
    fi["cumulative_pct"] = fi["importance_pct"].cumsum()
    
    print(f"\n  Top {top_n} features:")
    print(f"  {'Feature':<35} {'Importance':>10} {'%':>7} {'Cumul%':>8}")
    print(f"  {'-'*62}")
    for _, row in fi.head(top_n).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:>10.4f} {row['importance_pct']:>6.1f}% {row['cumulative_pct']:>7.1f}%")
    
    # Features con importancia < 1%
    low_importance = fi[fi["importance_pct"] < 1.0]
    print(f"\n  Features con importancia < 1%: {len(low_importance)} de {len(fi)}")
    print(f"  Estas features son candidatas a ELIMINAR:")
    for _, row in low_importance.iterrows():
        print(f"    - {row['feature']} ({row['importance_pct']:.2f}%)")
    
    # Cuantas features acumulan el 90% de importancia
    top_90 = fi[fi["cumulative_pct"] <= 90]
    print(f"\n  Features que acumulan 90% de importancia: {len(top_90)} de {len(fi)}")
    
    return fi


def analyze_shap_values(
    trainer: ModelTrainer,
    X_test: pd.DataFrame,
    top_n: int = 20,
    save_plots: bool = False,
    output_dir: Path = None,
):
    """
    Analisis 2: SHAP values para entender el impacto real de cada feature.
    """
    print(f"\n{'='*60}")
    print(f"  2. SHAP VALUES")
    print(f"{'='*60}")
    
    try:
        import shap
    except ImportError:
        print("\n  SHAP no esta instalado. Instalar con: pip install shap")
        print("  Saltando analisis SHAP...")
        return None
    
    # Crear explainer
    print("  Calculando SHAP values (puede tardar 1-2 minutos)...")
    
    # Usar un subset para que sea mas rapido
    n_samples = min(500, len(X_test))
    X_sample = X_test.sample(n=n_samples, random_state=42)
    
    explainer = shap.TreeExplainer(trainer.model)
    shap_values = explainer.shap_values(X_sample)
    
    # Para binario, shap_values puede ser una lista o array
    if isinstance(shap_values, list):
        # Para clasificacion binaria, usar la clase positiva (UP)
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values
    
    # Mean absolute SHAP por feature
    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    
    shap_df["shap_pct"] = shap_df["mean_abs_shap"] / shap_df["mean_abs_shap"].sum() * 100
    
    print(f"\n  Top {top_n} features por SHAP:")
    print(f"  {'Feature':<35} {'Mean|SHAP|':>12} {'%':>7}")
    print(f"  {'-'*56}")
    for _, row in shap_df.head(top_n).iterrows():
        print(f"  {row['feature']:<35} {row['mean_abs_shap']:>12.6f} {row['shap_pct']:>6.1f}%")
    
    # Features con SHAP insignificante (< 0.5% del total)
    low_shap = shap_df[shap_df["shap_pct"] < 0.5]
    print(f"\n  Features con SHAP < 0.5%: {len(low_shap)} (candidatas a eliminar)")
    
    # Guardar plots si se solicita
    if save_plots and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(sv, X_sample, max_display=top_n, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Plot guardado: {output_dir / 'shap_summary.png'}")
            
            # Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(sv, X_sample, plot_type="bar", max_display=top_n, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Plot guardado: {output_dir / 'shap_bar.png'}")
            
        except Exception as e:
            print(f"  Error guardando plots: {e}")
    
    return shap_df


def analyze_correlations(X: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """
    Analisis 3: Correlacion entre features (multicolinealidad).
    Features altamente correlacionadas son redundantes.
    """
    print(f"\n{'='*60}")
    print(f"  3. CORRELACION ENTRE FEATURES (multicolinealidad)")
    print(f"{'='*60}")
    
    corr_matrix = X.corr().abs()
    
    # Encontrar pares con correlacion > threshold
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > threshold:
                high_corr.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": corr_val,
                })
    
    high_corr_df = pd.DataFrame(high_corr).sort_values("correlation", ascending=False)
    
    print(f"\n  Pares con correlacion > {threshold}:")
    print(f"  {'Feature 1':<30} {'Feature 2':<30} {'Corr':>6}")
    print(f"  {'-'*68}")
    
    if high_corr_df.empty:
        print(f"  Ninguno encontrado.")
    else:
        for _, row in high_corr_df.iterrows():
            print(f"  {row['feature_1']:<30} {row['feature_2']:<30} {row['correlation']:>5.3f}")
    
    # Identificar features a eliminar (la segunda de cada par)
    features_to_remove = set()
    for _, row in high_corr_df.iterrows():
        # Eliminar la que aparece mas veces en pares correlacionados
        features_to_remove.add(row["feature_2"])
    
    print(f"\n  Features candidatas a eliminar por multicolinealidad ({len(features_to_remove)}):")
    for f in sorted(features_to_remove):
        print(f"    - {f}")
    
    return high_corr_df


def analyze_feature_target_correlation(X: pd.DataFrame, y: pd.Series, min_corr: float = 0.02) -> pd.DataFrame:
    """
    Analisis 4: Correlacion feature-target.
    Features con muy baja correlacion con el target probablemente son ruido.
    """
    print(f"\n{'='*60}")
    print(f"  4. CORRELACION FEATURE -> TARGET")
    print(f"{'='*60}")
    
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    corr_df = pd.DataFrame({
        "feature": correlations.index,
        "abs_correlation": correlations.values,
    })
    
    print(f"\n  Top 15 features mas correlacionadas con target:")
    print(f"  {'Feature':<35} {'|Corr|':>8}")
    print(f"  {'-'*45}")
    for _, row in corr_df.head(15).iterrows():
        print(f"  {row['feature']:<35} {row['abs_correlation']:>7.4f}")
    
    # Features con correlacion casi nula
    low_corr = corr_df[corr_df["abs_correlation"] < min_corr]
    print(f"\n  Features con |corr| < {min_corr} (probablemente ruido): {len(low_corr)}")
    for _, row in low_corr.iterrows():
        print(f"    - {row['feature']} (corr={row['abs_correlation']:.4f})")
    
    return corr_df


def generate_recommendations(
    fi: pd.DataFrame,
    shap_df: pd.DataFrame,
    high_corr: pd.DataFrame,
    target_corr: pd.DataFrame,
    feature_names: list,
) -> dict:
    """
    Generar recomendaciones consolidadas de features a eliminar y conservar.
    """
    print(f"\n{'='*60}")
    print(f"  5. RECOMENDACIONES")
    print(f"{'='*60}")
    
    # Features con baja importancia en el modelo
    low_fi = set(fi[fi["importance_pct"] < 1.0]["feature"].tolist())
    
    # Features con bajo SHAP (si disponible)
    low_shap = set()
    if shap_df is not None:
        low_shap = set(shap_df[shap_df["shap_pct"] < 0.5]["feature"].tolist())
    
    # Features redundantes por correlacion
    redundant = set()
    if not high_corr.empty:
        redundant = set(high_corr["feature_2"].tolist())
    
    # Features sin correlacion con target
    low_target = set(target_corr[target_corr["abs_correlation"] < 0.02]["feature"].tolist())
    
    # Consolidar: features que aparecen en 2+ listas negras
    all_features = set(feature_names)
    removal_scores = {}
    
    for f in all_features:
        score = 0
        reasons = []
        if f in low_fi:
            score += 1
            reasons.append("baja importancia")
        if f in low_shap:
            score += 1
            reasons.append("bajo SHAP")
        if f in redundant:
            score += 1
            reasons.append("redundante")
        if f in low_target:
            score += 1
            reasons.append("baja corr target")
        removal_scores[f] = {"score": score, "reasons": reasons}
    
    # Ordenar por score
    sorted_features = sorted(removal_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    # Features a ELIMINAR (score >= 2)
    to_remove = [f for f, s in sorted_features if s["score"] >= 2]
    to_keep = [f for f, s in sorted_features if s["score"] < 2]
    
    print(f"\n  FEATURES A ELIMINAR ({len(to_remove)}):")
    for f, s in sorted_features:
        if s["score"] >= 2:
            print(f"    ELIMINAR: {f:<35} (score={s['score']}, {', '.join(s['reasons'])})")
    
    print(f"\n  FEATURES A CONSERVAR ({len(to_keep)}):")
    for f, s in sorted_features:
        if s["score"] < 2:
            reasons = ", ".join(s["reasons"]) if s["reasons"] else "OK"
            print(f"    CONSERVAR: {f:<35} ({reasons})")
    
    print(f"\n  Resumen: {len(to_remove)} a eliminar, {len(to_keep)} a conservar de {len(all_features)} total")
    print(f"  Reduccion: {len(to_remove)/len(all_features)*100:.0f}% de features eliminadas")
    
    return {
        "to_remove": to_remove,
        "to_keep": to_keep,
        "removal_scores": removal_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Analisis de features con SHAP")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Par de divisas")
    parser.add_argument("--model", type=str, default=None, help="Tipo de modelo ML")
    parser.add_argument("--top", type=int, default=20, help="Numero de features a mostrar")
    parser.add_argument("--save-plots", action="store_true", help="Guardar plots SHAP")
    parser.add_argument("--verbose", action="store_true", help="Logging detallado")
    args = parser.parse_args()
    
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format=settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)
    
    print(f"\n{'='*60}")
    print(f"  ANALISIS DE FEATURES - {args.symbol}")
    print(f"{'='*60}")
    
    # Cargar y preparar datos
    print("\nCargando y preparando datos...")
    X_train, y_train, X_test, y_test, trainer, feature_names = load_prepared_data(
        args.symbol, args.model,
    )
    print(f"  Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")
    print(f"  Features: {len(feature_names)}")
    
    # Entrenar modelo
    print("Entrenando modelo para analisis...")
    val_split = int(len(X_train) * 0.85)
    X_tr = X_train.iloc[:val_split]
    y_tr = y_train.iloc[:val_split]
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]
    trainer.train(X_tr, y_tr, X_val, y_val)
    
    # 1. Feature importance
    fi = analyze_model_importance(trainer, top_n=args.top)
    
    # 2. SHAP values
    output_dir = settings.BASE_DIR / "analysis"
    shap_df = analyze_shap_values(
        trainer, X_test, top_n=args.top,
        save_plots=args.save_plots, output_dir=output_dir,
    )
    
    # 3. Correlaciones entre features
    high_corr = analyze_correlations(X_train)
    
    # 4. Correlacion feature-target
    target_corr = analyze_feature_target_correlation(X_train, y_train)
    
    # 5. Recomendaciones
    recommendations = generate_recommendations(
        fi, shap_df, high_corr, target_corr, feature_names,
    )
    
    # Guardar recomendaciones a archivo
    output_dir.mkdir(parents=True, exist_ok=True)
    rec_path = output_dir / "feature_recommendations.txt"
    with open(rec_path, "w") as f:
        f.write(f"Feature Analysis - {args.symbol}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total features: {len(feature_names)}\n")
        f.write(f"Features a eliminar: {len(recommendations['to_remove'])}\n")
        f.write(f"Features a conservar: {len(recommendations['to_keep'])}\n\n")
        
        f.write("ELIMINAR:\n")
        for feat in recommendations["to_remove"]:
            s = recommendations["removal_scores"][feat]
            f.write(f"  - {feat} (score={s['score']}, {', '.join(s['reasons'])})\n")
        
        f.write("\nCONSERVAR:\n")
        for feat in recommendations["to_keep"]:
            f.write(f"  - {feat}\n")
    
    print(f"\n  Recomendaciones guardadas en: {rec_path}")
    print(f"\nAnalisis completado.")


if __name__ == "__main__":
    main()

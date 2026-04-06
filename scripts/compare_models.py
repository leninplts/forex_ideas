"""
Comparar modelos con diferentes n_estimators (300, 500, 1000).
Incluye metricas extendidas: AUC, Precision, Recall, Brier Score.
Ejecutar: python scripts/compare_models.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)

from forex_bot.data.preprocessor import DataPreprocessor
from forex_bot.data.feature_engine import FeatureEngine
from forex_bot.config import settings


def prepare_data():
    """Cargar y preparar datos con features + multi-timeframe."""
    pp = DataPreprocessor()
    fe = FeatureEngine()

    # Cargar H1, H4, D1
    df = pp.clean_data(pd.read_csv("data_cache/EURUSDm_H1.csv", index_col="time", parse_dates=True))
    df_h4 = pp.clean_data(pd.read_csv("data_cache/EURUSDm_H4.csv", index_col="time", parse_dates=True))
    df_d1 = pp.clean_data(pd.read_csv("data_cache/EURUSDm_D1.csv", index_col="time", parse_dates=True))

    # Features H1
    df = fe.add_all_features(df)

    # Multi-timeframe
    df = fe.add_higher_timeframe_features(df, df_h4, suffix="h4")
    df = fe.add_higher_timeframe_features(df, df_d1, suffix="d1")

    # Target binario
    df["target"] = fe.create_target(df, horizon=3, min_pips=0, pip_size=0.0001)

    # Preparar X, y
    exclude = {"open", "high", "low", "close", "tick_volume", "volume", "spread", "target"}
    feat_cols = [c for c in df.columns if c not in exclude]
    X = df[feat_cols]
    y = df["target"]
    valid_mask = y.notna() & (~X.isna().any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask].astype(int)

    # Split: 70% train+val, 30% test
    split = int(len(X) * 0.7)
    val_split = int(split * 0.85)

    return {
        "X_train": X.iloc[:val_split],
        "y_train": y.iloc[:val_split],
        "X_val": X.iloc[val_split:split],
        "y_val": y.iloc[val_split:split],
        "X_test": X.iloc[split:],
        "y_test": y.iloc[split:],
        "feat_cols": feat_cols,
    }


def train_and_evaluate(data, n_estimators, max_depth=4):
    """Entrenar modelo y calcular TODAS las metricas."""
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        verbosity=0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        data["X_train"], data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        verbose=False,
    )

    # Predicciones
    y_pred = model.predict(data["X_test"])
    y_proba = model.predict_proba(data["X_test"])[:, 1]  # Probabilidad de UP

    y_test = data["y_test"]

    # --- METRICAS ---
    metrics = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "best_iteration": model.best_iteration,

        # Clasificacion
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),

        # Discriminacion
        "auc": roc_auc_score(y_test, y_proba),

        # Calibracion de probabilidades
        "brier": brier_score_loss(y_test, y_proba),

        # Por clase
        "precision_down": precision_score(y_test, y_pred, pos_label=0),
        "recall_down": recall_score(y_test, y_pred, pos_label=0),
        "precision_up": precision_score(y_test, y_pred, pos_label=1),
        "recall_up": recall_score(y_test, y_pred, pos_label=1),
    }

    # Trading sim con threshold 0.52
    threshold = settings.CONFIDENCE_THRESHOLD
    buy_mask = y_proba > threshold
    sell_mask = y_proba < (1 - threshold)
    n_buy = buy_mask.sum()
    n_sell = sell_mask.sum()
    n_trades = n_buy + n_sell
    actual = y_test.values
    correct = (actual[buy_mask] == 1).sum() + (actual[sell_mask] == 0).sum()
    trade_acc = correct / max(n_trades, 1)

    metrics["n_trades_sim"] = n_trades
    metrics["trade_accuracy"] = trade_acc

    # Feature importance top 5
    fi = pd.DataFrame({"feature": data["feat_cols"], "importance": model.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    metrics["top_features"] = fi.head(5)["feature"].tolist()

    return metrics, model


def print_comparison(results):
    """Imprimir tabla comparativa."""
    print("\n" + "=" * 90)
    print("  COMPARACION DE MODELOS - XGBoost con diferentes n_estimators")
    print("=" * 90)

    # Header
    configs = [f"n={r['n_estimators']}" for r in results]
    header = f"{'Metrica':<25}"
    for c in configs:
        header += f" {c:>15}"
    header += f" {'Mejor':>10}"
    print(header)
    print("-" * 90)

    # Metricas a comparar
    rows = [
        ("best_iteration", "Best Iteration", "max"),
        ("accuracy", "Accuracy", "max"),
        ("f1", "F1 Score", "max"),
        ("precision", "Precision (weighted)", "max"),
        ("recall", "Recall (weighted)", "max"),
        ("auc", "AUC-ROC", "max"),
        ("brier", "Brier Score", "min"),
        ("precision_down", "Precision DOWN", "max"),
        ("recall_down", "Recall DOWN", "max"),
        ("precision_up", "Precision UP", "max"),
        ("recall_up", "Recall UP", "max"),
        ("n_trades_sim", "Trades (sim)", "max"),
        ("trade_accuracy", "Trade Accuracy", "max"),
    ]

    for key, label, direction in rows:
        values = [r[key] for r in results]
        line = f"  {label:<23}"

        if key in ("best_iteration", "n_trades_sim"):
            for v in values:
                line += f" {v:>15d}"
        else:
            for v in values:
                line += f" {v:>14.4f}"

        # Marcar el mejor
        if direction == "max":
            best_idx = values.index(max(values))
        else:
            best_idx = values.index(min(values))

        best_label = configs[best_idx]
        line += f" {best_label:>10}"

        print(line)

    # Top features
    print()
    print("  Top 5 Features por modelo:")
    for r in results:
        print(f"    n={r['n_estimators']}: {', '.join(r['top_features'])}")

    print("=" * 90)


def main():
    print("Preparando datos...")
    data = prepare_data()
    print(f"  Train: {len(data['X_train'])} | Val: {len(data['X_val'])} | Test: {len(data['X_test'])}")
    print(f"  Features: {len(data['feat_cols'])}")
    print()

    results = []
    models = {}

    for n_est in [300, 500, 1000]:
        print(f"Entrenando modelo con n_estimators={n_est}...")
        metrics, model = train_and_evaluate(data, n_est)
        results.append(metrics)
        models[n_est] = model
        print(f"  Best iteration: {metrics['best_iteration']} | Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['auc']:.4f}")

    # Comparacion
    print_comparison(results)

    # Guardar el mejor modelo
    best = max(results, key=lambda r: r["auc"])
    best_n = best["n_estimators"]
    best_model = models[best_n]

    save_path = Path("forex_bot/models/saved") / f"xgboost_best_n{best_n}"
    best_model.save_model(str(save_path.with_suffix(".json")))

    # Guardar metadata
    import json
    from datetime import datetime
    meta = {
        "model_type": "xgboost",
        "train_date": datetime.now().isoformat(),
        "n_estimators": best_n,
        "max_depth": best["max_depth"],
        "best_iteration": best["best_iteration"],
        "accuracy": best["accuracy"],
        "auc": best["auc"],
        "brier": best["brier"],
        "f1": best["f1"],
        "n_features": len(data["feat_cols"]),
        "feature_names": data["feat_cols"],
    }
    with open(save_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nMejor modelo (n={best_n}) guardado en: {save_path.with_suffix('.json')}")
    print(f"  AUC: {best['auc']:.4f} | Accuracy: {best['accuracy']:.4f} | Brier: {best['brier']:.4f}")


if __name__ == "__main__":
    main()

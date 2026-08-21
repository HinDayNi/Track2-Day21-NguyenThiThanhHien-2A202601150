import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# MLflow local tracking
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("income-model")

F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATE = 0.248
DRIFT_THRESHOLD = 0.05


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        mlflow.log_params(params)

        model = GradientBoostingClassifier(
            **params,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # =====================================================
        # METRIC GOC - threshold 0.5
        # =====================================================
        probabilities = model.predict_proba(X_eval)[:, 1]
        preds = (probabilities >= 0.5).astype(int)

        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)

        # =====================================================
        # BONUS 2 - TIM THRESHOLD TOT NHAT
        # =====================================================
        best_threshold = 0.5
        best_f1 = f1

        threshold_results = []

        for threshold in np.arange(0.1, 0.91, 0.05):
            threshold = round(float(threshold), 2)

            threshold_preds = (
                probabilities >= threshold
            ).astype(int)

            threshold_f1 = f1_score(
                y_eval,
                threshold_preds,
            )

            threshold_results.append(
                {
                    "threshold": threshold,
                    "f1": float(threshold_f1),
                }
            )

            if threshold_f1 > best_f1:
                best_f1 = threshold_f1
                best_threshold = threshold

        # =====================================================
        # BONUS 3 - PRECISION / RECALL / CONFUSION MATRIX
        # =====================================================
        precision = precision_score(
            y_eval,
            preds,
            zero_division=0,
        )

        recall = recall_score(
            y_eval,
            preds,
            zero_division=0,
        )

        cm = confusion_matrix(y_eval, preds)

        tn, fp, fn, tp = cm.ravel()

        # =====================================================
        # BONUS 5 - DATA DRIFT
        # =====================================================
        positive_rate = float(y_train.mean())

        drift_amount = abs(
            positive_rate - REFERENCE_POSITIVE_RATE
        )

        drift_warning = (
            drift_amount > DRIFT_THRESHOLD
        )

        if drift_warning:
            print(
                "WARNING DATA DRIFT: "
                f"positive_rate={positive_rate:.2%}, "
                f"reference={REFERENCE_POSITIVE_RATE:.2%}"
            )
        else:
            print(
                "Data distribution OK: "
                f"positive_rate={positive_rate:.2%}, "
                f"reference={REFERENCE_POSITIVE_RATE:.2%}"
            )

        # =====================================================
        # MLFLOW
        # =====================================================
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        mlflow.log_metric(
            "precision_positive",
            precision,
        )
        mlflow.log_metric(
            "recall_positive",
            recall,
        )

        mlflow.log_metric(
            "best_threshold",
            best_threshold,
        )
        mlflow.log_metric(
            "best_threshold_f1",
            best_f1,
        )

        mlflow.log_metric(
            "positive_rate",
            positive_rate,
        )

        mlflow.sklearn.log_model(
            model,
            "model",
        )

        # =====================================================
        # OUTPUT
        # =====================================================
        print(
            f"F1: {f1:.4f} | "
            f"Accuracy: {acc:.4f}"
        )

        print(
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f}"
        )

        print(
            f"Best threshold: "
            f"{best_threshold:.2f} | "
            f"Best F1: {best_f1:.4f}"
        )

        print(
            "Confusion matrix: "
            f"TN={tn}, FP={fp}, "
            f"FN={fn}, TP={tp}"
        )

        os.makedirs(
            "outputs",
            exist_ok=True,
        )

        report = {
            "f1_score": float(f1),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "best_threshold": float(best_threshold),
            "best_threshold_f1": float(best_f1),
            "positive_rate": positive_rate,
            "reference_positive_rate": (
                REFERENCE_POSITIVE_RATE
            ),
            "data_drift_warning": drift_warning,
        }

        with open(
            "outputs/report.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
            )

        # BONUS 2 artifact
        with open(
            "outputs/thresholds.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                threshold_results,
                f,
                indent=2,
            )

        # BONUS 3 artifact
        with open(
            "outputs/detail.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                f"""Precision/Recall Report

F1: {f1:.4f}
Accuracy: {acc:.4f}
Precision positive: {precision:.4f}
Recall positive: {recall:.4f}

Confusion Matrix
TN={tn}
FP={fp}
FN={fn}
TP={tp}

Best threshold: {best_threshold:.2f}
Best threshold F1: {best_f1:.4f}

Positive rate: {positive_rate:.4f}
Reference positive rate: {REFERENCE_POSITIVE_RATE:.4f}
Data drift warning: {drift_warning}
"""
            )

        os.makedirs(
            "models",
            exist_ok=True,
        )

        joblib.dump(
            model,
            "models/model.joblib",
        )

    return f1


if __name__ == "__main__":

    with open(
        "params.yaml",
        encoding="utf-8",
    ) as f:
        params = yaml.safe_load(f)

    train(params)
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import balanced_accuracy_score, roc_auc_score



def classification_metrics(predictions: pl.DataFrame) -> dict[str, float]:
    if predictions.height == 0:
        return {"auc": 0.0, "balanced_accuracy": 0.0, "accuracy": 0.0}

    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    p_up = predictions["p_up"].to_numpy()

    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, p_up))

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float((y_true == y_pred).mean())
    return {"auc": auc, "balanced_accuracy": bal_acc, "accuracy": acc}


def merge_metrics(model_name: str, cls_metrics: dict[str, float], bt_metrics: dict[str, float]) -> dict[str, Any]:
    return {"model_name": model_name, **cls_metrics, **bt_metrics}

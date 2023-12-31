import os
import pickle
import numpy as np
import pandas as pd
from dagster import asset, AssetIn
from dagster_fd.assets.splitting import (
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    ord_train_val
)
from dagster_fd.assets.tuning import tuning_result_baseline_model
from dagster_fd.assets.code.metrics import calc_metrics
from constants import MODELS_TRAINING_FD
from p2_ml.model_src.baseline_models import RS_baseline_usr_mov


@asset(
    ins={
        "X_train": AssetIn(),
        "X_val": AssetIn(),
        "X_test": AssetIn(),
        "y_train": AssetIn(),
        "y_val": AssetIn(),
        "y_test": AssetIn(),
        "ord_train_val": AssetIn(),
        "tuning_result_baseline_model": AssetIn()
    },
    group="training"
)
def trained_baseline_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ord_train_val: np.ndarray,
    tuning_result_baseline_model: float
) -> RS_baseline_usr_mov:
    X_train_val = pd.concat((X_train, X_val), ignore_index=True)
    X_train_val = X_train_val.iloc[ord_train_val]

    y_train_val = np.hstack((y_train, y_val))
    assert y_train_val.shape[0] == y_train.shape[0] + y_val.shape[0]

    p = tuning_result_baseline_model
    model = RS_baseline_usr_mov(p=p)
    model.fit(X_train_val, y_train_val)

    y_pred = model.predict(X_test)
    md = calc_metrics(y_test, y_pred)

    path = os.path.join(MODELS_TRAINING_FD, f"baseline__{int(p * 100)}pp.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    return model

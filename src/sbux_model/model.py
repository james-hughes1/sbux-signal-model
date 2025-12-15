import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def walk_forward_eval(X, y, pipeline, train_window, horizon, expanding=False):
    """
    Perform walk-forward (rolling or expanding window) evaluation for a regression model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features), indexed by time.
    y : pd.Series
        Target vector with shape (n_samples,), indexed by time.
    pipeline : sklearn.pipeline.Pipeline
        A scikit-learn pipeline containing preprocessing and regression model.
    train_window : int
        Number of observations to use for training in each window.
    horizon : int
        Number of observations to predict in each out-of-sample window.
    expanding : bool, default=False
        If True, the training window expands with each iteration; otherwise, a fixed-size rolling window is used.

    Returns
    -------
    preds : pd.Series
        Out-of-sample predictions, indexed by time.
    truths : pd.Series
        Corresponding true target values for the out-of-sample periods, indexed by time.
    metrics : dict
        Dictionary of aggregated evaluation metrics:
            - "r2_oos": Out-of-sample R-squared
            - "rmse_oos": Out-of-sample root mean squared error
            - "n_oos": Number of out-of-sample observations

    Notes
    -----
    - The function iteratively trains the model on the specified training window and predicts the next horizon of observations.
    - Initial training period predictions are NaN and excluded from metrics.
    - Assumes that X and y are aligned and indexed by time, typically in chronological order.
    - Useful for time series regression where standard cross-validation would introduce lookahead bias.
    """
    n = len(X)
    preds = pd.Series(index=X.index, dtype=float)
    truths = pd.Series(index=X.index, dtype=float)

    start = train_window
    while start + horizon <= n:

        # Train indices
        if expanding:
            tr_idx = range(0, start)
        else:
            tr_idx = range(start - train_window, start)

        te_idx = range(start, start + horizon)

        X_tr = X.iloc[list(tr_idx)]
        y_tr = y.iloc[list(tr_idx)]
        X_te = X.iloc[list(te_idx)]
        y_te = y.iloc[list(te_idx)]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)

        preds.iloc[list(te_idx)] = y_pred
        truths.iloc[list(te_idx)] = y_te.values

        start += horizon

    # Clean NA (initial training period)
    valid = truths.dropna().index
    preds = preds.loc[valid]
    truths = truths.loc[valid]

    mse = mean_squared_error(truths, preds)
    r2 = 1 - mse / np.var(truths, ddof=0)
    rmse = np.sqrt(mse)

    metrics = {
        "r2_oos": float(r2),
        "rmse_oos": float(rmse),
        "n_oos": int(len(valid))
    }

    return preds, truths, metrics


def zero_predictor_baseline(y, oos_index):
    """
    Zero predictor baseline for alpha.
    Assumes ŷ = 0 for all OOS observations.
    """
    y_oos = y.loc[oos_index]
    preds = np.zeros(len(y_oos))

    mse = np.mean((y_oos.values - preds) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - mse / np.var(y_oos.values, ddof=0)

    return {
        "r2_oos": float(r2),
        "rmse_oos": float(rmse),
        "n_oos": int(len(y_oos)),
    }

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def walk_forward_eval(X, y, pipeline, train_window, horizon, expanding=False):
    """
    Walk-forward / rolling window evaluation.
    Fits model on each window, predicts next horizon, rolls forward.
    Returns OOS predictions, truth, and metrics.
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
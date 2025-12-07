import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sbux_model.io import read_table, save_table
from sbux_model.model import walk_forward_eval

CONFIG_PATH = "src/config/train_config.json"

# ===============================================================
# Load configuration
# ===============================================================
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

stage_name = config["stage_name"]
input_stage = config["input_stage"]

# Walk-forward parameters
wf_cfg = config.get("test", {})
train_window = wf_cfg.get("train_window", 156)
horizon = wf_cfg.get("horizon", 4)
expanding = wf_cfg.get("expanding", False)

# ===============================================================
# Load feature table
# ===============================================================
df = read_table(stage_name=input_stage, config=config.get("input"))

target_col = config["target"]
feature_cols = config["feature_columns"]

X = df[feature_cols].copy()
y = df[target_col].copy()

# ===============================================================
# Print most correlated feature pairs
# ===============================================================
def print_top_correlated_features(df, top_n=10):
    corr_matrix = df.corr().abs()  # absolute correlations
    # Mask lower triangle
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_matrix_triu = corr_matrix.where(mask)
    # Stack and sort
    sorted_pairs = corr_matrix_triu.stack().sort_values(ascending=False)
    print(f"\nTop {top_n} most correlated feature pairs:\n")
    print(sorted_pairs.head(top_n))

print_top_correlated_features(X, top_n=10)

# ===============================================================
# Model selection
# ===============================================================
model_cfg = config.get("model", {"type": "ridge"})
model_type = model_cfg.get("type", "ridge").lower()

if model_type == "linear":
    base_model = LinearRegression(fit_intercept=model_cfg.get("fit_intercept", True))
elif model_type == "ridge":
    base_model = Ridge(alpha=model_cfg.get("alpha", 1.0),
                       fit_intercept=model_cfg.get("fit_intercept", True))
elif model_type == "lasso":
    base_model = Lasso(alpha=model_cfg.get("alpha", 1.0),
                       fit_intercept=model_cfg.get("fit_intercept", True))
else:
    raise ValueError(f"Unknown model type: {model_type}")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", base_model)
])

# ===============================================================
# Walk-Forward Evaluation
# ===============================================================
print("Running walk-forward evaluation...\n")
_, truths_oos, oos_metrics = walk_forward_eval(
    X, y, pipeline,
    train_window=train_window,
    horizon=horizon,
    expanding=expanding
)

print("OOS Metrics:")
for k, v in oos_metrics.items():
    print(f"{k}: {v}")

# Save predictions of full data (train & oos) into df
df["pred_" + target_col] = pipeline.predict(X)

# ===============================================================
# Fit final model on full dataset
# ===============================================================
pipeline.fit(X, y)

# ===============================================================
# Save predictions
# ===============================================================
pred_df = df[[target_col, "pred_" + target_col] + feature_cols]
pred_output_path = save_table(pred_df, stage_name, config.get("output_predictions"))
print(f"\nSaved predictions → {pred_output_path}")

# ===============================================================
# Save trained model
# ===============================================================
model_dir = f"data/{stage_name}"
os.makedirs(model_dir, exist_ok=True)

if config.get("output_model") and config["output_model"].get("filename"):
    model_path = os.path.join(model_dir, config["output_model"]["filename"])
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{stage_name}_{timestamp}.pkl")

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print(f"Saved model → {model_path}")

# ===============================================================
# Determine OOS cutoff date
# ===============================================================
if expanding:
    oos_start_idx = train_window
else:
    oos_start_idx = len(df) - len(truths_oos)  # first OOS row
oos_cutoff_date = df.index[oos_start_idx - 1] if hasattr(df.index, "__getitem__") else df.iloc[oos_start_idx - 1]["Date"]

# ===============================================================
# Save metrics with extra info
# ===============================================================
metrics = {
    "walk_forward": oos_metrics,
    "model_type": model_type,
    "model_params": model_cfg,
    "n_rows": len(df),
    "train_window": train_window,
    "horizon": horizon,
    "features_used": feature_cols,
    "target_col": target_col,
    "predicted_col": "pred_" + target_col,
    "oos_cutoff_date": str(oos_cutoff_date)
}

metrics_path = model_path.replace(".pkl", "_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Saved metrics → {metrics_path}")

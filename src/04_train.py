import os
import json
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sbux_model.io import read_table, save_table

CONFIG_PATH = "src/config/train_config.json"

# Load config
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

stage_name = config["stage_name"]
input_stage = config["input_stage"]

# --- Load feature table ---
df = read_table(stage_name=input_stage, config=config.get("input"))

# --- Target and feature setup ---
target_col = config["target"]
feature_cols = config["feature_columns"]

X = df[feature_cols]
y = df[target_col]

# --- Train-test split ---
test_size = config["test"]["size"]
test_type = config["test"]["type"]

if test_type == "last_n":
    n = test_size
    X_train, X_test = X.iloc[:-n], X.iloc[-n:]
    y_train, y_test = y.iloc[:-n], y.iloc[-n:]
elif test_type == "fraction":
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
else:
    raise ValueError(f"Unknown test split type: {test_type}")

# --- Model setup ---
model_cfg = config.get("model", {"type": "linear"})
model_type = model_cfg.get("type", "linear").lower()

if model_type == "linear":
    base_model = LinearRegression(fit_intercept=model_cfg.get("fit_intercept", True))
elif model_type == "ridge":
    base_model = Ridge(alpha=model_cfg.get("alpha", 1.0), fit_intercept=model_cfg.get("fit_intercept", True))
elif model_type == "lasso":
    base_model = Lasso(alpha=model_cfg.get("alpha", 1.0), fit_intercept=model_cfg.get("fit_intercept", True))
else:
    raise ValueError(f"Unknown model type: {model_type}")

# --- Create pipeline with scaler ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", base_model)
])

# --- Train ---
pipeline.fit(X_train, y_train)

# --- Predictions ---
df["pred_" + target_col] = pipeline.predict(X)

# --- Metrics ---
metrics = {
    "r2_train": r2_score(y_train, pipeline.predict(X_train)),
    "r2_test": r2_score(y_test, pipeline.predict(X_test)),
    "rmse_test": root_mean_squared_error(y_test, pipeline.predict(X_test)),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "features_used": feature_cols,
    "model_type": model_type,
    "model_params": model_cfg
}

print("\nMODEL METRICS")
for k, v in metrics.items():
    print(f"{k}: {v}")

# --- Save predictions CSV ---
pred_df = df[[target_col, "pred_" + target_col] + feature_cols]
pred_output_path = save_table(pred_df, stage_name, config.get("output_predictions"))
print(f"Saved predictions → {pred_output_path}")

# --- Save model ---
model_dir = f"data/{stage_name}"
os.makedirs(model_dir, exist_ok=True)
if config.get("output_model") and config["output_model"].get("filename"):
    model_path = os.path.join(model_dir, config["output_model"]["filename"])
else:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{stage_name}_{timestamp}.pkl")

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print(f"Saved model → {model_path}")

# --- Save metrics JSON ---
metrics_path = model_path.replace(".pkl", "_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Saved metrics → {metrics_path}")

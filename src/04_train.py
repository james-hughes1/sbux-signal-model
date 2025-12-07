import os
import json
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

from sbux_model.io import read_table, save_table

CONFIG_PATH = "src/config/train_config.json"

# Load config
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

stage_name = config["stage_name"]            # e.g. "model"
input_stage = config["input_stage"]          # e.g. "features"

# --- Load feature table ---
df = read_table(stage_name=input_stage, config=config.get("input"))

# --- Target and feature setup ---
target_col = config["target"]                # e.g. "excess_ret_fwd_1"
feature_cols = config["feature_columns"]     # list of feature column names

X = df[feature_cols]
y = df[target_col]

# --- Train-test split ---
test_size = config["test"]["size"]
test_type = config["test"]["type"]  # "last_n" or "fraction"

if test_type == "last_n":
    n = test_size
    X_train, X_test = X.iloc[:-n], X.iloc[-n:]
    y_train, y_test = y.iloc[:-n], y.iloc[-n:]
elif test_type == "fraction":
    frac = test_size
    split_idx = int(len(df) * (1 - frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
else:
    raise ValueError("Unknown test split type")

# --- Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Predictions ---
df["pred_" + target_col] = model.predict(X)

# --- Metrics ---
metrics = {
    "r2_train": r2_score(y_train, model.predict(X_train)),
    "r2_test": r2_score(y_test, model.predict(X_test)),
    "rmse_test": root_mean_squared_error(y_test, model.predict(X_test)),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "features_used": feature_cols
}

print("\nMODEL METRICS")
for k, v in metrics.items():
    print(f"{k}: {v}")

# --- Save predictions CSV ---
# Only keep target, predicted target, and features used
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
    pickle.dump(model, f)

print(f"Saved model → {model_path}")

# --- Save metrics json ---
metrics_path = model_path.replace(".pkl", "_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Saved metrics → {metrics_path}")

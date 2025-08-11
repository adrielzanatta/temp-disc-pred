import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from src.tools import outlier_removal, prepare_df
import tomli

# ===== CONFIG =====

with open("config.toml", "rb") as toml_file:
    config_dict = tomli.load(toml_file)

print("Loading model for predictions...")

model: Pipeline = joblib.load(config_dict["MODEL_PATH"])

pred_data_path = input("Enter the path to the prediction data file: ")[1:-1]

df = pd.read_csv(
    pred_data_path, sep=";", decimal=",", parse_dates=["timestamp"], dayfirst=True
)
ts = df["timestamp"]
df = prepare_df(df, timestamp_col="timestamp")
df = outlier_removal(df, config_dict["NOT_FILTER_OUTLIERS"])

X = df.drop(columns=config_dict["TARGET_COLUMN"])

# ===== PREDICTION =====
predictions = model.predict(X)

with open(config_dict["VERSION_PATH"], "r") as f:
    version = f.read().strip()

# ===== SAVE PREDICTIONS =====
df.drop(columns=["timestamp_numeric"], inplace=True)
df["predictions"], df["version"], df["timestamp"] = predictions, version, ts

df["error"] = df["predictions"] - df[config_dict["TARGET_COLUMN"]]

df["abs_error"] = df["error"].abs()

print(df["abs_error"].describe())

df.to_csv("predictions_with_version.csv", index=False, decimal=",", sep=";")

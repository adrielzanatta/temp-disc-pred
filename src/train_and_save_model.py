import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
import joblib
import json
from datetime import datetime
from tools import outlier_removal, prepare_df
import tomli


# ===== CONFIG =====
with open("config.toml", "rb") as toml_file:
    config_dict = tomli.load(toml_file)


# ===== LOAD DATA =====
df = pd.read_csv(
    config_dict["TRAIN_DATA_PATH"],
    sep=";",
    decimal=",",
    parse_dates=["timestamp"],
    dayfirst=True,
)

df = prepare_df(df, timestamp_col="timestamp")

# ===== OUTLIER REMOVAL BEFORE CV =====
df = outlier_removal(df, config_dict["NOT_FILTER_OUTLIERS"])


# ===== FEATURE/TARGET SPLIT =====
y = df[config_dict["TARGET_COLUMN"]]
X = df.drop(columns=[config_dict["TARGET_COLUMN"]])

print(X.head())
# ===== DEFINE MODELS =====
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=config_dict["RANDOM_STATE"]),
    "Lasso": Lasso(random_state=config_dict["RANDOM_STATE"]),
    "DecisionTree": DecisionTreeRegressor(random_state=config_dict["RANDOM_STATE"]),
    "RandomForest": RandomForestRegressor(random_state=config_dict["RANDOM_STATE"]),
    "GradientBoosting": GradientBoostingRegressor(
        random_state=config_dict["RANDOM_STATE"]
    ),
    "SVR": SVR(),
}

# ===== CV EVALUATION =====
results = {}
for name, model in models.items():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                config_dict["CATEGORIAL_FEATURES"],
            ),
            (
                "num",
                SimpleImputer(strategy="mean"),
                make_column_selector(dtype_include=np.number),
            ),
        ],
        remainder="drop",
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )

    r2_scores = cross_val_score(pipe, X, y, cv=config_dict["N_SPLITS"], scoring="r2")
    results[name] = {
        "R2_mean": float(np.mean(r2_scores)),
        "R2_std": float(np.std(r2_scores)),
    }

# Save metrics
with open(config_dict["METRICS_PATH"], "w") as f:
    json.dump(results, f, indent=4)

# ===== SELECT BEST MODEL =====
best_model_name = max(results, key=lambda k: results[k]["R2_mean"])
best_model = models[best_model_name]

# ===== FINAL PIPELINE =====
final_preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            config_dict["CATEGORIAL_FEATURES"],
        ),
        (
            "num",
            SimpleImputer(strategy="mean"),
            make_column_selector(dtype_include=np.number),
        ),
    ],
    remainder="drop",
)

final_pipe = Pipeline(
    [
        ("preprocessor", final_preprocessor),
        ("regressor", best_model),
    ]
)

final_pipe.fit(X, y)

# ===== SAVE MODEL =====
joblib.dump(final_pipe, config_dict["MODEL_PATH"])

# ===== SAVE VERSION =====
version_str = f"{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
with open(config_dict["VERSION_PATH"], "w") as f:
    f.write(version_str)

print(
    f"✅ Saved best model: {best_model_name} with R²={results[best_model_name]['R2_mean']:.4f}"
)
print(f"📂 Model artifact: {config_dict['MODEL_PATH']}")
print(f"📄 Version: {version_str}")
print(f"📊 Metrics saved to {config_dict['METRICS_PATH']}")

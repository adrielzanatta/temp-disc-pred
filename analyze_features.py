import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tools import outlier_removal
from FeatureEngineer import FeatureEngineer

# ===== CONFIG =====
DATA_PATH = "data/301A_2ND_CLEAN.csv"
TARGET_COLUMN = "temp_dis"
CATEGORICAL_FEATURES = ["prod_grade"]
NOT_FILTER_OUTLIERS = ["prod_grade", "timestamp"]
SEED = 42

# ===== LOAD DATA =====
df = pd.read_csv(
    DATA_PATH, sep=";", decimal=",", parse_dates=["timestamp"], dayfirst=True
)


# ===== OUTLIER REMOVAL =====
df = outlier_removal(df, NOT_FILTER_OUTLIERS)

# ===== BASIC INFO =====
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isna().sum())

# ===== CORRELATION MATRIX =====
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()
print("📊 Correlation matrix saved to correlation_matrix.png")

# ===== VIF (Multicollinearity) =====
features_for_vif = numeric_df.drop(
    columns=[TARGET_COLUMN] + [cat for cat in CATEGORICAL_FEATURES]
).dropna()
vif_data = pd.DataFrame()
vif_data["Feature"] = features_for_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(features_for_vif.values, i)
    for i in range(features_for_vif.shape[1])
]
print("\n--- Variance Inflation Factor (VIF) ---")
print(vif_data.sort_values(by="VIF", ascending=False))

# ===== PERMUTATION IMPORTANCE =====
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

numeric_features = [col for col in X.columns if col not in CATEGORICAL_FEATURES]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", SimpleImputer(strategy="mean"), numeric_features),
    ]
)

model = RandomForestRegressor(random_state=SEED)
pipe = Pipeline(
    [
        ("features", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

pipe.fit(X, y)
perm_importance = permutation_importance(pipe, X, y, n_repeats=10, random_state=SEED)

# Get final feature names
encoded_cat = (
    pipe.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(CATEGORICAL_FEATURES)
)
all_feature_names = np.concatenate([encoded_cat, numeric_features])

importance_df = pd.DataFrame(
    {"Feature": all_feature_names, "Importance": perm_importance.importances_mean}
).sort_values(by="Importance", ascending=False)

print("\n--- Permutation Importance ---")
print(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Permutation Importance (RandomForest)")
plt.tight_layout()
plt.savefig("permutation_importance.png")
plt.close()
print("📊 Permutation importance saved to permutation_importance.png")

# ===== DROP CANDIDATES =====
high_vif = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
low_importance = importance_df[importance_df["Importance"] <= 0]["Feature"].tolist()
suggested_drops = set(high_vif).intersection(set(low_importance))

print("\n--- Suggested Drop Candidates ---")
if suggested_drops:
    print(suggested_drops)
else:
    print("No strong drop candidates found")

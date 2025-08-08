from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, timestamp_col="timestamp", cat_col="prod_grade"):
        self.timestamp_col = timestamp_col
        self.cat_col = cat_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Convert timestamp to numeric
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(
                df[self.timestamp_col], errors="coerce"
            )
            df.dropna(subset=[self.timestamp_col], inplace=True)
            df["timestamp_numeric"] = df[self.timestamp_col].astype("int64") // 10**9
            df.drop(columns=[self.timestamp_col], inplace=True)

        # Create p_ratio
        if "p_dis" in df.columns and "p_suc" in df.columns:
            df["p_ratio"] = (df["p_dis"] + 1) / (df["p_suc"] + 1)

        return df

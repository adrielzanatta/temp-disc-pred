from typing import List
import numpy as np
import pandas as pd
import tomli


def outlier_removal(df: pd.DataFrame, not_filter_outliers: List[str]):
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols_for_outliers = [
        col for col in numerical_cols if col not in not_filter_outliers
    ]
    Q1 = df[numerical_cols_for_outliers].quantile(0.25)
    Q3 = df[numerical_cols_for_outliers].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    for column in numerical_cols_for_outliers:
        df = df[
            (df[column] >= lower_bound[column]) & (df[column] <= upper_bound[column])
        ]
    return df


def prepare_df(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Prepares the DataFrame by converting the timestamp column to numeric and
    calculating the p_ratio.
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df.dropna(subset=[timestamp_col], inplace=True)
    df["timestamp_numeric"] = df[timestamp_col].astype("int64") // 10**9
    df.drop(columns=[timestamp_col], inplace=True)

    df["p_ratio"] = (df["p_dis"] + 1) / (df["p_suc"] + 1)

    return df


def load_config(config_path: str, config_list: List[str]) -> dict:
    """
    Loads configuration from a TOML file.
    """
    config_dict = {}
    with open(config_path, "rb") as f:
        tomli_dict = tomli.load(f)

        for key in config_list:
            if key not in tomli_dict:
                raise KeyError(f"Missing key '{key}' in configuration file.")
            config_dict[key] = tomli_dict[key]

    return config_dict

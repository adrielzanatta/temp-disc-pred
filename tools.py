from typing import List
import numpy as np
import pandas as pd


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

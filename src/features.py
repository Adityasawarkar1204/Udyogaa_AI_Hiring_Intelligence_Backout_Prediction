# src/features.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path="data/udyogaa_synthetic.csv"):
    df = pd.read_csv(path)
    return df


def build_features(df):
    df = df.copy()

    # Feature Engineering
    df["salary_gap"] = df["salary_expectation_lpa"] - df["salary_offered_lpa"]
    df["rel_salary_hike"] = df["salary_offered_lpa"] / (df["salary_expectation_lpa"] + 1e-6)

    df["stability_score"] = np.clip(
        1 - (df["past_job_switches"] / 6) + (df["avg_tenure_months"] / 48),
        0,
        2
    )

    df["comm_bin"] = pd.cut(
        df["communication_score"],
        bins=[0, 40, 65, 100],
        labels=["low", "mid", "high"]
    )

    y = df["joined"].astype(int)

    drop_cols = ["candidate_id", "joined"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y


def get_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )

    return preprocessor, numeric_cols, cat_cols


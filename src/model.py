# src/model.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from features import load_data, build_features, get_preprocessor


MODEL_DIR = "src/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train(save=True, data_path="data/udyogaa_synthetic.csv"):
    print("Loading data...")
    df = load_data(data_path)

    X, y = build_features(df)
    preprocessor, _, _ = get_preprocessor(X)

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training models...\n")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            eval_metric="logloss"
        ),
        "SVM": SVC(probability=True)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        results[name] = auc
        trained_models[name] = model
        print(f"{name} AUC: {auc:.4f}")

    # Select best model
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]

    print("\nBest Model Selected:", best_model_name)
    print("Best AUC Score:", round(results[best_model_name], 4))

    if save:
        joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.joblib")
        joblib.dump(best_model, f"{MODEL_DIR}/best_model.joblib")
        print("\nBest model and preprocessor saved successfully!")

    return best_model


if __name__ == "__main__":
    train()
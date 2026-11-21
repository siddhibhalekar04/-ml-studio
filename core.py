"""Reusable ML operations; no Streamlit dependencies so workflows are testable."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000
MAX_CATEGORIES = 100
CLASSIFICATION_MODELS = ("Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting", "KNN")
REGRESSION_MODELS = ("Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting", "KNN")


@dataclass
class FitResults:
    models: dict[str, Pipeline]
    scores: pd.DataFrame
    predictions: dict[str, np.ndarray]
    y_test: pd.Series
    x_test: pd.DataFrame
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    problem_type: str
    warnings: list[str]


def eligible_features(df: pd.DataFrame, target: str) -> tuple[list[str], list[str], list[str]]:
    """Return numeric, categorical and unsupported columns (target excluded)."""
    available = df.drop(columns=[target])
    num = available.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = available.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    unsupported = [c for c in available if c not in num + cat]
    return num, cat, unsupported


def validate_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("The dataset is empty.")
    if df.shape[0] > MAX_ROWS:
        raise ValueError(f"This version supports up to {MAX_ROWS:,} rows per dataset.")
    if df.columns.has_duplicates:
        raise ValueError("Duplicate column names detected. Rename columns before uploading.")
    if any(not isinstance(c, str) or not c.strip() for c in df.columns):
        raise ValueError("Every column must have a nonempty name.")


def prepare_training(
    df: pd.DataFrame, target: str, features: list[str], problem: str,
    test_fraction: float, random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str], list[str]]:
    validate_data(df)
    if target not in df.columns:
        raise ValueError("Selected target column is missing.")
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("The test fraction must be between 10% and 40%.")
    if not features or target in features or len(set(features)) != len(features):
        raise ValueError("Select one or more distinct input features, excluding the target.")
    if any(c not in df.columns for c in features):
        raise ValueError("One or more selected features are missing from the dataset.")
    if problem not in ("Classification", "Regression"):
        raise ValueError("Choose Classification or Regression.")

    clean = df.loc[df[target].notna(), features + [target]].copy()
    if len(clean) < 12:
        raise ValueError("At least 12 rows with non-missing target values are required.")
    x = clean[features].copy()
    y = clean[target].copy()
    num = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if len(num) + len(cat) != len(features):
        unsupported = sorted(set(features) - set(num) - set(cat))
        raise ValueError(f"Unsupported feature types: {', '.join(unsupported)}. Convert or exclude these columns.")
    if any(x[c].isna().all() for c in features):
        empty_cols = [c for c in features if x[c].isna().all()]
        raise ValueError(f"These selected columns are entirely missing: {', '.join(empty_cols)}.")
    for col in num:
        x[col] = pd.to_numeric(x[col], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for col in cat:
        distinct = x[col].nunique(dropna=True)
        if distinct > MAX_CATEGORIES:
            raise ValueError(f"'{col}' contains {distinct:,} unique values. Exclude ID/name/free-text columns (limit {MAX_CATEGORIES} categories per feature).")
        x[col] = x[col].astype("object")
        x[col] = x[col].where(x[col].isna(), x[col].astype(str))

    if problem == "Regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression requires a numeric target. Choose Classification or a numerical target.")
        y = pd.to_numeric(y, errors="coerce")
        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError("Regression targets must be finite numeric values.")
        stratify = None
    else:
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
        if y.nunique() > 30:
            raise ValueError("Classification currently supports up to 30 classes; this target looks continuous or is an identifier.")
        counts = y.value_counts()
        if counts.min() < 2:
            raise ValueError("Every target class needs at least two examples for stratified splitting.")
        n_classes = len(counts)
        n_test = int(np.ceil(len(y) * test_fraction))
        if n_test < n_classes or len(y) - n_test < n_classes:
            raise ValueError("The train/test split is too small for the number of classes. Increase the test size or use a dataset with more rows.")
        stratify = y
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_fraction, random_state=random_state, stratify=stratify
    )
    train_empty = [col for col in features if x_train[col].isna().all()]
    if train_empty:
        raise ValueError("Columns have no usable values in the training split: " + ", ".join(train_empty))
    return x_train, x_test, y_train, y_test, num, cat, []


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(name: str, problem: str, train_size: int):
    if problem == "Classification":
        choices = {
            "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=42),
            "Decision Tree": lambda: DecisionTreeClassifier(max_depth=8, random_state=42),
            "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=100,max_depth=12,random_state=42,n_jobs=-1),
            "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=100,random_state=42),
            "KNN": lambda: KNeighborsClassifier(n_neighbors=min(5, train_size)),
        }
    else:
        choices = {
            "Linear Regression": lambda: LinearRegression(),
            "Ridge Regression": lambda: Ridge(alpha=1.0),
            "Decision Tree": lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
            "Random Forest": lambda: RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "Extra Trees": lambda: ExtraTreesRegressor(n_estimators=100,max_depth=12,random_state=42,n_jobs=-1),
            "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100,random_state=42),
            "KNN": lambda: KNeighborsRegressor(n_neighbors=min(5, train_size)),
        }
    if name not in choices:
        raise ValueError(f"'{name}' isn't a valid {problem.lower()} algorithm.")
    return choices[name]()


def run_models(df: pd.DataFrame, target: str, features: list[str], problem: str,
               test_fraction: float, selected_models: list[str]) -> FitResults:
    if not selected_models:
        raise ValueError("Choose at least one algorithm.")
    x_train, x_test, y_train, y_test, num, cat, warnings = prepare_training(
        df, target, features, problem, test_fraction
    )
    models: dict[str, Pipeline] = {}
    preds: dict[str, np.ndarray] = {}
    metrics_rows = []
    for name in selected_models:
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(num, cat)),
            ("model", make_model(name, problem, len(x_train))),
        ])
        started = perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = perf_counter() - started
        pred = pipeline.predict(x_test)
        row: dict[str, Any] = {"Model": name, "Train time (s)":round(fit_seconds,3)}
        if problem == "Classification":
            row.update({
                "Accuracy": accuracy_score(y_test, pred),
                "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
                "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
                "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
            })
        else:
            row.update({
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                "R²": r2_score(y_test, pred),
            })
        models[name] = pipeline
        preds[name] = pred
        metrics_rows.append(row)
    metric = "F1 (weighted)" if problem == "Classification" else "RMSE"
    scores = pd.DataFrame(metrics_rows).sort_values(metric, ascending=(problem == "Regression")).reset_index(drop=True)
    return FitResults(models, scores, preds, y_test, x_test, features, num, cat, problem, warnings)


def predictions_table(model: Pipeline, new_data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    absent = [c for c in feature_columns if c not in new_data]
    if absent:
        raise ValueError(f"Missing required columns: {', '.join(absent)}")
    output = new_data.copy()
    output["Prediction"] = model.predict(new_data[feature_columns])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(new_data[feature_columns])
        if probabilities.shape[1] <= 10:
            for idx, label in enumerate(model.named_steps["model"].classes_):
                output[f"Probability ({label})"] = probabilities[:, idx].round(4)
    return output

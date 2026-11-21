"""Lightweight, deterministic dataset diagnostics for the Streamlit app."""
from __future__ import annotations

import pandas as pd


def quality_report(df: pd.DataFrame) -> dict:
    total = max(1, df.size)
    missing_counts = df.isna().sum()
    unique = df.nunique(dropna=True)
    identifiers = [col for col in df if (
        (str(col).lower().endswith('id') and unique[col] >= len(df)*.9)
        or (len(df)>20 and unique[col]==len(df) and df[col].dtype=='object')
    )]
    columns = pd.DataFrame({
        'Column':df.columns,
        'Type':df.dtypes.astype(str).values,
        'Missing':missing_counts.values,
        'Missing %':(100*missing_counts/len(df)).round(2).values,
        'Unique':unique.values,
        'Potential ID': [col in identifiers for col in df.columns],
    })
    return {
        'completeness': 100*(1-df.isna().sum().sum()/total),
        'duplicates':int(df.duplicated().sum()),
        'constant_columns':[c for c in df if unique[c]<=1],
        'identifier_columns':identifiers,
        'columns':columns,
    }


def generate_insights(df: pd.DataFrame) -> list[str]:
    profile=quality_report(df)
    tips=[]
    if profile['duplicates']:
        tips.append(f"{profile['duplicates']:,} duplicate rows detected. Review whether they represent repeated measurements before removing them.")
    high_missing=profile['columns'].loc[profile['columns']['Missing %']>=30,'Column'].tolist()
    if high_missing:
        tips.append('Columns with at least 30% missing data: '+', '.join(high_missing[:8])+'. Consider investigating missingness before modeling.')
    if profile['constant_columns']:
        tips.append('Constant or empty columns may add no useful predictive information: '+', '.join(map(str,profile['constant_columns'][:8])))
    if profile['identifier_columns']:
        tips.append('Potential identifier columns should normally be excluded from model features: '+', '.join(map(str,profile['identifier_columns'][:8])))
    high_card=[col for col in df.select_dtypes(include=['object','string','category']) if df[col].nunique(dropna=True)>100]
    if high_card:
        tips.append('High-cardinality text fields may be unsuitable for one-hot encoding: '+', '.join(map(str,high_card[:8])))
    if not tips:
        tips.append('No basic structural issues detected. Still check the target distribution, data provenance, and potential leakage before training.')
    return tips


def score_guide(problem: str) -> list[str]:
    if problem=='Classification':
        return [
            'Accuracy: fraction of all records predicted correctly; can hide class imbalance.',
            'Weighted F1: class-level F1 averaged by class support; inspect individual classes as well.',
            'Confusion matrix: shows which true classes are confused with predicted classes.',
            'ROC-AUC (binary): summarizes discrimination across decision thresholds; not a measure of calibrated probability.'
        ]
    return [
        'MAE: mean absolute prediction error; lower is better, in target units.',
        'RMSE: square root of mean squared error, penalizing larger errors; lower is better.',
        'R²: comparison against predicting the mean; may be negative on held-out data.',
        'Residuals: inspect patterns to diagnose nonlinearities, outliers or systematic errors.'
    ]

# 🧠 ML Studio Pro — Enhanced No-Code Machine Learning

An expanded, slate-blue dashboard-style Streamlit application for dataset exploration, data quality diagnostics, supervised model training, explainable evaluation, predictions, and exports. Built on the original ML Studio codebase.

## New in this enhanced version

- Slate-blue and glass-inspired dashboard with overview KPI cards.
- Expanded **Explorer** with feature-to-feature scatter plots, box plots and categorical frequency lab.
- **Data Quality** workspace: column-level quality table, duplicates, constant features, likely identifiers, basic data recommendations and safe downloadable cleaning options.
- **Advanced Evaluation**: class-by-class report, binary ROC/AUC (where supported), feature influence for supported models, model comparison and training duration.
- **Additional algorithms**: Extra Trees and Gradient Boosting classifiers/regressors.
- **Experiment History** for the current Streamlit session, with CSV download.

## Features

- Upload custom CSV files (up to 20 MB / 100,000 rows) or try two synthetic example datasets.
- Preview dataset; inspect missing values, duplicates, distributions, correlations and summary statistics.
- Select targets, predictors, classification/regression task, and train/test ratio.
- Automatic train-only preprocessing: median numeric imputation, scaling, most-frequent categorical imputation, and one-hot encoding.
- Classification: Logistic Regression, Decision Tree, Random Forest, and KNN.
- Regression: Linear Regression, Ridge, Decision Tree, Random Forest, and KNN.
- Multiple-model comparison: Accuracy / weighted Precision / weighted Recall / weighted F1, or MAE / RMSE / R².
- Confusion matrix, regression scatter and residual charts, single-row prediction form, and bulk CSV prediction.
- Export predictions, evaluation results, fitted models, and metadata.

## Getting started on Windows (VS Code)

**Python 3.13 recommended.** Open the extracted project folder in VS Code.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The commands above **do not require activation** of the virtual environment, so they work even if your PowerShell execution policy blocks activation scripts. Open http://localhost:8501.

Alternatively, double-click `start_windows.bat` after installation.

## Setup using existing environment

If you already have a working Python 3.13 `.venv` in your original project folder, you can copy the contents of this ZIP *except* any old environment and install requirements before launching.

**Before replacing an existing `app.py`, back it up.** Extract this ZIP into a **new folder** first to keep your previous project safe.

## Demo datasets

- `data/demo_passengers.csv`: 500 rows of synthetic passenger data for binary classification (`Survived`).
- `data/demo_housing.csv`: 500 rows of synthetic house listings for regression (`SalePrice`).

These examples are fully synthetic, not official Kaggle or real-world records. You may also upload your original Kaggle Titanic `train.csv` via the sidebar.

## Usage

1. Upload a CSV or select a synthetic example in the sidebar.
2. Explore your dataset under **Explore**.
3. Select a target, features, task and models under **Configure & Train**.
4. Click **Train selected models**.
5. Open **Evaluate** to inspect held-out test-set metrics and charts.
6. Open **Predict & Export** to predict a single row or batch CSV and download results/models.

After changing configuration or uploading different data, retrain the models before evaluating or predicting.

## Limitations & safety

- This is a supervised learning teaching/portfolio tool, not an automated machine learning service for high-stakes decisions.
- For classification, up to 30 target classes are supported; categorical features over 100 distinct values are rejected; ID and free-text features should be excluded.
- Missing targets are dropped; entirely missing selected feature columns are rejected.
- The application requires reasonable class sizes for stratified splitting.
- Evaluation uses one stratified held-out test split for classification, and one randomized held-out split for regression. It does **not** perform cross-validation, hyperparameter tuning, confidence calibration, or fairness evaluation.
- Results from tuning choices based on the same hold-out set may be optimistic; reserve an independent final test set for serious deployments.
- Experiment history and trained models persist in the current Streamlit session, not across restarts. Download joblib if needed; only load model files from trusted sources.
- Date/time and other unsupported column dtypes must be converted to numeric/categorical or excluded.
- Hosting a publicly accessible app may expose uploads to the service operator. Do not upload sensitive data to untrusted hosting.

## Folder layout

```
ML-Studio-Complete/
├── app.py                    # Streamlit UI
├── mlstudio/core.py          # Training, validation and prediction logic
├── mlstudio/insights.py      # Dataset quality profiling and rule-based recommendations
├── data/                     # Synthetic demonstration datasets
├── tests/test_core.py        # Automated workflow tests
├── .streamlit/config.toml    # Streamlit theme
├── requirements.txt
├── start_windows.bat
└── README.md
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

## Portfolio notes

A credible project writeup should include the dataset source, selected features, algorithm comparisons, held-out test metrics, limitations, screenshots and a live demo. For the provided synthetic data, clearly label results as synthetic.

## Version notes

This enhanced build is separate from the original. The bundled datasets are synthetic. The Streamlit UI must be visually reviewed locally after installation: automated model/insights tests and Python compilation were run in the build environment, but Streamlit was not available there for a live browser smoke test.

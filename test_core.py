import numpy as np
import pandas as pd
import pytest

from mlstudio.core import (
    prepare_training, predictions_table, run_models, validate_data,
)


def test_classification_demo():
    df = pd.read_csv('data/demo_passengers.csv')
    result = run_models(df, 'Survived', ['Pclass', 'Sex', 'Age', 'Fare'],
                        'Classification', .2, ['Logistic Regression', 'Random Forest', 'KNN'])
    assert len(result.y_test) == 100
    assert set(result.scores['Model']) == {'Logistic Regression', 'Random Forest', 'KNN'}
    assert result.scores['Accuracy'].between(0, 1).all()
    assert result.scores['F1 (weighted)'].between(0, 1).all()
    sample = pd.DataFrame([{'Pclass': 3, 'Sex': 'female', 'Age': np.nan, 'Fare': 12.5}])
    prediction = predictions_table(result.models['Logistic Regression'], sample, result.feature_columns)
    assert len(prediction) == 1
    assert 'Prediction' in prediction
    assert 'Probability (1)' in prediction


def test_regression_demo():
    df = pd.read_csv('data/demo_housing.csv')
    result = run_models(df, 'SalePrice', ['AreaSqFt', 'Bedrooms', 'Location', 'YearBuilt'],
                        'Regression', .2, ['Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'KNN'])
    assert result.scores['MAE'].ge(0).all()
    assert result.scores['RMSE'].ge(0).all()
    assert len(result.y_test) == 100
    assert len(predictions_table(result.models['Random Forest'], df.head(2), result.feature_columns)) == 2


def test_validation_rejects_target_as_feature():
    df = pd.read_csv('data/demo_passengers.csv')
    with pytest.raises(ValueError, match='excluding the target'):
        prepare_training(df, 'Survived', ['Survived', 'Age'], 'Classification', .2)


def test_validation_rejects_id_target():
    df = pd.read_csv('data/demo_passengers.csv')
    df['PassengerId'] = range(len(df))
    with pytest.raises(ValueError, match='up to 30 classes'):
        prepare_training(df, 'PassengerId', ['Age'], 'Classification', .2)


def test_prediction_requires_features():
    df = pd.read_csv('data/demo_passengers.csv')
    result = run_models(df, 'Survived', ['Sex', 'Age'], 'Classification', .2, ['Logistic Regression'])
    with pytest.raises(ValueError, match='Missing required columns'):
        predictions_table(result.models['Logistic Regression'], pd.DataFrame({'Age': [30]}), result.feature_columns)


def test_multiclass():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({'X': rng.normal(size=90), 'Group': rng.choice(['a','b'],90),
                       'target': np.repeat(['red', 'green', 'blue'],30)})
    result = run_models(df, 'target', ['X','Group'], 'Classification', .2, ['Logistic Regression'])
    assert len(result.y_test) == 18
    assert result.scores['F1 (weighted)'].between(0,1).all()


def test_duplicate_columns_validation():
    df = pd.DataFrame([[1,2]], columns=['X','X'])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_data(df)


def test_new_algorithms_available():
    from mlstudio.core import CLASSIFICATION_MODELS, REGRESSION_MODELS, make_model
    assert "Extra Trees" in CLASSIFICATION_MODELS
    assert "Gradient Boosting" in REGRESSION_MODELS
    assert make_model("Extra Trees", "Classification", 50) is not None
    assert make_model("Gradient Boosting", "Regression", 50) is not None


def test_data_quality_diagnostics():
    import pandas as pd
    from mlstudio.insights import quality_report, generate_insights
    df = pd.DataFrame({"CustomerID": [1,2,3,4], "Const": ["x"]*4, "A": [1,None,2,3]})
    report=quality_report(df)
    assert "Const" in report["constant_columns"]
    assert report["columns"].loc[report["columns"]["Column"]=="A","Missing"].iloc[0]==1
    assert generate_insights(df)

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def create_model_and_data():

    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model, X_test, y_test


def test_random_forest_accuracy():

    model, X_test, y_test = create_model_and_data()

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("Test accuracy:", accuracy)

    assert accuracy >= 0.90


def test_prediction_length():

    model, X_test, _ = create_model_and_data()

    predictions = model.predict(
        X_test
    )

    assert len(predictions) == len(X_test)


def test_prediction_classes():

    model, X_test, _ = create_model_and_data()

    predictions = model.predict(
        X_test
    )

    valid_classes = {0, 1, 2}

    assert set(predictions).issubset(
        valid_classes
    )


def test_model_reproducibility():

    model1, X_test, _ = create_model_and_data()
    model2, _, _ = create_model_and_data()

    predictions1 = model1.predict(
        X_test
    )

    predictions2 = model2.predict(
        X_test
    )

    assert np.array_equal(
        predictions1,
        predictions2
    )
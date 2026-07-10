from __future__ import annotations

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def time_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
):
    """
    Splits data into train/test sets.
    """

    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Random Forest classifier to make trade or not-trade binary decision given
    predicted next-day return.

    Target:
    1 if next-day return > 0
    0 otherwise
    """

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=random_state,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    return model


def predict_up_close_probability(
    model: RandomForestClassifier,
    X: pd.DataFrame,
) -> pd.Series:
    """
    Returns the probability that next-day return is positive.
    """

    class_labels = list(model.classes_)

    if 1 not in class_labels:
        raise ValueError("Model was not properly trained.")

    class_1_index = class_labels.index(1)

    probabilities = model.predict_proba(X)[:, class_1_index]

    return pd.Series(
        probabilities,
        index=X.index,
        name="ml_probability",
    )


def probability_to_signal(
    probabilities: pd.Series,
    threshold: float = 0.60,
) -> pd.Series:
    """
    Converts model probabilities into long-only trading signals.

    rule:
    - Long if probability > 0.60
    - Flat if probability <= 0.60

    Returns:
    - 1 for long
    - 0 for flat
    """
    # converts threshold booleans to signals
    signals = (probabilities > threshold).astype(int)
    signals.name = "ml_signal"

    return signals


def print_model_summary(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    symbol: str = "the ticker",
) -> None:
    """
    Prints classification performance and explains the confusion matrix.

    Target meaning:
    - 0 = next-day return <= 0
    - 1 = next-day return > 0
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    # The row, col position in matrix give us actual and predicted respectively
    true_negative = matrix[0, 0]
    false_positive = matrix[0, 1]
    false_negative = matrix[1, 0]
    true_positive = matrix[1, 1]

    print("\nModel Evaluation")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nConfusion Matrix Meaning")
    print("-" * 40)
    print(f"{'Count':>8} | Meaning")
    print(f"{'-' * 8} | {'-' * 70}")
    print(f"{true_negative:>8} | Model correctly predicted {symbol} would not go up")
    print(f"{false_positive:>8} | Model predicted {symbol} would go up, but it did not")
    print(f"{false_negative:>8} | Model predicted {symbol} would not go up, but it did")
    print(f"{true_positive:>8} | Model correctly predicted {symbol} would go up")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )
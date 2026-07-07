from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedPCA:
    scaler: StandardScaler
    pca: PCA
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    feature_columns: list[str]


def fit_pca(
    X_train: pd.DataFrame,
    variance_threshold: float = 0.80,
) -> tuple[pd.DataFrame, FittedPCA]:
    """
    Fits StandardScaler and PCA on ml features data.

    Keeps smallest-number of components needed to explain at least
    80% of the total feature variance.
    """

    if X_train.empty:
        raise ValueError("X_train is empty.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    full_pca = PCA()
    full_pca.fit(X_train_scaled)

    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)

    pca = PCA(n_components=n_components)
    X_train_pca_array = pca.fit_transform(X_train_scaled)

    component_columns = [f"PC{i + 1}" for i in range(n_components)]

    X_train_pca = pd.DataFrame(
        X_train_pca_array,
        index=X_train.index,
        columns=component_columns,
    )

    fitted_pca = FittedPCA(
        scaler=scaler,
        pca=pca,
        n_components=n_components,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cumulative_variance_ratio=np.cumsum(pca.explained_variance_ratio_),
        feature_columns=list(X_train.columns),
    )

    return X_train_pca, fitted_pca


def transform_pca(
    X: pd.DataFrame,
    fitted_pca: FittedPCA,
) -> pd.DataFrame:
    """
    Applies a previously fitted scaler and PCA transformer to new data.
    """

    missing = [col for col in fitted_pca.feature_columns if col not in X.columns]

    if missing:
        raise ValueError(f"Missing columns required for PCA transform: {missing}")

    X = X[fitted_pca.feature_columns].copy()

    X_scaled = fitted_pca.scaler.transform(X)
    X_pca_array = fitted_pca.pca.transform(X_scaled)

    component_columns = [f"PC{i + 1}" for i in range(fitted_pca.n_components)]

    return pd.DataFrame(
        X_pca_array,
        index=X.index,
        columns=component_columns,
    )


def print_pca_summary(fitted_pca: FittedPCA) -> None:
    """
    Print debug for explained variance information.
    """

    print("\nPCA Summary")
    print("-" * 40)
    print(f"Selected components: {fitted_pca.n_components}")

    for i, ratio in enumerate(fitted_pca.explained_variance_ratio, start=1):
        print(f"PC{i}: {ratio:.4f}")

    print(f"Total explained variance: {fitted_pca.explained_variance_ratio.sum():.4f}")
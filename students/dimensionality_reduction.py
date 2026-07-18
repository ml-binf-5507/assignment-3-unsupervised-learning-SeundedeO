"""
dimensionality_reduction.py

Students implement three wrappers that apply PCA, t-SNE, and UMAP to a feature
matrix X and return the transformed data.
"""

from typing import Any

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def apply_pca(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Perform PCA on the input array.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    n_components : int, default=2
        Number of principal components to keep.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Transformed data with shape (n_samples, n_components).
    """

    pca = PCA(
        n_components=n_components,
        random_state=random_state
    )

    X_pca = pca.fit_transform(X)

    return X_pca



def apply_tsne(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Apply t-SNE to the data.

    The sklearn.manifold.TSNE class may be used. The function should return the
    2-D embedding.
    """

    tsne = TSNE(
        n_components=n_components,
        random_state=random_state
    )

    X_tsne = tsne.fit_transform(X)

    return X_tsne



def apply_umap(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Apply UMAP to the data.

    Requires the `umap-learn` package. Return an array of shape
    (n_samples, n_components).
    """

    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state
    )

    X_umap = reducer.fit_transform(X)

    return X_umap
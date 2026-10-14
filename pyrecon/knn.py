import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings


def _compute_weights(distances, scheme="bisquare", eps=1e-12):
    
    """
    Convert neighbor distances into normalized weights for KNN prediction.

    Parameters
    ----------
    distances : array-like
        Euclidean distances to the k nearest neighbors, shape (k,).
    
    scheme : str, optional
        Weighting scheme to apply:
        - 'bisquare'  : smooth kernel (1 - u²)² where u = d/d_max.
                        Gives high weight to close neighbors, tapers to
                        near-zero for the farthest. 
        - 'distance'  : inverse distance (1/d). Sharper drop-off than bisquare.
        - 'uniform'   : equal weight for all neighbors. Simple average.
    
    eps : float, optional
        Small constant added to distances to prevent division by zero.
        Default is 1e-12.

    Returns
    -------
    
    w : np.ndarray
        Normalized weights of shape (k,) that sum to 1.

    """
   
    d = np.asarray(distances, dtype=float)
    if np.allclose(d, 0.0):
        w = np.zeros_like(d); w[0] = 1.0
        return w
    if scheme == "uniform":
        w = np.ones_like(d)
    elif scheme == "distance":
        w = 1.0 / (d + eps)
    elif scheme == "bisquare":
        dmax = d.max() + eps
        u = d / dmax
        w = np.zeros_like(d)
        m = u < 1.0
        w[m] = (1 - u[m]**2)**2
        if not np.any(w > 0):
            w = 1.0 / (d + eps)
    else:
        raise ValueError("scheme must be 'bisquare', 'distance', or 'uniform'")
    s = w.sum()
    return w/s if s > 0 else np.full_like(w, 1.0/len(w))



def loocv_knn_mean(PCA_scores, observed, number_pca, calib_start, calib_end, k=9, weight_scheme="bisquare"):
    
    """
    Leave-one-out cross-validated KNN skill over the calibration period.


    Parameters
    ----------
    
    PCA_scores : pandas.DataFrame
        PCA scores, year-indexed, with columns named 'PC 1', 'PC 2', ..
    
    observed : pandas.Series
        Instrumental target (e.g. streamflow), year-indexed.
    
    number_pca : int
        Number of PCs to choose.
    
    calib_start, calib_end : int
        First and last calibration years.
    
    k : int, optional
        Number of nearest neighbors. Default is 9.
    
    weight_scheme : {'bisquare', 'distance', 'uniform'}, optional
        Weighting scheme passed to ``_compute_weights``. Default 'bisquare'.

    Returns
    -------
    
    y_hat : np.ndarray, shape (n_calib,)
        LOOCV predicted streamflow for each calibration year.
    
    r2 : float
        LOOCV coefficient of determination.
    
    rmse : float
        LOOCV root-mean-square error.
    
    """

    pc_cols = [f'PC {i + 1}' for i in range(number_pca)]
    X = PCA_scores[pc_cols].loc[calib_start:calib_end]
    y = np.asarray(observed.loc[calib_start:calib_end], dtype=float)

    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X)
    distances, indices = nbrs.kneighbors(X, return_distance=True)

    y_hat = np.empty(n, float)

    for i in range(n):
        d_all, idx_all = distances[i], indices[i]
        d = d_all[1:][:k]
        idx = idx_all[1:][:k]
        w = _compute_weights(d, scheme=weight_scheme)
        y_hat[i] = np.dot(w, y[idx])
        
    r2  = r2_score(y, y_hat)
    print(f"Leave-One-Out Cross-Validation (LOOCV) R² : {r2:.4f}")
    
    
    rmse = np.sqrt(np.mean((y - y_hat)**2))
    print(f"Leave-One-Out Cross-Validation (LOOCV) RMSE : {rmse:.4f}")

    return y_hat,r2,rmse


def knn_reconstruct(pca_scores, observed, number_pca,
                    calib_start, calib_end, recon_start, recon_end,
                    k=9, weight_scheme="bisquare", ensemble_size=0):
    
    """
    Full-period streamflow reconstruction knn regression.


    Parameters
    ----------
    
    pca_scores : pandas.DataFrame
        PCA scores, year-indexed, with columns named 'PC 1', 'PC 2', .
    
    observed : pandas.Series
        Instrumental target (e.g. streamflow), year-indexed.
    
    number_pca : int
        Number of leading PCs to use.
    
    calib_start, calib_end : int
        First and last calibration years.
    
    recon_start, recon_end : int
        First and last years to reconstruct.
    
    k : int, optional
        Number of nearest neighbors. Default is 9.
    
    weight_scheme : {'bisquare', 'distance', 'uniform'}, optional
        Weighting scheme passed to ``_compute_weights``. Default 'bisquare'.
    
    ensemble_size : int, optional
        Number of bootstrap draws per year (sampling neighbors with probability
        equal to their weight) for uncertainty estimation. 0 (default) skips it.


    Returns
    -------
    recon : pandas.Series, shape (n_target,)
        Reconstructed streamflow over recon_start..recon_end, indexed by year,
        named 'knn_reconstruction'.
    
    ensembles : list of np.ndarray or None
        Per-year bootstrap samples of shape (ensemble_size,), for computing
        confidence bands (e.g. np.percentile). None if ensemble_size=0.
    
    indices : np.ndarray, shape (n_target, k)
        Calibration-row indices of the k nearest neighbors for each target year.
    
    distances : np.ndarray, shape (n_target, k)
        Euclidean distances to those k nearest neighbors.


    """

    pc_cols = [f'PC {i + 1}' for i in range(number_pca)]
    Z_train = pca_scores[pc_cols].loc[calib_start:calib_end].values
    Z_target = pca_scores[pc_cols].loc[recon_start:recon_end].values
    target_index = pca_scores[pc_cols].loc[recon_start:recon_end].index
    y_train = np.asarray(observed.loc[calib_start:calib_end], dtype=float).ravel()


    rng = np.random.default_rng(42)
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Z_train)
    distances, indices = nbrs.kneighbors(Z_target, return_distance=True)

    n_target = Z_target.shape[0]
    mean_pred = np.empty(n_target, float)
    ensembles = [] if ensemble_size and ensemble_size > 0 else None

    for i in range(n_target):
        w = _compute_weights(distances[i], scheme=weight_scheme)
        yi = y_train[indices[i]]
        mean_pred[i] = np.dot(w, yi)
        if ensembles is not None:
            ensembles.append(rng.choice(yi, size=ensemble_size, replace=True, p=w))

    recon = pd.Series(mean_pred, index=target_index, name="knn_reconstruction")
    
    return recon, ensembles, indices, distances


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score



def find_best_set_of_PCA(chrons, max_pca, recon_start, recon_end, threshold=None, plot =True):
    
    """
    Standardize predictors, fit a PCA, plot the variance, and return everything.

    Parameters
    ----------
    chrons : pandas.DataFrame
        Year-indexed chronology matrix (rows = years, columns = chronologies).
    
    max_pca : int
        Maximum Number of principal components to retain.
    
    recon_start, recon_end : int
        First and last year of the full reconstrution period (inclusive), used to
        slice ``PCA_scores`` on its year index.
    
    threshold : float, optional
        If given, draw a horizontal reference line (in %) on the cumulative panel.
    
    plot : bool, optional
        If "true" (default), plot the explained variance ration against each
        PCA, and cumalative explained variance against #of PCAs.
    

    Returns
    -------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler (needed to transform the full/paleo period consistently).
    
    pca : sklearn.decomposition.PCA
        Fitted PCA object.
    
    scores : pandas.DataFrame
        PC scores, year-indexed, columns 'PC 1' … 'PC n'.
    
    loadings : pandas.DataFrame
        Component loadings (PCs × chronologies).
    
    variance : pandas.DataFrame
        Per-PC explained variance, explained variance ratio, and singular values.
    
    cum_var : numpy.ndarray
        Cumulative explained variance ratio.
    
    fig, axes : matplotlib Figure and Axes
        Scree (left) and cumulative explained variance (right).
    
    
    """
    
    
    chrons = chrons.loc[recon_start:recon_end]
    scaler = StandardScaler()
    X_std = scaler.fit_transform(chrons.values)

    
    pca = PCA(n_components=max_pca)
    scores_array = pca.fit_transform(X_std)

    
    pc_names = [f'PC {i + 1}' for i in range(pca.n_components_)]
    scores = pd.DataFrame(scores_array, index=chrons.index, columns=pc_names)
    loadings = pd.DataFrame(pca.components_, index=pc_names, columns=chrons.columns)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    variance = pd.DataFrame({
        'explained_variance':       pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'singular_values':          pca.singular_values_,
    }, index=pc_names)

    fig, axes = None, None
    
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(np.arange(1, len(pca.explained_variance_ratio_) + 1),
                    pca.explained_variance_ratio_ * 100)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio (%)')
        axes[1].plot(np.arange(1, len(cum_var) + 1), cum_var * 100, marker='o')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance (%)')
        if threshold is not None:
            axes[1].axhline(threshold, ls='--', lw=0.8, color='grey')
        fig.tight_layout()

    
    return scaler, pca, scores, loadings, variance, cum_var, fig, axes





def observed_vs_pca_regression(PCA_scores, observed, number_pca, calib_start, calib_end):
    
    """
    Fit a principal-component regression over the calibration period and score it.

    Parameters
    ----------
    PCA_scores : pandas.DataFrame
        Year-indexed PC scores with columns named ``'PC 1'``, ``'PC 2'``, …
        (as produced by :func:`select_PCA`).
    
    observed : pandas.Series
        Instrumental target (e.g. streamflow, precipitation) over the calibration period,
        aligned to the same year index as "best_PCA".
    
    number_pca : int
        Number of leading principal components to include as predictors
        (``PC 1`` through ``PC number_pca``).
    
    calib_start, calib_end : int
        First and last year of the calibration period (inclusive), used to
        slice ``PCA_scores`` on its year index.

    Returns
    -------
    pipe : sklearn.pipeline.Pipeline
        The fitted model (standardization followed by linear regression),
        trained on the full calibration period and then passed to the next Function.
    
    recon : numpy.ndarray
        In-sample predictions of the target over the calibration period.
    
    r2_in : float
        In-sample R², measuring fit on the calibration data itself. 
    
    cv_scores : numpy.ndarray
        Cross-validated R² scores, one per fold, from repeated k-fold
        cross-validation (5 splits, 10 repeats). Their mean and standard
        deviation summarize generalization skill.

    """
    
    pc_cols = [f'PC {i + 1}' for i in range(number_pca)]
    X = PCA_scores[pc_cols].loc[calib_start:calib_end]

    
    pipe = Pipeline([('lr', LinearRegression())])
    pipe.fit(X, observed)
    recon = pipe.predict(X)

    
    r2_in = r2_score(observed, recon)
    print(f"In-sample R² (first {number_pca} PCs): {r2_in:.4f}")

    
    cv1 = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    cv_scores = cross_val_score(pipe, X, observed, scoring='r2', cv=cv1, n_jobs=-1)
    print(f"Cross-validated R² (first {number_pca} PCs): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipe, recon, r2_in, cv_scores




def pca_regression_reconstruction(PCA_scores, number_pca, recon_start, recon_end, pipe):
    
    """
    Generate the full-period reconstruction from a fitted PCA regression model.

    Parameters
    ----------
    PCA_scores : pandas.DataFrame
        Year-indexed PC scores with columns ``'PC 1'``, ``'PC 2'``, … (as produced
        by :func:`find_best_set_of_PCA`), covering the full reconstruction period.
    
    number_pca : int
        Number of leading principal components to use as predictors — must match
        the count ``pipe`` was fitted on.
    
    recon_start, recon_end : int
        First and last year of the reconstruction period (inclusive), used to
        slice ``PCA_scores`` on its year index.
    
    pipe : sklearn.pipeline.Pipeline
        A model already fitted on the calibration period, as returned by
        :func:`observed_vs_pca_regression`. No fitting is performed here; the
        model is only applied.

    Returns
    -------
    full_recon : pandas.Series
        Reconstruction over the full period.
    
    """
    
    pc_cols = [f'PC {i + 1}' for i in range(number_pca)]
    X = PCA_scores[pc_cols].loc[recon_start:recon_end]

    full_recon = pd.Series(pipe.predict(X), index=X.index, name='pca_regression')
    return full_recon


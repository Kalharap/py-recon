import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



def find_best_set_of_chrons(chrons, observed, max_chrons, calib_start, calib_end, direction="forward", plot=True):
    
    """Select the best-performing subset of chronologies by sequential feature selection.

    Parameters
    ----------
    chrons : pandas.DataFrame
        Tree-ring chronologies over the calibration period, indexed by year.
        
    observed : pandas.Series
        observed target (e.g. streamflow, precipitation) over the calibration period,
        aligned to the same year index as "chrons".
   
   max_chrons : int
        Maximum subset size to evaluate. The search runs over subset sizes
        k = 1, 2, ..., max_chrons. should be equal or less than total number of chrons.
    
    direction : {"forward", "backward"}, optional
        Direction of the sequential selection. "forward " (default) starts
        from no chronologies and adds them one at a time; "backward" starts
        from all chronologies and removes them one at a time.
    
    plot : bool, optional
        If "true" (default), plot the cross-validated score against the number
        of selected chronologies.

   
   Returns
    -------
    number_of_chrons : int
        Number of chronologies selected from sequential feature selection.
    
    best_chrons : list of str
        Column names of the chronologies in the best-scoring subset.
    
    best_score : float
        Mean cross-validated score of the best subset (negative mean squared
        error; higher, i.e. closer to zero, is better).
   
    best_std : float
        Standard deviation of the cross-validated score across folds for the
        best subset.
    
    results : list of tuple
        (number of selected chrons, selected_chrons, mean_score, std_score)


    """

    chrons = chrons.loc[calib_start:calib_end]
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    
    results = []
    best_tuple = None

    for k in range(1, max_chrons + 1):
        sfs = SequentialFeatureSelector(
            pipe,
            n_features_to_select=k,
            direction= direction ,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1
        )

        sfs.fit(chrons, observed)
        sel = list(chrons.columns[sfs.get_support()])
        scores = cross_val_score(pipe, chrons[sel], observed, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
        mean_score, std_score = scores.mean(), scores.std()
        results.append((k, sel, mean_score, std_score))
        
        if (best_tuple is None) or (mean_score > best_tuple[2]):
            best_tuple = (k, sel, mean_score, std_score)
    
    number_of_chrons, best_chrons, best_score, best_std = best_tuple

    print("\n=== Best subset (sequential selection) ===")
    print(f"Direction: {direction}")
    print(f"Number of chrons : {number_of_chrons}")
    print(f"Best chrons: {best_chrons}")
    print(f"CV neg-MSE: {best_score:.4f} ± {best_std:.4f}")
    
    if plot:
        ks = [r[0] for r in results]
        scores_plot = [r[2] for r in results]
        plt.figure()
        plt.plot(ks, scores_plot, marker='o')
        plt.xlabel('Number of Selected Features (k)')
        plt.ylabel('CV neg-MSE')
        plt.title('Sequential Feature Selection')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    
    return number_of_chrons, best_chrons, best_score, best_std, results




def observed_vs_stepwise_regression(best_chrons, observed, calib_start, calib_end):
    
    
    """

    Fits a standardized linear regression on the selected chronologies over the
    calibration period and evaluates how well it reproduces the observed target.

    Parameters
    ----------
    best_chrons : pandas.DataFrame
        Selected tree-ring chronologies over the calibration period, indexed by year.
    
    observed : pandas.Series
        Instrumental target (e.g. streamflow, precipitation) over the calibration period,
        aligned to the same year index as "best_chrons".
        
    calib_start, calib_end : int
        First and last year of the calibration period (inclusive), used to
        slice ``best_chrons`` on its year index.

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
    best_chrons = df[best_chrons].loc[calib_start:calib_end]
    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    pipe.fit(best_chrons, observed)
    recon = pipe.predict(best_chrons)
    
    r2_in = r2_score(observed, recon)
    print(f"In-sample R² : {r2_in:.4f}")
    
    cv1 = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    cv_scores = cross_val_score(pipe, best_chrons, observed, scoring='r2', cv=cv1, n_jobs=-1)
    print(f"Cross-validated R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
   
    return pipe, recon, r2_in, cv_scores




def stepwise_regression_reconstruction(best_chrons,recon_start, recon_end, pipe):
    
    """Generate the full-period streamflow reconstruction from a fitted model.

    Parameters
    ----------
    paleo_chrons : pandas.DataFrame
        Tree-ring chronologies over the full reconstruction period, indexed by
        year. Must contain the same chronology columns, in the same order, that
        ``pipe`` was fitted on.
        
    recon_start, recon_end : int
        First and last year of the reconstruction period (inclusive), used to
        slice ``df[best_chrons]`` on its year index.
   
    pipe : sklearn.pipeline.Pipeline
        A model already fitted on the calibration period, as returned by
        :func:`observed_vs_stepwise_regression`. No fitting is performed here, the model is only applied.

    Returns
    -------
    full_recon : pandas.Series
        Reconstruction over the full period.


    """
    paleo_chrons = df[best_chrons].loc[recon_start:recon_end]
    full_recon = pd.Series(pipe.predict(paleo_chrons), index=paleo_chrons.index, name='stepwise_regression')

    return full_recon
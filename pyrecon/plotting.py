#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_observed_vs_reconstructed(recon, observed, calib_start, calib_end,
                                   ylabel='Streamflow (MAF)', xlabel='Year',
                                   title='Observed vs Reconstructed Streamflow'):
    years = np.arange(calib_start, calib_end + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, recon, label='Reconstructed')
    ax.plot(years, observed, label='Observed')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    return ax



def plot_full_reconstruction(full_recon, recon_start, recon_end,
                             smooth=False, smooth_years=10,
                             ylabel='Streamflow (MAF)', xlabel='Year',
                             title='Reconstructed Streamflow'):

    s = full_recon.loc[recon_start:recon_end]
    years = s.index

    fig, ax = plt.subplots(figsize=(10, 5))

    if smooth:
        ax.plot(years, s.values, color='steelblue', lw=0.8, alpha=0.4,
                label='Reconstructed (annual)')

        roll = s.rolling(window=smooth_years, center=True, min_periods=1).mean()

        ax.plot(years, roll.values, color='steelblue', lw=2.0,
                label=f'{smooth_years}-yr centered mean')
    else:
        ax.plot(years, s.values, label='Reconstructed', color='steelblue')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    return ax


def reconstruction_stats(full_recon, recon_start, recon_end, name=None):
    """
    Summary statistics for a reconstruction over a given window.

    Parameters
    ----------
    full_recon : pandas.Series
        Reconstructed series, year-indexed.
    recon_start, recon_end : int
        First and last years (inclusive) to summarize.
    name : str, optional
        Row label. Defaults to the Series name, else 'reconstruction'.

    Returns
    -------
    stats : pandas.DataFrame
        One row: Mean, SD, Skewness, r(1), Minimum, Maximum.
    """
    s = full_recon.loc[recon_start:recon_end].dropna()

    label = name or full_recon.name or "reconstruction"

    stats = pd.DataFrame(
        {
            "Mean":     [s.mean()],
            "SD":       [s.std()],                 
            "Skewness": [s.skew()],
            "r(1)":     [s.autocorr(lag=1)],       
            "Minimum":  [s.min()],
            "Maximum":  [s.max()],
        },
        index=[label],
    )
    return stats


import matplotlib.pyplot as plt


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



def plot_full_reconstruction(full_recon, recon_start, recon_end, ylabel='Streamflow (MAF)',
                             xlabel='Year',title ='Reconstructed Streamflow'):

    years = np.arange(recon_start, recon_end + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, recon, label='Reconstructed', color='steelblue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return ax
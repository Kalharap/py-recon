"""pyrecon — calibration-verification streamflow reconstruction from tree-ring chronologies."""


from pyrecon.stepwise import (
    find_best_set_of_chrons,
    observed_vs_stepwise_regression,
    stepwise_regression_reconstruction,
)
from pyrecon.pca import (
    find_best_set_of_PCA,
    observed_vs_pca_regression,
    pca_regression_reconstruction,
)
from pyrecon.KNN import (
    knn_reconstruct,
    loocv_knn_mean,

)
from pyrecon.plotting import (
    plot_observed_vs_reconstructed,
    plot_full_reconstruction,
)

__version__ = "0.0.1"


__all__ = [
    # stepwise
    "find_best_set_of_chrons",
    "observed_vs_stepwise_regression",
    "stepwise_regression_reconstruction",
    # pca
    "find_best_set_of_PCA",
    "observed_vs_pca_regression",
    "pca_regression_reconstruction",
    # knn
    "select_pcs_knn",
    "observed_vs_knn",
    "knn_reconstruct",
    # plotting
    "plot_observed_vs_reconstructed",
    "plot_full_reconstruction",
]
# experiments_ols.py
import numpy as np
from sklearn.model_selection import train_test_split

# Use YOUR modules
from .data_utils import polynomial_features, standardize_train_test, add_intercept
from .metrics_utils import mse
from .closed_form import ols_theta




def compute_train_test_mse_vs_degree(
    x, y, degrees, test_size=0.2, seed=3155
):
    """
    Takes raw 1D arrays x and y, loops over 'degrees' and for each degree:
      1) builds polynomial features (no intercept),
      2) splits into train/test (same 'seed' each time),
      3) standardizes using train stats, then adds a manual intercept column,
      4) fits OLS by pseudo-inverse on TRAIN (X_tr) only,
      5) returns per-degree Train and Test MSE.

    Parameters
    ----------
    x : array-like, shape (N,)
    y : array-like, shape (N,)
    degrees : iterable[int]  (e.g. range(1, 16))
    test_size : float        (default 0.2)
    seed : int               (default 3155)

    Returns
    -------
    degrees_list : list[int]
    train_mse_list : list[float]
    test_mse_list  : list[float]
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    degs = list(degrees)

    train_mse_list = []
    test_mse_list  = []

    for d in degs:
        X = polynomial_features(x, d, intercept=False)

        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=seed
        )

        # Standardize on train; apply to test; then add intercept col
        X_tr, X_te, mu, std = standardize_train_test(X_tr_raw, X_te_raw)
        X_tr = add_intercept(X_tr)
        X_te = add_intercept(X_te)

        # OLS closed-form on standardized+intercept design
        theta = ols_theta(X_tr, y_tr)
       

        train_mse_list.append(mse(y_tr, X_tr @ theta))
        test_mse_list.append(mse(y_te, X_te @ theta))

    return degs, train_mse_list, test_mse_list


def kfold_cv_mse_vs_degree(
    x, y, degrees, k=10, seed=3155
):
    """
    Optional utility: compute simple K-fold CV MSE per degree for OLS.
    We still standardize per fold and add a manual intercept column before
    fitting pseudo-inverse on the training fold.

    Returns
    -------
    degrees_list : list[int]
    cv_mse_mean : list[float]       # mean across folds (per degree)
    cv_mse_all  : list[list[float]] # raw per-fold MSEs (per degree)
    """
    from resampling import kfold_indices  # your module

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    degs = list(degrees)

    folds = kfold_indices(len(y), k, shuffle=True, seed=seed)
    cv_mse_mean = []
    cv_mse_all  = []

    for d in degs:
        Xd = polynomial_features(x, d, intercept=False)
        fold_mses = []
        for i in range(k):
            te_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            X_tr_raw, X_te_raw = Xd[tr_idx], Xd[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            X_tr, X_te, mu, std = standardize_train_test(X_tr_raw, X_te_raw)
            X_tr = add_intercept(X_tr)
            X_te = add_intercept(X_te)

            theta = np.linalg.pinv(X_tr) @ y_tr
            fold_mses.append(mse(y_te, X_te @ theta))

        cv_mse_all.append(fold_mses)
        cv_mse_mean.append(float(np.mean(fold_mses)))

    return degs, cv_mse_mean, cv_mse_all

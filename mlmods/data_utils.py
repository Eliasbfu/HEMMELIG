# data_utils.py
import numpy as np
from sklearn.model_selection import train_test_split

def runge(x):
    """Return the classic Runge function 1/(1+25x^2)."""
    x = np.asarray(x)
    return 1.0 / (1.0 + 25.0 * x**2)

def polynomial_features(x, degree, intercept=False):
    """
    Build [x, x^2, ..., x^degree]. If intercept=True, prepend a 1-column.
    """
    x = np.asarray(x).reshape(-1, 1)
    X = np.hstack([x**p for p in range(1, degree+1)])
    if intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X

def standardize_train_test(
    X_tr, X_te, y_tr=None, y_te=None,
    mode="zscore",          # "zscore" | "center" | "none"
    center_y=False,
    eps=1e-12,
):
    """
    Standardize/center using TRAIN statistics only.

    Returns:
      if y_* is None:
        X_tr_t, X_te_t, X_mu, X_sd
      else:
        X_tr_t, X_te_t, y_tr_t, y_te_t, stats
        where stats={"mode","X_mu","X_sd","y_mu","y_sd"}
    """
    X_tr = np.asarray(X_tr, float)
    X_te = np.asarray(X_te, float) if X_te is not None else None

    X_mu = X_tr.mean(axis=0, keepdims=True)
    if mode == "zscore":
        X_sd = X_tr.std(axis=0, keepdims=True) + eps
        X_tr_t = (X_tr - X_mu) / X_sd
        X_te_t = (X_te - X_mu) / X_sd if X_te is not None else None
    elif mode == "center":
        X_sd = np.ones_like(X_mu)
        X_tr_t = X_tr - X_mu
        X_te_t = X_te - X_mu if X_te is not None else None
    elif mode == "none":
        X_sd = np.ones_like(X_mu)
        X_tr_t, X_te_t = X_tr, X_te
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if y_tr is None:
        # old 4-tuple for existing code that didn’t pass y
        return X_tr_t, X_te_t, X_mu, X_sd

    # y handling only if provided
    y_tr = np.asarray(y_tr, float)
    y_te = np.asarray(y_te, float) if y_te is not None else None
    if center_y:
        y_mu = float(y_tr.mean())
        if mode == "zscore":
            y_sd = float(y_tr.std() + eps)
            y_tr_t = (y_tr - y_mu) / y_sd
            y_te_t = (y_te - y_mu) / y_sd if y_te is not None else None
        else:
            y_sd = 1.0
            y_tr_t = (y_tr - y_mu)
            y_te_t = (y_te - y_mu) if y_te is not None else None
    else:
        y_mu, y_sd = 0.0, 1.0
        y_tr_t, y_te_t = y_tr, y_te

    stats = {"mode": mode, "X_mu": X_mu, "X_sd": X_sd, "y_mu": y_mu, "y_sd": y_sd}
    return X_tr_t, X_te_t, y_tr_t, y_te_t, stats

def apply_standardization(X_new, stats):
    """Apply the SAME transform captured in `stats` to new X."""
    X_new = np.asarray(X_new, float)
    if stats["mode"] == "zscore":
        return (X_new - stats["X_mu"]) / stats["X_sd"]
    if stats["mode"] == "center":
        return X_new - stats["X_mu"]
    return X_new  # "none"

def add_intercept(X):
    X = np.asarray(X, float)
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])
    
def make_dataset(
    N, degree, seed=3155, noise_std=1.0,
    mode="zscore", center_y=False, split=0.2, x_sampling="linspace"
):
    """
    Returns:
      (X_tr_raw, X_te_raw, y_tr, y_te),
      (X_tr, X_te, mu, std)
    where X_tr/X_te have an intercept column appended AFTER preprocessing.
    NOTE: y_tr/y_te are returned as column vectors to match your trainers.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=N) if x_sampling == "uniform" else np.linspace(-1.0, 1.0, N)
    y = runge(x) + rng.normal(0.0, noise_std, size=N)

    X = polynomial_features(x, degree, intercept=False)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=split, shuffle=True, random_state=seed
    )

    # preprocess with TRAIN stats only
    X_tr_t, X_te_t, y_tr_t, y_te_t, stats = standardize_train_test(
        X_tr_raw, X_te_raw, y_tr, y_te, mode=mode, center_y=center_y
    )

    # add intercept AFTER preprocessing
    X_tr = add_intercept(X_tr_t)
    X_te = add_intercept(X_te_t)

    # keep backward-compatible tuple shape
    return (
        (X_tr_raw, X_te_raw, y_tr_t.reshape(-1, 1), y_te_t.reshape(-1, 1)),
        (X_tr, X_te, stats["X_mu"], stats["X_sd"])
    )


def prepare_design_from_indices(x, y, degree, tr_idx, te_idx, mode="center", center_y=False):
    """
    Build polynomial features (no intercept), standardize with TRAIN stats only,
    then append intercept after. Returns X_tr, X_te, y_tr, y_te (y 1D here is fine for OLS/plots).
    """
    X_all = polynomial_features(x, degree, intercept=False)
    X_tr_raw, X_te_raw = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # use 4-tuple (y untouched); pass y_* if you ever want to transform y
    X_tr_t, X_te_t, _, _ = standardize_train_test(X_tr_raw, X_te_raw, mode=mode)
    X_tr = add_intercept(X_tr_t)
    X_te = add_intercept(X_te_t)
    return X_tr, X_te, y_tr, y_te

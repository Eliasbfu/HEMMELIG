# metrics_utils.py
import numpy as np

def mse(y, yhat):
    """Plain mean squared error."""
    y = np.asarray(y).reshape(-1,1)
    yhat = np.asarray(yhat).reshape(-1,1)
    return float(np.mean((y - yhat)**2))

def r2_score(y, yhat):
    """Coefficient of determination (R²)."""
    y = np.asarray(y).reshape(-1,1)
    yhat = np.asarray(yhat).reshape(-1,1)
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    return 1.0 - ss_res / (ss_tot + 1e-18)

def max_abs_diff(a, b):
    """Max elementwise absolute difference between arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

def eta_safe_from_hessian(X, lam=0.0):
    """
    Conservative 1/L step size for (1/n)||y-Xθ||^2 + lam*penalty.
    L ≈ (2/n)*σ_max(X)^2 + 2*lam. Intercept penalty is independent of X here.
    """
    n = X.shape[0]
    smax = np.linalg.svd(X, compute_uv=False)[0]
    L = (2.0/n) * (smax**2) + 2.0 * lam
    return 1.0 / (L + 1e-18)

def train_mse_over_traj(X, y, traj, mse_fn):
    """
    Compute plain train MSE along a parameter trajectory list.
    """
    return [mse_fn(y, X @ th) for th in traj]
# closed_form.py
import numpy as np

def ols_closed_form(X, y):
    """
    Solve min (1/n)||y - Xθ||^2 via pseudoinverse (same design as training).
    """
    return np.linalg.pinv(X) @ y

def ridge_closed_form(X, y, lam):
    """
    Solve min (1/n)||y - Xθ||^2 + lam*||θ_{1:}||^2 with NO penalty on intercept.
    System: (X^T X + n*lam*D)θ = X^T y, where D=diag(0,1,1,...).
    """
    n, p1 = X.shape
    D = np.eye(p1); D[0,0] = 0.0
    return np.linalg.solve(X.T @ X + n * lam * D, X.T @ y)
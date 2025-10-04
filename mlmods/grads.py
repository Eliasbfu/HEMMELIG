# grads.py
import numpy as np

def ols_grad_full_builder(X, y):
    """
    Full-batch OLS gradient for (1/n)||y - Xθ||^2.
    Returns a function grad(theta) -> (p,1)
    """
    n = X.shape[0]
    def grad(theta):
        return (2.0/n) * (X.T @ (X @ theta - y))
    return grad

def ols_grad_minibatch(theta, Xb, yb):
    """Mini-batch OLS gradient with m = |batch|."""
    m = Xb.shape[0]
    return (2.0/m) * (Xb.T @ (Xb @ theta - yb))

def ridge_grad_full_builder(X, y, lam, intercept_free=True):
    """
    Full-batch Ridge gradient for (1/n)||y - Xθ||^2 + lam * ||θ_{1:}||^2.
    """
    n = X.shape[0]
    def grad(theta):
        g = (2.0/n) * (X.T @ (X @ theta - y))
        if lam != 0.0:
            g_reg = 2.0 * lam * theta
            if intercept_free:
                g_reg[0] = 0.0
            g = g + g_reg
        return g
    return grad

def ridge_grad_minibatch(theta, Xb, yb, lam, intercept_free=True):
    """Mini-batch Ridge gradient (same penalty handling as full)."""
    m = Xb.shape[0]
    g = (2.0/m) * (Xb.T @ (Xb @ theta - yb))
    if lam != 0.0:
        g_reg = 2.0 * lam * theta
        if intercept_free:
            g_reg[0] = 0.0
        g = g + g_reg
    return g

def lasso_grad_full_builder(X, y, lam, intercept_free=True):
    """
    Full-batch LASSO subgradient for (1/n)||y - Xθ||^2 + lam * ||θ_{1:}||_1.
    Non-differentiable at 0; we use subgradient sign(θ). Intercept not penalized if intercept_free.
    """
    n = X.shape[0]
    def grad(theta):
        g = (2.0/n) * (X.T @ (X @ theta - y))
        if lam != 0.0:
            s = np.sign(theta)
            if intercept_free:
                s[0] = 0.0
            g = g + lam * s
        return g
    return grad

def lasso_grad_minibatch(theta, Xb, yb, lam, intercept_free=True):
    """Mini-batch LASSO subgradient (see full builder for details)."""
    m = Xb.shape[0]
    g = (2.0/m) * (Xb.T @ (Xb @ theta - yb))
    if lam != 0.0:
        s = np.sign(theta)
        if intercept_free:
            s[0] = 0.0
        g = g + lam * s
    return g
# optims.py
import numpy as np

def gd(grad, th0, lr=1e-2, iters=1000):
    """Plain full-batch GD. Returns (theta_final, trajectory)."""
    th = th0.copy(); traj = [th.copy()]
    for _ in range(iters):
        th -= lr * grad(th)
        traj.append(th.copy())
    return th, traj

def gd_momentum(grad, th0, lr=1e-2, iters=1000, beta=0.9):
    """GD with Polyak momentum."""
    th = th0.copy(); v = np.zeros_like(th); traj = [th.copy()]
    for _ in range(iters):
        g = grad(th)
        v = beta*v + (1.0 - beta)*g
        th -= lr * v
        traj.append(th.copy())
    return th, traj

def adagrad(grad, th0, lr=1e-2, iters=1000, eps=1e-8):
    """Full-batch AdaGrad."""
    th = th0.copy(); G = np.zeros_like(th); traj = [th.copy()]
    for _ in range(iters):
        g = grad(th)
        G += g*g
        th -= (lr/(np.sqrt(G) + eps)) * g
        traj.append(th.copy())
    return th, traj

def rmsprop(grad, th0, lr=1e-3, iters=1000, beta=0.9, eps=1e-8):
    """Full-batch RMSprop."""
    th = th0.copy(); S = np.zeros_like(th); traj = [th.copy()]
    for _ in range(iters):
        g = grad(th)
        S = beta*S + (1.0 - beta)*(g*g)
        th -= lr * g / (np.sqrt(S) + eps)
        traj.append(th.copy())
    return th, traj

def adam(grad, th0, lr=1e-3, iters=1000, beta1=0.9, beta2=0.999, eps=1e-8):
    """Full-batch Adam."""
    th = th0.copy(); m = np.zeros_like(th); v = np.zeros_like(th); traj = [th.copy()]
    for t in range(1, iters+1):
        g = grad(th)
        m = beta1*m + (1.0 - beta1)*g
        v = beta2*v + (1.0 - beta2)*(g*g)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        th -= lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(th.copy())
    return th, traj

# --------- SGD core + variants ---------
def _iterate_minibatches(X, y, batch_size, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        bi = idx[start:start+batch_size]
        yield X[bi], y[bi]

def _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, update_rule):
    th = th0.copy()
    traj = [th.copy()]  # snapshot once per epoch
    state = {}
    for ep in range(epochs):
        for Xb, yb in _iterate_minibatches(X, y, batch_size, seed + ep):
            th = update_rule(th, Xb, yb, grad_mb, lr, state)
        traj.append(th.copy())
    return th, traj

def _upd_sgd(th, Xb, yb, grad_mb, lr, state):
    g = grad_mb(th, Xb, yb); return th - lr*g

def _upd_mom(th, Xb, yb, grad_mb, lr, state):
    if "v" not in state: state["v"] = np.zeros_like(th)
    beta = 0.9
    g = grad_mb(th, Xb, yb)
    state["v"] = beta*state["v"] + (1.0 - beta)*g
    return th - lr*state["v"]

def _upd_ada(th, Xb, yb, grad_mb, lr, state):
    if "G" not in state: state["G"] = np.zeros_like(th)
    eps = 1e-8
    g = grad_mb(th, Xb, yb)
    state["G"] += g*g
    return th - (lr/(np.sqrt(state["G"])+eps)) * g

def _upd_rms(th, Xb, yb, grad_mb, lr, state):
    if "S" not in state: state["S"] = np.zeros_like(th)
    beta, eps = 0.9, 1e-8
    g = grad_mb(th, Xb, yb)
    state["S"] = beta*state["S"] + (1.0 - beta)*(g*g)
    return th - lr * g / (np.sqrt(state["S"]) + eps)

def _upd_adam(th, Xb, yb, grad_mb, lr, state):
    if "m" not in state: state["m"] = np.zeros_like(th)
    if "v" not in state: state["v"] = np.zeros_like(th)
    if "t" not in state: state["t"] = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    g = grad_mb(th, Xb, yb)
    state["t"] += 1
    state["m"] = beta1*state["m"] + (1.0 - beta1)*g
    state["v"] = beta2*state["v"] + (1.0 - beta2)*(g*g)
    m_hat = state["m"] / (1.0 - beta1**state["t"])
    v_hat = state["v"] / (1.0 - beta2**state["t"])
    return th - lr * m_hat / (np.sqrt(v_hat) + eps)

def sgd(grad_mb, th0, X, y, lr, epochs, batch_size, seed=0):
    return _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, _upd_sgd)

def sgd_momentum(grad_mb, th0, X, y, lr, epochs, batch_size, seed=0):
    return _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, _upd_mom)

def sgd_adagrad(grad_mb, th0, X, y, lr, epochs, batch_size, seed=0):
    return _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, _upd_ada)

def sgd_rmsprop(grad_mb, th0, X, y, lr, epochs, batch_size, seed=0):
    return _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, _upd_rms)

def sgd_adam(grad_mb, th0, X, y, lr, epochs, batch_size, seed=0):
    return _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, _upd_adam)
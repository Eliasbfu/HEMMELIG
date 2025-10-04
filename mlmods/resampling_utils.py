# mlmods/resampling_utils.py
import numpy as np

def bootstrap_indices(n, B, seed=None):
    """
    Return a generator of B bootstrap index arrays of length n.
    Each array samples {0..n-1} with replacement.
    seed (int or None): reproducibility hook.
    """
    rng = np.random.default_rng(seed)
    for _ in range(B):
        yield rng.integers(0, n, size=n)

def kfold_indices(n, k, shuffle=True, seed=None):
    """
    Return a list of k arrays, each the indices of one fold.
    If shuffle=True, indices are shuffled once before splitting.
    Example: folds = kfold_indices(100, 10, shuffle=True, seed=3155)
    """
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    # split as evenly as possible
    return np.array_split(idx, k)

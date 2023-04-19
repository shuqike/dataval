import numpy as np


def error(mem):
    """Evaluate
    """
    if len(mem) < 100:
        return 1.0
    all_vals = (np.cumsum(mem, 0)/np.reshape(np.arange(1, len(mem)+1), (-1,1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)

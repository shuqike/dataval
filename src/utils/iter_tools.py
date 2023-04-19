import numpy as np


def error(history):
    """Evaluate the changes of the most recent 100 updates
    """
    if len(history) < 100:
        return 1.0
    all_vals = (np.cumsum(history, 0)/np.reshape(np.arange(1, len(history)+1), (-1,1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)

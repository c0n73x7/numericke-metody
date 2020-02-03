import numpy as np


def simple_iteration(phi, x0, eps=0.001, max_iter=100, show_progress=False):
    x_vals, abs_err_vals = list(), list()
    x = x0
    for it in range(max_iter):
        x_vals.append(x)
        x = phi(x)
        abs_err_vals.append(np.abs(x - x_vals[-1]))
        if abs_err_vals[-1] < eps:
            x_vals.append(x)
            break
    result = {
        'x_approx': x_vals[-1],
        'abs_err': abs_err_vals[-1],
        'iters': it,
        'x_vals': x_vals,
        'abs_err_vals': abs_err_vals,
    }
    return result

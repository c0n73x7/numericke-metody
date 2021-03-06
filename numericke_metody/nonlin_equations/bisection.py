import numpy as np


def bisection(f, a, b, eps, max_iter=300, show_progress=False):
    '''
    Bisection Method

    Input Params
    ------------
    f .............. input function
    a .............. beginning of interval
    b .............. end of interval
    eps ............ tolerance
    max_iter ....... maximum number of iterations
    show_progress .. print progress of computation

    Output Params
    -------------
    result -> dict
        x_approx .. approximation of solution
        iters ..... number of iterations
        s_vals .... approximations of solution during the computation
    '''
    s_vals = list()
    for it in range(1, max_iter+1):
        s = (a + b) / 2.
        s_vals.append(s)
        if show_progress:
            print(f'Iteration: {it}')
            print(f'x_approx: {s}')
            print()
        if b - a < eps or _is_zero(f(s)):
            break
        if f(a) * f(s) < 0:
            b = s
        else:
            a = s
    result = {
        'x_approx': s_vals[-1],
        'iters': it,
        's_vals': s_vals,
    }
    return result


def _is_zero(val):
    return np.abs(val) < np.finfo(np.float64).eps

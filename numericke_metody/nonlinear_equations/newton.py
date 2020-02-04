import numpy as np


def newton(f, df, x0, eps=0.000001, max_iter=300, show_progress=False):
    '''
    Newton's method

    Input Params
    ------------
    f .............. TODO
    df ............. TODO
    x0 ............. TODO
    eps ............ TODO
    max_iter ....... TODO
    show_progress .. TODO

    Output Params
    -------------
    result -> dict
        x_approx ...... TODO
        abs_err ....... TODO
        iters ......... TODO
        x_vals ........ TODO
        abs_err_vals .. TODO
    '''
    x_vals, abs_err_vals = list(), [None]
    x = x0
    if show_progress:
        print(f'Initial approximation x0 = {x}')
        print()
    for it in range(1, max_iter+1):
        x_vals.append(x)
        x = x - f(x) / df(x)
        abs_err_vals.append(np.abs(x - x_vals[-1]))
        if show_progress:
            print(f'Iteration: {it}')
            print(f'x_approx = {x}, abs_err = {abs_err_vals[-1]}')
            print()
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

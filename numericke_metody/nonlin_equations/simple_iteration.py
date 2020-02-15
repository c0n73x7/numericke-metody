import numpy as np


def simple_iteration(phi, x0, eps=0.001, max_iter=300, show_progress=False):
    '''
    Simple Iteration Method

    Input Params
    ------------
    phi ............ input function phi
    x0 ............. initial approximation
    eps ............ tolerance
    max_iter ....... maximum number of iterations
    show_progress .. print progress of computation

    Output Params
    -------------
    result -> dict
        x_approx ...... approximation of solution
        abs_err ....... absolute error
        iters ......... number of iterations
        x_vals ........ approximations of solution during the computation
        abs_err_vals .. absolute errors during the computation
    '''
    x_vals, abs_err_vals = list(), [None]
    x = x0
    if show_progress:
        print(f'Initial approximation x0 = {x}')
        print()
    for it in range(1, max_iter+1):
        x_vals.append(x)
        x = phi(x)
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

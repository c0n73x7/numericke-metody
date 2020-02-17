import numpy as np


def secant(f, x0, x1, eps=0.001, max_iter=300, show_progress=False):
    '''
    Secant method

    Input Params
    ------------
    f .............. input function
    x0 ............. initial approximation
    x1 ............. initial approximation
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
    x_vals, abs_err_vals = [x0], [np.abs(x1 - x0)]
    x = x1
    if show_progress:
        print(f'Initial approximations: x0 = {x0}, x1 = {x1}')
        print()
    for it in range(1, max_iter+1):
        x_vals.append(x)
        dif = (f(x_vals[-2]) - f(x_vals[-1])) / (x_vals[-2] - x_vals[-1])
        x = x - f(x) / dif
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

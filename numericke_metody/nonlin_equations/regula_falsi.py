import numpy as np


def regula_falsi(f, a, b, delta, max_iter=300, show_progress=False):
    '''
    Regula Falsi Method

    Input Params
    ------------
    f .............. input function
    a .............. beginning of interval
    b .............. end of interval
    delta .......... tolerance
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
        s = a - (f(a) * (b - a)) / (f(b) - f(a))
        s_vals.append(s)
        if show_progress:
            print(f'Iteration: {it}')
            print(f'x_approx: {s}')
            print()
        if np.abs(f(s)) < delta:
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

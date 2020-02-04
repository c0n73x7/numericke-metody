import numpy as np


def regula_falsi(f, a, b, delta, max_iter=300, show_progress=False):
    '''
    Regula Falsi Method

    Input Params
    ------------
    f .............. TODO
    a .............. TODO
    b .............. TODO
    delta .......... TODO
    max_iter ....... TODO
    show_progress .. TODO

    Output Params
    -------------
    result -> dict
        x_approx .. TODO
        s_vals .... TODO
    '''
    s_vals = list()
    for it in range(1, max_iter+1):
        s = a - (f(a) * (b - a)) / (f(b) - f(a))
        s_vals.append(s)
        if show_progress:
            print(f'Iteration: {it}')
            print(f'x_approx: {s_vals[-1]}')
            print()
        if np.abs(f(s)) < delta:
            break
        else:
            if f(a) * f(s) < 0:
                b = s
            else:
                a = s
    result = {
        'x_approx': s_vals[-1],
        's_vals': s_vals,
    }
    return result

import numpy as np


def sor(A, b, x0, omega, eps=0.001, max_iter=100, show_progress=False):
    '''
    SOR method

    Input Params
    ------------
    A .............. coefficient matrix
    b .............. right-hand side vector
    x0 ............. vector of the initial approximation
    omega .......... relaxation parameter from the interval (0, 2)
    eps ............ tolerance
    max_iter ....... maximum number of iterations
    show_progress .. print progress of computation

    Output Params
    -------------
    result -> dict
        x_approx ....... approximation of solution
        norm_err ....... euclidean norm of difference of the last two iterations
        iters .......... number of iterations
        x_vals ......... approximations of solution during the computation
        norm_err_vals .. euclidean norm of difference of two successive iterations
    '''
    D = np.diag(np.diagonal(A))
    U, L = np.triu(A) - D, np.tril(A) - D
    H = np.dot(np.linalg.inv(omega * L + D), (1 - omega) * D - omega * U)
    g = np.dot(np.linalg.inv(omega * L + D), omega * b)
    x_vals, norm_err_vals = list(), [None]
    x = x0
    if show_progress:
        print(f'Initial approximation x0 = {x.flatten()}')
        print()
    for it in range(1, max_iter + 1):
        x_vals.append(x)
        x = np.dot(H, x) + g
        norm_err_vals.append(np.linalg.norm(x_vals[-1] - x))
        if show_progress:
            print(f'Iteration: {it}')
            print(f'x = {x.flatten()}, norm_err = {norm_err_vals[-1]}')
            print()
        if norm_err_vals[-1] < eps:
            x_vals.append(x)
            break
    result = {
        'x_approx': x_vals[-1],
        'norm_err': norm_err_vals[-1],
        'iters': it,
        'x_vals': x_vals,
        'norm_err_vals': norm_err_vals,
    }
    return result


def compute_opt_omega(A):
    '''
    Compute optimal omega for SOR method

    Input Params
    ------------
    A .............. coefficient matrix

    Output Params
    -------------
    omega .......... optimal omega
    '''
    D = np.diag(np.diagonal(A))
    U, L = np.triu(A) - D, np.tril(A) - D
    H = np.dot(np.linalg.inv(-D), L + U)
    eig_max = np.max(np.linalg.eigvals(H))
    return 2 / (1 + np.sqrt(1 - np.power(eig_max, 2)))

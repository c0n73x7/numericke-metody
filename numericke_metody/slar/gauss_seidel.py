import numpy as np


def gauss_seidel_method(A, b, x0, eps=0.001, max_iter=100, show_progress=False):
    '''
    Gauss-Seidelova metoda pro řešení soustavy rovnic Ax=b
    Vstupní parametry
    -----------------
    A .............. matice soustavy
    b .............. vektor pravé strany
    x0 ............. vektor počáteční aproximace
    eps ............ požadovaná přesnost
    max_iter ....... maximální počet iterací
    show_progress .. vypisovat / nevypisovat průběh výpočtu

    Výstupní parametry (slovník)
    ----------------------------
    result keys
        x_approx ....... aproximace řešení
        norm_err ....... euklidovská norma rozdílu posledních dvou iterací
        iters .......... počet iteračních kroků metody
        x_vals ......... aproximace řešení v průběhu výpočtu
        norm_err_vals .. euklidovská norma rozdílu dvou po sobě jdoucích iterací v průběhu výpočtu
    '''
    D = np.diag(np.diagonal(A))
    U, L = np.triu(A) - D, np.tril(A) - D
    H = np.dot(np.linalg.inv(-(L + D)), U)
    g = np.dot(np.linalg.inv(L + D), b)
    x_vals, norm_err_vals = list(), [None]
    x = x0
    for it in range(1, max_iter + 1):
        if show_progress:
            print(f'Iterace: {it}')
            print(f'x = {x.flatten()}, norm_err = {norm_err_vals[-1]}')
            print()
        x_vals.append(x)
        x = np.dot(H, x) + g
        norm_err_vals.append(np.linalg.norm(x_vals[-1] - x))
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

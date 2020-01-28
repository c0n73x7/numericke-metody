import numpy as np


def gauss_seidel_method(A, b, x0, eps=0.001, max_iter=100):
    '''
    Gauss-Seidelova metoda pro řešení soustavy rovnic Ax=b
    Vstupní parametry
    -----------------
    A ......... matice soustavy
    b ......... vektor pravé strany
    x0 ........ vektor počáteční aproximace
    eps ....... požadovaná přesnost
    max_iter .. maximální počet iterací

    Výstupní parametry
    ------------------
    x ......... aproximace řešení
    iters ..... počet iteračních kroků metody
    norm_err .. euklidovská norma rozdílu posledních dvou iterací
    progress .. slovník s celým průběhem výpočtu (klíče x_vals, norm_err_vals)
    '''
    D = np.diag(np.diagonal(A))
    U, L = np.triu(A) - D, np.tril(A) - D
    H = np.dot(np.linalg.inv(-(L + D)), U)
    g = np.dot(np.linalg.inv(L + D), b)
    x_vals, norm_err_vals = list(), [None]
    x = x0
    for it in range(max_iter):
        x_vals.append(x)
        x = np.dot(H, x) + g
        norm_err_vals.append(np.linalg.norm(x_vals[-1] - x))
        if norm_err_vals[-1] < eps:
            x_vals.append(x)
            break
    progress = dict(x_vals=x_vals, norm_err_vals=norm_err_vals)
    return x_vals[-1], it+1, norm_err_vals[-1], progress

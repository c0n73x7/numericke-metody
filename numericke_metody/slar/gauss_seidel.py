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
    progress .. slovník s celým průběhem výpočtu (klíče x, norm_err)
    '''
    D = np.diag(np.diagonal(A))
    U, L = np.triu(A) - D, np.tril(A) - D
    H = np.dot(np.linalg.inv(-(L + D)), U)
    g = np.dot(np.linalg.inv(L + D), b)
    x_list, error_list = list(), [None]
    x = x0
    for it in range(max_iter):
        x_list.append(x)
        x = np.dot(H, x) + g
        error_list.append(np.linalg.norm(x_list[-1] - x))
        if error_list[-1] < eps:
            x_list.append(x)
            break
    progress = dict(x=x_list, norm_err=error_list)
    return x_list[-1], it, error_list[-1], progress

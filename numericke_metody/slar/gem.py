import numpy as np


def gaussian_elimination(A, b):
    '''
    Gaussova eliminační metoda pro řešení soustavy Ax = b

    Vstupní parametry
    -----------------
    A .............. matice soustavy
    b .............. vektor pravé strany

    Výstupní parametry
    ------------------
    x .............. vektor řešení soustavy
    '''
    n = A.shape[0]
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = -A[i, k] / A[k, k]
            for j in range(k + 1, n):
                A[i, j] = A[i, j] + m * A[k, j]
            b[i] = b[i] + m * b[k]
    x = np.zeros(n)
    for i in reversed(range(n)):
        temp = 0
        for j in range(i, n):
            temp = temp + A[i, j] * x[j]
        x[i] = 1 / A[i, i] * (b[i] - temp)
    return x

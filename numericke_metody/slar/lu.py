import numpy as np
from numericke_metody.factorization import lu as lu_factorization


def lu(A, b):
    '''
    LU method

    Input Params
    ------------
    A .............. coefficient matrix
    b .............. right-hand side vector

    Output Params
    -------------
    x .............. solution
    '''
    L, U = lu_factorization(A)
    n = A.shape[0]
    y = np.copy(b)
    for i in range(n):
        for j in range(i):
            y[i] = y[i] - L[i, j] * y[j]
    x = np.copy(y)
    for i in reversed(range(n)):
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x

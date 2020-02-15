import numpy as np


def lu(A):
    '''
    LU decompostion

    Input Params
    ------------
    A .............. square matrix

    Output Params
    -------------
    L .............. lower triangular matrix with ones on the diagonal
    U .............. upper triangular matrix
    '''
    A = np.copy(A)
    n = A.shape[0]
    for k in range(n-1):
        for i in range(k+1, n):
            A[i, k] = A[i, k] / A[k, k]
            for j in range(k+1, n):
                A[i, j] = A[i, j] - A[i, k] * A[k, j]
    I, J = np.eye(n), np.ones(n)
    L_mask, U_mask = np.tril(J) - I, np.triu(J)
    return L_mask * A + I, U_mask * A


def lu_pivoting(A):
    '''
    LU decompostion with partial pivoting

    Input Params
    ------------
    A .............. square matrix

    Output Params
    -------------
    L .............. lower triangular matrix with ones on the diagonal
    U .............. upper triangular matrix
    piv ............ pivoting indices
    '''
    A = np.copy(A)
    n = A.shape[0]
    piv = np.arange(n)
    for k in range(n-1):
        idx = np.argmax(abs(A[k:, k])) + k
        piv[[k, idx]] = piv[[idx, k]]
        A[[k, idx]] = A[[idx, k]]
        for i in range(k+1, n):
            A[i, k] = A[i, k] / A[k, k]
            for j in range(k+1, n):
                A[i, j] = A[i, j] - A[i, k] * A[k, j]
    I, J = np.eye(n), np.ones(n)
    L_mask, U_mask = np.tril(J) - I, np.triu(J)   
    return L_mask * A + I, U_mask * A, piv

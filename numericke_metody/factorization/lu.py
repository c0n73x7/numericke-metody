import numpy as np


def lu(A):
    '''
    LU rozklad bez pivotace

    Vstupní parametry
    -----------------
    A .............. čtvercová matice

    Výstupní parametry
    ------------------
    L .............. dolní trojúhelníková matice s jedničkami na diagonále
    U .............. horní trojúhelníková matice
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
    LU rozklad s částečnou pivotací

    Vstupní parametry
    -----------------
    A .............. čtvercová matice

    Výstupní parametry
    ------------------
    L .............. dolní trojúhelníková matice s jedničkami na diagonále
    U .............. horní trojúhelníková matice
    piv ............ indexy reprezentující přeházení řádků
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

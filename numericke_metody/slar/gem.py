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
    _check_input(A, b)
    n = A.shape[0]
    A, b = np.copy(A), np.copy(b)
    for k in range(n-1):
        for i in range(k+1, n):
            m = A[i, k] / A[k, k]
            for j in range(k+1, n):
                A[i, j] = A[i, j] - m * A[k, j]
            b[i, :] = b[i, :] - m * b[k, :]
    
    x = np.zeros(n)
    b = b.flatten()
    for i in reversed(range(n)):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] = x[i] - A[i, j] * x[j]
        x[i] = x[i] / A[i, i]
    return x


def gaussian_elimination_pivoting(A, b, show_progress=False):
    '''
    Gaussova eliminační metoda s částečnou pivotací pro řešení soustavy Ax = b

    Vstupní parametry
    -----------------
    A .............. matice soustavy
    b .............. vektor pravé strany
    show_progress .. vypisovat / nevypisovat průběh výpočtu

    Výstupní parametry
    ------------------
    x .............. vektor řešení soustavy
    '''
    _check_input(A, b)
    n = A.shape[0]
    A, b = np.copy(A), np.copy(b)
    for k in range(n-1):
        idx = np.argmax(np.abs(A[k:, k])) + k
        A[[k, idx]] = A[[idx, k]]
        b[[k, idx]] = b[[idx, k]]
        if show_progress and idx != k:
            print(f'Měním {idx}. řádek s {k}. řádkem')
            print()

        for i in range(k+1, n):
            m = -A[i, k] / A[k, k]
            for j in range(k+1, n):
                A[i, j] = A[i, j] + m * A[k, j]
            b[i, :] = b[i, :] + m * b[k, :]

    x = np.zeros(n)
    b = b.flatten()
    for i in reversed(range(n)):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] = x[i] - A[i, j] * x[j]
        x[i] = x[i] / A[i, i]
    return x


def _check_input(A, b):
    assert len(A.shape) == 2 and len(b.shape) == 2
    n_A, m_A = A.shape
    n_b, m_b = b.shape
    assert n_A == m_A == n_b and m_b == 1

from time import sleep
from IPython.display import clear_output

import numpy as np
from numericke_metody.factorization import lu as lu_factorization
from .utils import is_upper_triag


def lr_transform(A, max_iter = 25, progress = 0):
    '''
    Finds eigenvalues of square matrix using LR transformation 
    
    Input params
    -----------------
    A .... square numpy array
    max_iter ... maximum number of iterations
    progess .... show animated iterations
            0 ... show nothing
            >0 ... delay between iterations (seconds fractions)
    
    Result
    ----------------------------
    Vector (numpy array) of matrix A eigenvalues
    '''
    
    if progress:
        A_orig = np.copy(A)
    A = A.astype(np.float64)
    k = 0
    while not(is_upper_triag(A)) and k < max_iter:
        L, U = lu_factorization(A)
        A = U @ L
        if progress:
            clear_output(wait=True)
            with np.printoptions(precision=6, suppress=True):
                print(A)
            sleep(progress)
        k += 1
    if progress:
        print('\nVlastní čísla matice A:')
        print(f'        {np.diagonal(A)}')
    
    return np.diagonal(A)


def qr_transform(A, max_iter = 25, progress = 0):
    '''
    Finds eigenvalues of square matrix using QR transformation 
    
    Input params
    -----------------
    A .... square numpy array
    max_iter ... maximum number of iterations
    progess .... show animated iterations
            0 ... show nothing
            >0 ... delay between iterations (seconds fractions)
    
    Result
    ----------------------------
    Vector (numpy array) of matrix A eigenvalues
    '''
    
    if progress:
        A_orig = np.copy(A)
    A = A.astype(np.float64)
    k = 0
    while not(is_upper_triag(A)) and k < max_iter:
        Q, R = np.linalg.qr(A, mode='complete')
        A = R @ Q
        if progress:
            clear_output(wait=True)
            with np.printoptions(precision=6, suppress=True):
                print(A)
            sleep(progress)
        k += 1
    if progress:
        print('\nVlastní čísla matice A:')
        print(f'        {np.diagonal(A)}')
    
    return np.diagonal(A)
        
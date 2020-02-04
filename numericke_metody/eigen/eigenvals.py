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
    
    Output params
    ----------------------------
    Vector (numpy array) of matrix A eigenvalues
    '''
    
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
    
    Output params
    ----------------------------
    Vector (numpy array) of matrix A eigenvalues
    '''
    
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
        
    

def rotate(A, p, q):
    '''
    Zeroes matrix element at given position using Givens rotation
    
    Input params
    -----------------
    A ... square numpy array
    p ... row index
    q ... column index
    
    Output params
    ----------------------------
    Matrix (numpy array) B with zeroed element
    '''
    A = A.astype(np.float64)
    B = np.copy(A)
    r = np.sqrt((A[p,p] - A[q,q])**2 + 4*A[p,q]**2)
    
    if A[p,q] < 0:
        si = -1
    else:
        si = 1
        
    c = np.sqrt(.5 + (A[p,p] - A[q, q]) / (2*r))
    s = si * np.sqrt(.5 - (A[p,p] - A[q, q]) / (2*r))
    
    bpp = (A[p,p] + A[q,q] + r) / 2
    bqq = (A[p,p] * A[q,q] - A[p,q] * A[p,q]) / bpp
    
    for i in range(len(A)):
        if (i != p) and (i != q):
            bip = A[i,p] * c + A[i,q] * s
            biq = -A[i,p] * s + A[i,q] * c
            B[i,p] = B[p,i] = bip
            B[i,q] = B[q,i] = biq
    
    B[p,p] = bpp
    B[q,q] = bqq
    B[p,q] = B[q,p] = 0
    
    return B

    
def jacobi(A, eps=1e-5, max_iter=25, progress=0):
    '''
    Finds eigenvalues of square matrix using Jacobi diagonalization 
    Elements are zeroed in sequence.
    
    Input params
    -----------------
    A .......... square numpy array
    eps ........ absolute tolerance
    max_iter ... maximum number of iterations
    progess .... show animated iterations
            0 .... show nothing
            >0 ... delay between iterations (seconds fractions)
    
    Output params
    ----------------------------
    Vector (numpy array) of matrix A eigenvalues
    '''
    A = A.astype(np.float64)
    m, n = np.shape(A)
    if (m == n) and (np.linalg.norm(A-A.T) < 1e-5):
        k = 0
        err = 10000
        i = 0
        j = 1
        
        while (err > eps) and (k < max_iter):
            if i == n-1:
                i = 0
                j = 1
            #i += 1
            #j = i + 1


            while (err > eps) and (k < max_iter) and (j < n):
                A = rotate(A, i, j)
                err = np.linalg.norm(np.tril(A, -1))
                k += 1
                j += 1
                if progress:
                    clear_output(wait=True)
                    with np.printoptions(precision=6, suppress=True):
                        print(A)
                    sleep(progress)
            
            i += 1
            j = i + 1

            
    if progress:
        print(f'\nPočet iterací: {k}')
        print('Vlastní čísla matice A:')
        print(f'        {np.diagonal(A)}')
        
    return np.diagonal(A)


def jacobi_max(A, eps=1e-5, max_iter=25, progress=0):
    '''
    Finds eigenvalues of square matrix using Jacobi diagonalization.
    Maximum element is zeroed.
    
    
    Input params
    -----------------
    A .......... square numpy array
    eps ........ absolute tolerance
    max_iter ... maximum number of iterations
    progess .... show animated iterations
            0 .... show nothing
            >0 ... delay between iterations (seconds fractions)
    
    Output params
    ----------------------------
    Vector (numpy array) of matrix A eigenvalues
    '''    
    A = A.astype(np.float64)
    m, n = np.shape(A)
    if (m == n) and (np.linalg.norm(A-A.T) < 1e-5):
        k = 0
        err = 10000
        
        while (err > eps) and (k < max_iter):
            i, j = np.unravel_index(np.argmax(np.abs(np.triu(A, 1))), A.shape)
            
            A = rotate(A, i, j)
            err = np.linalg.norm(np.tril(A, -1))
            k += 1
            
            if progress:
                clear_output(wait=True)
                with np.printoptions(precision=6, suppress=True):
                    print(A)
                sleep(progress)
                        
    if progress:
        print(f'\nPočet iterací: {k}')
        print('Vlastní čísla matice A:')
        print(f'        {np.diagonal(A)}')
        
    return np.diagonal(A)






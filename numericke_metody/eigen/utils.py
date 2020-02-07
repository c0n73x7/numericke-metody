import numpy as np

def is_upper_triag(A, eps = 1e-15): 
    for i in range(len(A)): 
        for j in range(i): 
            if(np.abs(A[i, j]) > eps):  
                    return False
    return True
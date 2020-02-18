import numpy as np
import sympy as sym

def str_to_func(s):
    '''
    Makes a numpy function out of string

    Input params
    -----------------
    s .... string with function variable x (for example sin(1/x)).
    
    Output params
    ------------------
    Numpy function.
    
    '''
    x = sym.symbols('x')
    f_sym = sym.sympify(s)
    return sym.lambdify(x,f_sym,'numpy')


def forward_difference(f_str, x0, h):
    '''
    Calculate derivative of a function at point x0 using forward difference
    
    Input params
    -----------------
    f_str .... function (string)
    x0 ....... point to calculate derivative at
    h ........ step
    
    Output params
    ------------------
    Numeric value of f derivative at x0.
    
    '''
    f = str_to_func(f_str)
    return (f(x0+h) - f(x0)) / h


def backward_difference(f_str, x0, h):
    '''
    Calculate derivative of a function at point x0 using backward difference
    
    Input params
    -----------------
    f_str .... function (string)
    x0 ....... point to calculate derivative at
    h ........ step
    
    Output params
    ------------------
    Numeric value of f derivative at x0.
    
    
    '''
    f = str_to_func(f_str)
    return (f(x0) - f(x0-h)) / h


def central_difference(f_str, x0, h):
    '''
   Calculate derivative of a function at point x0 using central difference
    
    Input params
    -----------------
    f_str .... function (string)
    x0 ....... point to calculate derivative at
    h ........ step
    
    Output params
    ------------------
    Numeric value of f derivative at x0.
    
    
    '''
    f = str_to_func(f_str)    
    return (f(x0+h) - f(x0-h)) / (2*h)


def der_richardson(f_str, x0, h_init, N, method='cd'):
    '''
    Calculate derivative of a function at point x0 using Richardson extrapolation
    
    Input params
    -----------------
    f_str .... function (string)
    x0 ....... point to calculate derivative at
    h_init.... base step
    N ........ number of corrections
    method ... used difference
            fd ... forward
            bd ... backward
            cd ... central
    
    Output params (dictionary)
    ----------------------------
    result keys
        f_der ........ numeric value of f derivative at x0
        f_der_vals ... values of used method and all its corrections
    '''
    
    hs = [h_init/(2**i) for i in range(N+1)]
    D = []
    if method == 'cd':
        D.append([central_difference(f_str,x0,h) for h in hs])
        m = 2
    elif method == 'fd': 
        D.append([forward_difference(f_str,x0,h) for h in hs])
        m = 1
    elif method == 'bd': 
        D.append([backward_difference(f_str,x0,h) for h in hs])
        m = 1
    else:
        help(der_richardson)
        result = {
            'f_der': None,
            'f_der_vals': None
        }
        return result
        
    for i in range(N):
        D.append([D[i][j+1] - (D[i][j+1] - D[i][j]) / (1 - 2**(m * (i+1))) for j in range(len(D[i]) - 1)])
        
    result = {
        'f_der': D[-1][-1],
        'f_der_vals': D
    }
    
    return result


def second_central_difference(f_str, x0, h):
    '''
    Calculate derivative of a function at point x0 using second-order central difference
    
    Input params
    -----------------
    f_str .... function (string)
    x0 ....... point to calculate derivative at
    h ........ step
    
    Output params
    ------------------
    Numeric value of f derivative at x0.
    
    '''
    f = str_to_func(f_str)    
    return (f(x0+h) - 2*f(x0) + f(x0-h)) / h**2

def sym_diff(f_str, x0):
    '''
    Calculate symbolically derivative of a function at point x0.
    
    Input params
    -----------------
    f_str .... function (string)
    x0 ....... point to calculate derivative at
    
    Output params
    ------------------
    Value of f derivative at x0.
    
    '''
    x = sym.symbols('x')
    f_sym = sym.sympify(f_str)
    return sym.diff(f_sym, x).subs(x, x0).evalf()

def sym_diff2(f_str, x0):
    '''
    Calculate symbolically second derivative of a function at point x0.
    
    Input params
    -----------------
    f_str .... function (string)
    x0 ....... point to calculate derivative at
    
    Output params
    ------------------
    Value of f second derivative at x0.
    
    '''
    x = sym.symbols('x')
    f_sym = sym.sympify(f_str)
    return sym.diff(f_sym, x, 2).subs(x, x0).evalf()




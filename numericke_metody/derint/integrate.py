import numpy as np
import sympy as sym

from .derivate import str_to_func

def sym_integrate(f_str, a, b):
    '''
    Symbolically calculates integral of a function.
    
    Input params
    -----------------
    f_str .... function (string)
    a ........ lower bound
    b ........ upper bound
    
    Output params
    ------------------
    Value of integral with bounds a and b.
    '''
    x = sym.symbols('x')
    f_sym = sym.sympify(f_str)
    return sym.integrate(f_sym, (x, a, b)).evalf()


def rectangle_integrate(f_str, a, b, N):
    '''
    Calculates value of integral with bounds using rectangle method.
    
    Input params
    -----------------
    f_str .... function (string)
    a ........ lower bound
    b ........ upper bound
    N ........ number of subintervals
    
    Output params
    ------------------
    Numeric value of integral with bounds a and b.
    
    '''
    f = str_to_func(f_str)
    h = (b - a) / N
    result = 0
    
    for i in range(N):
        x = a + (i + .5) * h
        result += f(x) * h
    
    return result


def trapezoid_integrate(f_str, a, b, N):
    '''
    Calculates value of integral with bounds using trapezoid rule.
    
    Input params
    -----------------
    f_str .... function (string)
    a ........ lower bound
    b ........ upper bound
    N ........ number of subintervals
    
    Output params
    ------------------
    Numeric value of integral with bounds a and b.
    
    '''    
    f = str_to_func(f_str)
    h = (b - a) / N
    result = 0
    
    for i in range(N):
          xl = a + i * h
          xr = a + (i+1) * h
          result += (f(xl) + f(xr)) * h/2

    return result


def simpson_integrate(f_str, a, b, N):
    '''
    Calculates value of integral with bounds using Simpsons rule.
    
    Input params
    -----------------
    f_str .... function (string)
    a ........ lower bound
    b ........ upper bound
    N ........ number of subintervals
    
    Output params
    ------------------
    Numeric value of integral with bounds a and b.
    
    '''    
    f = str_to_func(f_str)
    
    if N % 2:
      N += 1
    
    h = (b - a) / N
    result = 0
    
    for i in range(int(N/2)):
          xl = a + 2*i * h
          xc = a + (2*i+1) * h            
          xr = a + 2*(i+1) * h
          result += (f(xr) + 4*f(xc) + f(xl)) * h/3
    
    return result

def richardson_integrate(f_str, a, b, N, cor_num = 3, method = 'rt'):
    '''
    Calculates value of integral with bounds using Richardson extrapolation.
    
    Input params
    -----------------
    f_str .... function (string)
    a ........ lower bound
    b ........ upper bound
    N ........ number of subintervals
    cor_num .. number of corrections
    method ... used rule
            rt ... rectangle rule
            tr ... trapezoid rule
            sm ... simpsons rule
    
    Output params (dictionary)
    ----------------------------
    result keys
        f_int ........ numeric value of integral with bounds a and b
        f_int_vals ... values of used method and all its corrections
    
    '''
    I = []
    if method == 'rt':
        I.append([rectangle_integrate(f_str, a, b, N*(2**i)) for i in range(cor_num+1)])
        m = 2
    elif method == 'tr': 
        I.append([trapezoid_integrate(f_str, a, b, N*(2**i)) for i in range(cor_num+1)])
        m = 2
    elif method == 'sm': 
        I.append([simpson_integrate(f_str, a, b, N*(2**i)) for i in range(cor_num+1)])
        m = 4
    else:
        help(richardson_integrate)
        result = {
            'f_int': None,
            'f_int_vals': None
        }
        return result
        
    for i in range(cor_num):
        I.append([I[i][j+1] - (I[i][j+1] - I[i][j]) / (1 - 2**(m * (i+1))) for j in range(len(I[i])-1)])
    
    result = {
        'f_int': I[-1][-1],
        'f_int_vals': I
    }
    
    return result


def gauss_1_integrate(f_str, a, b, N):
    '''
    Calculates value of integral with bounds using 1-point Gauss quadrature.
    
    Input params
    -----------------
    f_str .... function (string)
    a ........ lower bound
    b ........ upper bound
    N ........ number of subintervals
    
    Output params
    ------------------
    Numeric value of integral with bounds a and b.
    
    '''   
    
    f = str_to_func(f_str)
    h = (b - a) / N
    result = 0
    
    for i in range(N):
        xl = a + i * h
        xr = a + (i+1) * h
        x = (xl + xr) / 2
        result += h * f(x)
        
    return result


def gauss_2_integrate(f_str, a, b, N):
    '''
    Calculates value of integral with bounds using 2-points Gauss quadrature.
    
    Input params
    -----------------
    f_str .... function (string)
    a ........ lower bound
    b ........ upper bound
    N ........ number of subintervals
    
    Output params
    ------------------
    Numeric value of integral with bounds a and b.
    
    '''
    f = str_to_func(f_str)
    h = (b - a) / N
    result = 0
    
    for i in range(N):
        xl = a + i * h
        xr = a + (i+1) * h
        
        x1 = (xl + xr) / 2 - (1/3)**.5 * (xr - xl) / 2
        x2 = (xl + xr) / 2 + (1/3)**.5 * (xr - xl) / 2

        result += h/2 * (f(x1)+f(x2))
        
    return result


def gauss_3_integrate(f_str, a, b, N):
    '''
    Calculates value of integral with bounds using 3-points Gauss quadrature.
    
    Input params
    -----------------
    f_str .... function (string)
    a ........ lower bound
    b ........ upper bound
    N ........ number of subintervals
    
    Output params
    ------------------
    Numeric value of integral with bounds a and b.
    
    '''
    f = str_to_func(f_str)
    h = (b - a) / N
    result = 0
    
    for i in range(N):
        xl = a + i * h
        xr = a + (i+1) * h
        
        x1 = (xl + xr) / 2 - (3/5)**.5 * (xr - xl) / 2
        x2 = (xl + xr) / 2
        x3 = (xl + xr) / 2 + (3/5)**.5 * (xr - xl) / 2

        result += 5/18*h * f(x1) + 8/18*h *f(x2) + 5/18*h *f(x3)
        
    return result
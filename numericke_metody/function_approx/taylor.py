import sympy as sy


def taylor(f, x0, n):
    '''
    Taylor series for function f at point x0
    
    Input params
    ------------
    f .............. sympy function
    x0 ............. point
    n .............. order of the polynomial
    
    Output params
    -------------
    p .............. output polynom
    '''
    p, x = 0, sy.Symbol('x')
    for i in range(n+1):
        p += f.diff(x, i).subs(x, x0) / _factorial(i) * (x - x0)**i
    return p


def _factorial(n):
    if n == 0:
        return 1
    else:
        return n * _factorial(n-1)

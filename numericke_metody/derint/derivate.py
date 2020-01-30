import numpy as np
import sympy as sym

def str_to_func(s):
    '''
    Ze stringu vytvoří numpy fci.

    Vstupní parametry
    -----------------
    s .... string obsahující fci v proměnné x (např. sin(1/x)).
    
    Výstupní parametry
    ------------------
    Numpy funkce.
    
    '''
    x = sym.symbols('x')
    f_sym = sym.sympify(s)
    return sym.lambdify(x,f_sym,'numpy')


def forward_difference(f_str, x0, h):
    '''
    Vypočítá derivaci funkce f v bodě x0 pomocí pravé diference
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    x0 ....... bod v kterém chci derivaci f
    h ........ krok
    
    Výstupní parametry
    ------------------
    Numerická hodnota derivace fce v bodě x0.
    
    '''
    f = str_to_func(f_str)
    return (f(x0+h)-f(x0))/h


def backward_difference(f_str, x0, h):
    '''
    Vypočítá derivaci funkce f v bodě x0 pomocí levé diference
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    x0 ....... bod v kterém chci derivaci f
    h ........ krok
    
    Výstupní parametry
    ------------------
    Numerická hodnota derivace fce v bodě x0.
    
    '''
    f = str_to_func(f_str)
    return (f(x0)-f(x0-h))/h


def central_difference(f_str, x0, h):
    '''
    Vypočítá derivaci funkce f v bodě x0 pomocí centrální diference
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    x0 ....... bod v kterém chci derivaci f
    h ........ krok
    
    Výstupní parametry
    ------------------
    Numerická hodnota derivace fce v bodě x0.
    
    '''
    f = str_to_func(f_str)    
    return (f(x0+h)-f(x0-h))/(2*h)


def der_richardson(f_str, x0, h_init, N, method='cd'):
    '''
    Vypočítá derivaci funkce f v bodě x0 pomocí Richardsonovy extrapolace
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    x0 ....... bod v kterém chci derivaci f
    h_init.... základní krok
    N ........ počet korekcí 
    method ... použitá defierence
            fd ... levá
            bd ... pravá
            cd ... centrální
    
    Výstupní parametry (slovník)
    ----------------------------
    result keys
        f_der ........ Numerická hodnota derivace fce v bodě x0.
        f_der_vals ... hodnoty použité metody a všech korekcí 
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
        D.append([D[i][j+1]-(D[i][j+1]-D[i][j])/(1-2**(m*(i+1))) for j in range(len(D[i])-1)])
        
    result = {
        'f_der': D[-1][-1],
        'f_der_vals': D
    }
    
    return result


def second_central_difference(f_str, x0, h):
    '''
    Vypočítá druhou derivaci funkce f v bodě x0 pomocí tříbodového vzorce
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    x0 ....... bod v kterém chci derivaci f
    h ........ krok
    
    Výstupní parametry
    ------------------
    Numerická hodnota druhé derivace fce v bodě x0.
    
    '''
    f = str_to_func(f_str)    
    return (f(x0+h)-2*f(x0)+f(x0-h))/h**2

def sym_diff(f_str, x0):
    '''
    Vypočte symbolicky hodnotu derivace funkce v bodě.
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    x0 ....... bod v kterém chci derivaci f
    
    Výstupní parametry
    ------------------
    Hodnota derivace fce v bodě x0.
    '''
    x = sym.symbols('x')
    f_sym = sym.sympify(f_str)
    return sym.diff(f_sym, x).subs(x, x0).evalf()

def sym_diff2(f_str, x0):
    '''
    Vypočte symbolicky hodnotu druhé derivace funkce v bodě.
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    x0 ....... bod v kterém chci derivaci f
    
    Výstupní parametry
    ------------------
    Hodnota druhé derivace fce v bodě x0.
    '''
    x = sym.symbols('x')
    f_sym = sym.sympify(f_str)
    return sym.diff(f_sym, x, 2).subs(x, x0).evalf()




import numpy as np
import sympy as sym

from .derivate import str_to_func

def sym_integrate(f_str, a, b):
    '''
    Vypočítá symbolicky určitý integrál funkce.
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    a ........ dolní mez
    b ........ horní mez
    
    Výstupní parametry
    ------------------
    Určitý integrál fce od a do b.
    '''
    x = sym.symbols('x')
    f_sym = sym.sympify(f_str)
    return sym.integrate(f_sym, (x, a, b)).evalf()


def rectangle_integrate(f_str, a, b, N):
    '''
    Vypočítá určitý integrál funkce pomocí obdélníkového pravidla.
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    a ........ dolní mez
    b ........ horní mez
    N ........ počet intervalů dělení
    
    Výstupní parametry
    ------------------
    Numerická hodnota určitého integrálu fce od a do b
    
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
    Vypočítá určitý integrál funkce pomocí lichoběžníkového pravidla.
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    a ........ dolní mez
    b ........ horní mez
    N ........ počet intervalů dělení
    
    Výstupní parametry
    ------------------
    Numerická hodnota určitého integrálu fce od a do b
    
    '''    
    f = str_to_func(f_str)
    h = (b - a) / N
    result = 0
    
    for i in range(N):
          xl = a + i * h
          xr = a + (i+1) * h
          result += (f(xl)+f(xr)) * h/2

    return result


def simpson_integrate(f_str, a, b, N):
    '''
    Vypočítá určitý integrál funkce pomocí Simpsonova pravidla.
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    a ........ dolní mez
    b ........ horní mez
    N ........ počet intervalů dělení
    
    Výstupní parametry
    ------------------
    Numerická hodnota určitého integrálu fce od a do b
    
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
          result += (f(xr)+4*f(xc)+f(xl)) * h/3
    
    return result

def richardson_integrate(f_str, a, b, N, cor_num = 3, method = 'rt'):
    '''
    Vypočítá určitý integrál funkce pomocí Richardsonovy extrapolace
    
    Vstupní parametry
    -----------------
    f_str .... zadaná funkce (string)
    a ........ dolní mez
    b ........ horní mez
    N ........ počet intervalů dělení
    cor_num .. počet korekcí 
    method ... použité pravidlo
            rt ... obdélníkové pravidlo
            tr ... lichoběžníkové pravidlo
            sm ... simpsonovo pravidlo
    
    Výstupní parametry (slovník)
    ----------------------------
    result keys
        f_int ........ numerická hodnota integrálu fce
        f_int_vals ... hodnoty použité metody a všech korekcí 
    '''
    
    #hs = [h_init/(2**i) for i in range(N+1)]
    
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
        I.append([I[i][j+1]-(I[i][j+1]-I[i][j])/(1-2**(m*(i+1))) for j in range(len(I[i])-1)])
    
    result = {
        'f_int': I[-1][-1],
        'f_int_vals': I
    }
    
    return result

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
    f ........ zadaná funkce (string)
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
    f ........ zadaná funkce (string)
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
    f ........ zadaná funkce (string)
    x0 ....... bod v kterém chci derivaci f
    h ........ krok
    
    Výstupní parametry
    ------------------
    Numerická hodnota derivace fce v bodě x0.
    
    '''
    f = str_to_func(f_str)    
    return (f(x0+h)-f(x0-h))/(2*h)

def second_central_difference(f_str, x0, h):
    '''
    Vypočítá druhou derivaci funkce f v bodě x0 pomocí tříbodového vzorce
    
    Vstupní parametry
    -----------------
    f ........ zadaná funkce (string)
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
    f ........ zadaná funkce (string)
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
    f ........ zadaná funkce (string)
    x0 ....... bod v kterém chci derivaci f
    
    Výstupní parametry
    ------------------
    Hodnota druhé derivace fce v bodě x0.
    '''
    x = sym.symbols('x')
    f_sym = sym.sympify(f_str)
    return sym.diff(f_sym, x, 2).subs(x, x0).evalf()




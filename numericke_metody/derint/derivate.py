import numpy as np
import sympy as sym

def finite_difference(f, x0, h, method='cd'):
    '''
    Vypočítá derivaci funkce f v bodě x0 pomocí centrální diference
    
    Vstupní parametry
    -----------------
    f ........ zadaná funkce (string)
    x0 ....... bod v kterém chci derivaci f
    h ........ krok
    method ... použitá metoda:
            fd ... pravá diference
            bd ... levá diference
            cd ... centrální diference
    
    Výstupní parametry
    ------------------
    res ....... výsledná hodnota derivace v bodě x0
    progress .. slovník s celým výpočtem:
        f_der_sym_x0 .... hodnota symbolické derivace v x0
        err ............. chyba |přesné-numerické| řešení
        
    
    TODO
    - vracet podklady pro vykreslení: (???)
            - původní funkci symbolicky,
            - její derivaci jako přímku,
            - numerickou derivaci jako přímku
    '''
    
    x = sym.symbols('x')
    f_sym = sym.sympify(f)
    f_der_sym_x0 = sym.diff(f_sym,x).subs(x,x0).evalf()
    
    f_x0 = f_sym.subs(x,x0).evalf()
    f_x0_plus_h = f_sym.subs(x,x0+h).evalf()
    f_x0_minus_h = f_sym.subs(x,x0-h).evalf()
    if method == 'fd':
        f_der_x0 = (f_x0_plus_h-f_x0)/h
    elif method == 'bd':
        f_der_x0 = (f_x0-f_x0_minus_h)/h
    else:
        f_der_x0 = (f_x0_plus_h-f_x0_minus_h)/h/2
    
    result = {
        'f_der_x0': f_der_x0,
        'f_der_sym_x0' : f_der_sym_x0,
        'err': np.abs(f_der_sym_x0-f_der_x0)
    }
    
    
    return result

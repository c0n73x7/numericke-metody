import numpy as np

from numericke_metody.utils.visualizations import multiplot_vals
from .derivate import *
from .integrate import *

def print_diff_example(fce, x0, h):
    f_diff_sym = sym_diff(fce,x0)
    f_diff_fd = forward_difference(fce,x0,h)
    f_diff_bd = backward_difference(fce,x0,h)
    f_diff_cd = central_difference(fce,x0,h)

    print(f'Přesná hodnota f\'(x):    {f_diff_sym:15.8f}')
    print(f'Numerická hodnota derivace ')
    print(f'   pomocí pravé diference:         {f_diff_fd:15.8f} (chyba: {f_diff_fd - f_diff_sym:12.8f})')
    print(f'   pomocí levé diference:          {f_diff_bd:15.8f} (chyba: {f_diff_bd - f_diff_sym:12.8f})')
    print(f'   pomocí centrální diference:     {f_diff_cd:15.8f} (chyba: {f_diff_cd - f_diff_sym:12.8f})')
    
    
def plot_diff_err(fce, x0, hs):
    f_diff_sym = sym_diff(fce,x0)

    err_forward = [np.abs(forward_difference(fce,x0,h) - f_diff_sym) for h in hs]
    err_backward = [np.abs(backward_difference(fce,x0,h) - f_diff_sym) for h in hs]
    err_central = [np.abs(central_difference(fce,x0,h) - f_diff_sym) for h in hs]

    vals = []

    vals.append({'ys': err_central, 'line': 'o-', 'label': 'Centralní dif.'})
    vals.append({'ys': err_forward, 'line': '>-', 'label': 'Pravá dif.'})
    vals.append({'ys': err_backward, 'line': '<-', 'label': 'Levá dif.'})

    multiplot_vals(hs,
                   vals,
                   'Chyba pro různé diference a různé velikosti kroku $h$',
                   xscale='log',
                   yscale='log')


def print_diff2central_example(fce, x0, h):
    f_diff2_sym = sym_diff2(fce,x0)
    f_diff2_cd = second_central_difference(fce,x0,h)

    print(f'Přesná hodnota f\'\'(x):     {f_diff2_sym:15.8f}')
    print(f'Numerická hodnota derivace ')
    print(f'   pomocí druhé centrální diference: {f_diff2_cd:15.8f} (chyba: {f_diff2_cd - f_diff2_sym:12.8f})')


def print_diff_richardson_example(fce, x0, h, N, method):
    f_diff_sym = sym_diff(fce,x0)
    f_diff_rich = der_richardson(fce, x0, h, N, method=method)['f_der']

    print(f'Přesná hodnota f\'(x):                  {f_diff_sym:15.8f}')
    print(f'Numerická hodnota derivace ')
    print(f'   pomocí Richardsonovy extrapolace: {f_diff_rich:15.8f} (chyba: {f_diff_rich - f_diff_sym:12.8f})')


def print_int_example(fce, a, b, N):
    i_sym = sym_integrate(fce, a, b)
    i_rectangle = rectangle_integrate(fce, a, b, N)
    i_trapezoid = trapezoid_integrate(fce, a, b, N)
    i_simpson = simpson_integrate(fce, a, b, N)

    print(f'Přesná hodnota integrálu:  {i_sym:15.8f}')
    print(f'Numerická hodnota integrálu ')
    print(f'   pomocí obdélníkového pravidla:      {i_rectangle:15.8f} (chyba: {i_rectangle - i_sym:11.8f})')
    print(f'   pomocí lichoběžníkového pravidla:   {i_trapezoid:15.8f} (chyba: {i_trapezoid - i_sym:11.8f})')
    print(f'   pomocí Simpsonova pravidla:         {i_simpson:15.8f} (chyba: {i_simpson - i_sym:11.8f})')


def plot_int_err(fce, a, b, Ns):
    i_sym = sym_integrate(fce, a, b)

    err_rectangle = [rectangle_integrate(fce, a, b, N) - i_sym for N in Ns]
    err_trapezoid = [trapezoid_integrate(fce, a, b, N) - i_sym for N in Ns]
    err_simpson = [simpson_integrate(fce, a, b, N) - i_sym for N in Ns]

    vals = []

    vals.append({'ys': err_rectangle, 'line': 'o-', 'label': 'Obdélníkové pravidlo'})
    vals.append({'ys': err_trapezoid, 'line': '>-', 'label': 'Lichoběžníkové pravidlo'})
    vals.append({'ys': err_simpson, 'line': '<-', 'label': 'Simpsonovo pravidlo'})

    multiplot_vals(Ns, vals,
                   'Chyba pro různé Newtonovy-Cotesovy kvadraturní vzorce a různý počet uzlů $N$',
                  )
    
    
    
def print_int_richardson_example(fce, a, b, N, method, cor_num):
    f_int_sym = sym_integrate(fce, a, b)
    f_int_rich = richardson_integrate(fce, a, b, N, cor_num = cor_num, method=method)['f_int']

    print(f'Přesná hodnota integrálu funkc f na [a,b]: {f_int_sym:15.8f}')
    print(f'Numerická hodnota integrálu ')
    print(f'    pomocí Richardsonovy extrapolace:     {f_int_rich:15.8f} (chyba: {f_int_rich - f_int_sym:12.8f})')


def print_int_gauss_example(fce, a, b, N):
    f_int_sym = sym_integrate(fce, a, b)
    f_int_gauss1 = gauss_1_integrate(fce, a, b, N)
    f_int_gauss2 = gauss_2_integrate(fce, a, b, N)
    f_int_gauss3 = gauss_3_integrate(fce, a, b, N)

    print(f'Přesná hodnota integrálu fuknce f na [a,b]:                 {f_int_sym:15.8f}')
    print(f'Numerická hodnota integrálu ')
    print(f'   s pomocí 1-bodového Gaussova kvadr. vzorce: {f_int_gauss1:15.8f} (chyba: {f_int_gauss1 - f_int_sym:12.8f})')
    print(f'   s pomocí 2-bodového Gaussova kvadr. vzorce: {f_int_gauss2:15.8f} (chyba: {f_int_gauss2 - f_int_sym:12.8f})')
    print(f'   s pomocí 3-bodového Gaussova kvadr. vzorce: {f_int_gauss3:15.8f} (chyba: {float(f_int_gauss3 - f_int_sym):12.8f})')




    
    
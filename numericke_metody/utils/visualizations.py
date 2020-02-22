import numpy as np
import matplotlib.pyplot as plt


def plot_vals(vals, title):
    '''
    Simple plot function
    '''
    plt.figure()
    plt.plot([x for x in range(1, len(vals) + 1)], vals)
    plt.title(title)
    plt.show()

    
def multiplot_vals(xs, vals, title, xscale='linear', yscale='linear', fsize=(12,8)):
    '''
    Plot multiple plots in one
    '''
    plt.figure(figsize=fsize)
    
    for i in range(len(vals)):
        plt.plot(xs, vals[i]['ys'], vals[i]['line'], label=vals[i]['label'])
    plt.xscale(xscale)
    plt.yscale(yscale)    
    plt.title(title)
    plt.legend()
    plt.show()


def nonlineqs_plot(f, vals, x_lims, title, figsize=(10, 6)):
    '''
    Plot for nonlinear equations
    '''
    x1 = np.linspace(x_lims[0], x_lims[1], int(np.abs(x_lims[1] - x_lims[0]) * 100))
    plt.figure(figsize=figsize)
    plt.plot(x1, f(x1), linewidth=2)
    plt.plot(vals, f(np.array(vals)), 'om', alpha=0.4)
    plt.plot(vals[-1], f(vals[-1]), 'or')
    plt.xlim(x_lims)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.title(title)
    plt.show()

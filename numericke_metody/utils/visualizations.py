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
    plt.yscale(xscale)
    plt.xscale(yscale)    
    plt.title(title)
    plt.legend()
    plt.show()
    
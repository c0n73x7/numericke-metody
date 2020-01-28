import matplotlib.pyplot as plt


def plot_vals(vals, title):
    '''
    Simple plot function
    '''
    plt.figure()
    plt.plot(vals[1:])
    plt.title(title)
    plt.show()

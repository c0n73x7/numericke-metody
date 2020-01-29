import matplotlib.pyplot as plt


def plot_vals(vals, title):
    '''
    Simple plot function
    '''
    plt.figure()
    plt.plot([x for x in range(1, len(vals) + 1)], vals)
    plt.title(title)
    plt.show()

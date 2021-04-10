from matplotlib import pyplot as plt
import matplotlib

def amdahls_law(p, s):
    """Speedup relative to proportion parallel
    Amdahl's Law gives an idealized speedup we
    can expect for an algorithm given the proportion
    that algorithm can be parallelized and the speed
    we gain from that parallelization. The best case
    scenario is that the speedup, `s`, is equal to
    the number of processors available.
    Args:
        p: proportion parallel
        s: speed up for the parallelized proportion
    """
    return 1 / ((1-p) + p/s)


def distances_plot(p):
    x = []
    y = []
    for s in range(1, 65537):
        x.append(s)
        y.append(amdahls_law(p, s))
    print(y[-1])
    fig, ax = plt.subplots(1, 1)
    label1 = f'Theoretical Speedup for P={p*100}%'
    ax.plot(x, y, label=label1, color='mediumseagreen')
    ax.axhline(y=146.367, label=f'Jacob\'s Vectorized', color='dodgerblue')
    # ax.axhline(y=223.950, label=f'My Vectorized', color='orangered')
    ax.set_title('Amdahl\'s Law for Centroid Distances')
    ax.set_xlabel('Number of Processors (P) in log scale')
    ax.set_ylabel('SpeedUp')
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    ax.set_xticks([16, 64, 256, 1024, 4096, 16384, 65536])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.savefig('../outputs/final/assignment3/results/question2.png')
    return x, y, label1


def all_3_plots(p, x2=None, y2=None, label2=None):
    x = []
    y = []
    for s in range(1, 262145, 10):
        x.append(s)
        y.append(amdahls_law(p, s))
    print(y[-1])
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, label=f'Theoretical Speedup for P={p*100:.2f}%', color='magenta')
    if x2 is not None and y2 is not None and label2 is not None:
        ax.plot(x2, y2, label=label2, color='mediumseagreen')
    # ax.axhline(y=126.95, label=f'Jacob\'s Vectorized with Numba', color='dodgerblue')
    # ax.axhline(y=245.6, label=f'My Vectorized', color='orangered')
    ax.set_title('Amdahl\'s Law')
    ax.set_xlabel('Number of Processors (P) in log scale')
    ax.set_ylabel('SpeedUp')
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    ax.set_xticks([16, 64, 256, 1024, 4096, 16384, 65536, 262144])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _, top_lim = plt.ylim()
    plt.ylim(top=top_lim+2)
    fig.savefig('../outputs/final/assignment3/results/question4a.png')


# x, y, label1 = distances_plot(p=0.9983)
all_3_plots(p=0.9999) #, x2=x, y2=y, label2=label1)
plt.show()
